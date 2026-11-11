"""Plan v3 stage S2: export BEAT's stored per-trial outputs to beat_audit input files.

    .venv\\Scripts\\python.exe scripts/audit_cohorts.py --config configs/audit.yaml

Writes, without running any audit (docs/plan_v3.md §3 "Input", §4 "Rungs exported per cohort"):
  artifacts/audit/inputs/{cohort}/{features|commands}_*.npz  one file per rung, tau and model seed
  artifacts/audit/inputs/{cohort}/*.trials.csv  source-trial key of each exported row (sidecar)
  results/audit/export_manifest.json  hashes, shapes, users, days, rows, drops and sources
  results/audit/utility.csv           deployment-day task figures per rung, tau and model seed
  results/audit/export_checks.json    comparisons with the studies' stored outputs and numbers
Yang2025 days 2-3 apply study M's stored final models, through M's own functions, to covariances
recomputed from data/processed/yang2025; nothing is refitted and nothing is written under
results/exp_* or artifacts/exp_*. Every configured destination must resolve under
artifacts/audit/ or results/audit/; otherwise the run stops before writing anything.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beat.common import write_csv, write_json  # noqa: E402
from beat.exp_b import _task_rows  # noqa: E402
from beat.exp_l import finite_trials, valid_covariances  # noqa: E402
from beat.exp_m import (CHECKSUMS, check_data, day_stream, load_cfg, load_oof,  # noqa: E402
                        torch_threads, versions)
from beat.verify import blob_sha1  # noqa: E402
from beat.yang2025 import load_trials, read_checksums, sha256_file  # noqa: E402

KEY = ["subject", "day", "run", "trial"]
SCHEMA = dict(
    outputs="float64 [N, d] transmitted features, or int64 [N] command codes 0..C-1",
    user="int64 [N] dataset subject number (pseudonymous)",
    day="int64 [N] recording day in time order (labels in each export's days)",
    segment="int64 [N] disjoint part of a day (labels in each export's segments)",
    order="int64 [N] 0-based position in time within user and day among exported rows",
    target="int64 [N] cue: 0 = left hand, 1 = right hand",
    C="int64 scalar, commands only: declared alphabet size, 3")
UTILITY = ("Deployment day(s) only, per user then mean over users (each user weight 1): task_ba = "
           "mean of the two class recalls of the argmax decision of the rung's decoder; "
           "hold_rate, correct_move_rate and wrong_move_rate = holds, moves toward the cue and "
           "moves away from it over all of the user's trials in the export.")
DROPS = {
    ("bnci2014_001", "features"): "B's quality mask (beat.data.quality_mask): artifact-flagged and "
                                  "non-finite trials excluded, as in B's attacks and task figures",
    ("bnci2014_001", "commands"): "K's streams: calibration day = B's cross-fitted logits of "
                                  "quality-masked session-T trials; deployment day keeps every "
                                  "trial (C's rule: artifact flags never drop rows)",
    ("lee2019", "features"): "L: non-finite trials excluded (beat.exp_l.finite_trials); the "
                             "dataset has no artifact flags",
    ("lee2019", "commands"): "K's streams: finite trials (cal_oof = L's cross-fitted session-1 "
                             "logits, dep = finite session-2 trials)",
    ("yang2025", "features"): "M: non-finite trials excluded (finite_trials on day 1, finite "
                              "representation rows on days 2-3, as in M1)",
    ("yang2025", "commands"): "M's make_stream: trials with non-finite logits or finite = False "
                              "excluded"}
OWNED = tuple((ROOT / p).resolve() for p in ("artifacts/audit", "results/audit"))
NOT_OWNED = ((ROOT / "results/audit/validation.json").resolve(),)  # written by the audit tool
DESTINATIONS = ("inputs_dir", "manifest", "utility", "checks")
TRIAL_KEYS = dict(
    file="{export stem}.trials.csv next to each export; row i describes row i of the export",
    columns=dict(row="0-based row of the export", user="export user", day="export day",
                 order="export order (renumbered after filtering, so not a cross-stream key)",
                 source_session="source study's session", source_run="source study's run",
                 source_trial="source study's trial"),
    join="put two exports of a cohort on common trial support by joining on (user, "
         "source_session, source_run, source_trial), never on (user, day, order)",
    source_session=dict(
        bnci2014_001="B's session label T or E; run and trial as stored in "
                     "artifacts/exp_b/final/*/seed*_outputs.npz (and K's command records)",
        lee2019="L's metadata frame T (session 1) or E (session 2); run and trial as stored in "
                "artifacts/exp_l/outputs_*.joblib (and K's command records)",
        yang2025="dayN of data/processed/yang2025/dayN_subSSS.npz; run and trial = that "
                 "archive's run and trial arrays"))


# ------------------------------------------------------------------ helpers

def confine(cfg):
    """Resolve every output destination; stop, before any write, at one outside the audit trees."""
    for name in DESTINATIONS:
        path = (ROOT / cfg[name]).resolve()
        if path in NOT_OWNED or not any(path.is_relative_to(base) for base in OWNED):
            raise SystemExit(f"audit_cohorts: refused {name} = {cfg[name]!r} (resolves to {path}): "
                             "outputs must lie under artifacts/audit/ or results/audit/ (not "
                             "results/audit/validation.json); nothing was written")
        cfg[name] = path


def register(registry, path, role):
    """Record a source file's SHA-256 once; returns its project-relative path."""
    rel = Path(path).resolve().relative_to(ROOT).as_posix()
    if rel not in registry:
        registry[rel] = dict(sha256=sha256_file(ROOT / rel), bytes=(ROOT / rel).stat().st_size,
                             roles=[])
    if role not in registry[rel]["roles"]:
        registry[rel]["roles"].append(role)
    return rel


def save_npz(path, arrays):
    """np.savez_compressed layout with fixed member timestamps: equal arrays give equal bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in arrays.items():
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with archive.open(info, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(value), allow_pickle=False)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def to_codes(stored, code_map):
    stored = np.asarray(stored, np.int64)
    if not np.isin(stored, list(code_map)).all():
        raise ValueError("Stored commands outside -1, 0, +1")
    return np.array([code_map[c] for c in (-1, 0, 1)], np.int64)[stored + 1]


def stream(cohort, rung, kind, seed, frame, outputs, totals, sources, day1, tau=None, *, source):
    """`source` = (every source trial: subject, session, run, trial, day, label and, for
    commands, code or move; full feature array or None) for the trial-key check."""
    return dict(cohort=cohort, rung=rung, kind=kind, seed=int(seed), tau=tau, frame=frame,
                outputs=outputs, totals=totals, sources=sources, day1_status=day1, source=source)


def select_records(records, cohort, spec, seed, tau):
    part = records[(records.dataset == cohort) & (records.condition == spec["condition"])
                   & (records.seed == seed) & np.isclose(records.tau, tau)
                   & records.source.isin(list(spec["sources"]))]
    return part.assign(day=part.source.map(spec["sources"]).astype(np.int64))


def record_exports(cfg, cohort, records, stored, sources_of):
    """2a and Lee2019 command exports: K's stored command records of the P2-RC condition.
    `stored(seed)` gives every stored trial (KEY, segment_rule, label, artifact, finite, pred)."""
    ccfg = cfg["cohorts"][cohort]
    spec, out = ccfg["commands"], []
    for seed in spec["seeds"]:
        trials = stored(seed)
        for tau in spec["taus"]:
            part = select_records(records, cohort, spec, seed, tau)
            if (part.session.map(ccfg["days"]) != part.day).any():
                raise ValueError(f"{cohort}: record source and session disagree")
            frame = part[KEY + ["segment", "target", "command"]].merge(
                trials, on=KEY, how="left", validate="one_to_one")
            if frame.pred.isna().any():
                raise ValueError(f"{cohort}: command record without a stored trial")
            frame = frame.assign(chrono=frame.run * 1000 + frame.trial,
                                 code=to_codes(frame.command, cfg["code_map"]))
            table = part.assign(label=part.target, code=to_codes(part.command, cfg["code_map"]))
            out.append(stream(cohort, spec["condition"], "commands", seed, frame,
                              frame.code.to_numpy(np.int64), trials, sources_of(seed),
                              "cross-fitted", tau, source=(table, None)))
    return out


# ------------------------------------------------------------------ cohorts

def bnci2014_001(cfg, records, registry):
    cohort = "bnci2014_001"
    ccfg = cfg["cohorts"][cohort]

    def load(key, seed):
        path = ROOT / ccfg["outputs"].format(key=key, seed=seed)
        data = load_npz(path)
        meta = pd.DataFrame({c: data[c] for c in ("subject", "session", "run", "trial", "label",
                                                  "artifact", "finite")})
        frame = pd.DataFrame(dict(
            subject=meta.subject.astype(np.int64), day=meta.session.map(ccfg["days"]),
            session=meta.session.astype(str),
            run=meta.run, trial=meta.trial, chrono=meta.run * 1000 + meta.trial,
            segment_rule=np.where(meta.run <= ccfg["segment_runs"], 1, 2),
            label=meta.label.astype(np.int64), artifact=meta.artifact.astype(bool),
            finite=meta.finite.astype(bool), pred=data["logits"].argmax(axis=1)))
        return frame, data["representation"], register(registry, path, "B final outputs")

    out = []
    for rung, spec in ccfg["features"].items():
        for seed in spec["seeds"]:
            trials, rep, src = load(spec["key"], seed)
            keep = (~trials.artifact & trials.finite).to_numpy()
            if not np.isfinite(rep[keep]).all():
                raise ValueError(f"2a {rung} {seed}: non-finite retained representation")
            frame = trials[keep].assign(segment=trials.segment_rule[keep],
                                        target=trials.label[keep])
            out.append(stream(cohort, rung, "features", seed, frame, rep[keep], trials, [src],
                              "in-sample reference", source=(trials, rep)))
    return out + commands_of(cfg, cohort, records, registry, load)


def commands_of(cfg, cohort, records, registry, load):
    ccfg = cfg["cohorts"][cohort]
    rec = register(registry, ROOT / cfg["command_records"], "K command records")
    key = ccfg["commands"]["key"]
    path = lambda seed: ROOT / ccfg["outputs"].format(key=key, seed=seed)
    return record_exports(cfg, cohort, records, lambda seed: load(key, seed)[0],
                          lambda seed: [rec, register(registry, path(seed), "logits for task BA")])


def lee2019(cfg, records, registry):
    cohort = "lee2019"
    ccfg = cfg["cohorts"][cohort]

    def load(key, seed):
        path = ROOT / ccfg["outputs"].format(key=key, seed=seed)
        packet = joblib.load(path)
        frames = []
        for session, rep, logits in (("T", packet["tz"], packet["tl"]),
                                     ("E", packet["ez"], packet["el"])):
            meta = packet[session].reset_index(drop=True)
            frames.append(pd.DataFrame(dict(
                subject=meta.subject.astype(np.int64), day=meta.session.map(ccfg["days"]),
                session=meta.session.astype(str),
                run=meta.run, trial=meta.trial, chrono=meta.run * 1000 + meta.trial,
                segment_rule=meta.phase.map(ccfg["phases"]), label=meta.label.astype(np.int64), artifact=meta.artifact.astype(bool),
                finite=meta.finite.astype(bool), pred=logits.argmax(axis=1))))
        trials = pd.concat(frames, ignore_index=True)
        rep = np.concatenate([packet["tz"], packet["ez"]])
        return trials, rep, register(registry, path, "L final outputs")

    out = []
    for rung, spec in ccfg["features"].items():
        for seed in spec["seeds"]:
            trials, rep, src = load(spec["key"], seed)
            keep = (trials.finite & np.isfinite(rep).all(axis=1)).to_numpy()
            frame = trials[keep].assign(segment=trials.segment_rule[keep],
                                        target=trials.label[keep])
            out.append(stream(cohort, rung, "features", seed, frame, rep[keep], trials, [src],
                              "in-sample reference", source=(trials, rep)))
    return out + commands_of(cfg, cohort, records, registry, load)


def stream_rows(meta, logits, s):
    """Rows behind a beat.exp_m.make_stream stream (finite rows sorted by user, trial)."""
    keep = np.isfinite(logits).all(axis=1) & meta.finite.to_numpy(bool)
    order = np.lexsort((meta.trial.to_numpy(), meta.subject.to_numpy()))
    rows = order[keep[order]]
    if not (np.array_equal(s.subject, meta.subject.to_numpy()[rows])
            and np.array_equal(s.target, meta.label.to_numpy()[rows])):
        raise ValueError("Command stream rows do not align with their trials")
    return rows


def yang2025(cfg, registry):
    """Exports, check rows and archive hashes of Yang2025 (study M's models and functions)."""
    cohort = "yang2025"
    ccfg = cfg["cohorts"][cohort]
    mcfg = load_cfg(ROOT / ccfg["exp_m_config"])
    register(registry, ROOT / ccfg["exp_m_config"], "M configuration")
    taus = ccfg["commands"]["taus"]
    if not np.allclose(taus, [mcfg["tau_star"], mcfg["tau_rule"]["secondary_tau"]]):
        raise ValueError("Yang2025 taus differ from configs/exp_m.yaml tau_star, secondary_tau")
    selection = json.loads((ROOT / ccfg["selection"]).read_text(encoding="utf-8"))
    sel = register(registry, ROOT / ccfg["selection"], "M temperatures and OOF digests")
    missing = check_data(mcfg)  # processed archives against their recorded SHA-256
    if missing:
        raise ValueError(f"Yang2025 user-days without a processed file: {missing}")
    archives = []
    for day in (1, 2, 3):
        recorded = read_checksums(ROOT / CHECKSUMS[day])
        register(registry, ROOT / CHECKSUMS[day], "recorded SHA-256 of the processed archives")
        for subject in mcfg["subjects"]:
            name = f"{mcfg['data_dir']}/day{day}_sub{subject:03d}.npz"
            sha = sha256_file(ROOT / name)
            archives.append(dict(path=name, sha256=sha, bytes=(ROOT / name).stat().st_size,
                                 equals_recorded=recorded.get(name) == sha))
    art = ROOT / ccfg["artifacts"]
    for key in ccfg["model_keys"]:
        for seed in ccfg["model_seeds"]:
            used = key in ccfg["features"]
            register(registry, ROOT / ccfg["models"].format(key=key, seed=seed),
                     "M final model" + ("" if used else " (hashed, not exported)"))
            used = key == ccfg["commands"]["condition"]
            register(registry, art / f"oof_{key}_{seed}.joblib",
                     "M day-1 cross-fitted logits" + ("" if used else " (hashed, not exported)"))
    seg_trials = mcfg["dev"]["segment_trials"]

    def frame_of(meta):
        meta = meta.reset_index(drop=True)
        return pd.DataFrame(dict(
            subject=meta.subject.astype(np.int64), day=meta.day.astype(np.int64),
            session="day" + meta.day.astype(str), run=meta.run,
            chrono=meta.trial.astype(np.int64), trial=meta.trial,
            segment=np.where(meta.trial <= seg_trials, 1, 2), target=meta.label.astype(np.int64),
            artifact=meta.artifact.astype(bool), finite=meta.finite.astype(bool)))

    torch_threads(mcfg)
    with threadpool_limits(limits=1):  # M's setting, so outputs match its stored figures
        cov = {}
        for day in (1, 2, 3):
            signals = load_trials(ROOT / mcfg["data_dir"], mcfg["subjects"], day,
                                  allow_later_days=day > 1, skip_missing=day > 1)
            cov[day] = valid_covariances(signals)
            del signals
        totals = pd.concat([frame_of(cov[d].meta) for d in (1, 2, 3)], ignore_index=True)
        cov[1] = finite_trials(cov[1])  # M's day-1 fits and day-1 P0 representation
        outputs, models = {}, {}
        for rung, spec in ccfg["features"].items():
            for seed in spec["seeds"]:
                path = ROOT / ccfg["models"].format(key=rung, seed=seed)
                model = joblib.load(path)
                models[(rung, seed)] = register(registry, path, "M final model")
                outputs[(rung, seed)] = {d: model(cov[d]) for d in spec["days"]}
    out, confusion = [], []
    for (rung, seed), by_day in outputs.items():
        parts, feats = [], []
        for d, (rep, logits) in by_day.items():
            if d > 1:
                good = cov[d].meta.finite.to_numpy(bool)
                confusion.extend(_task_rows(cov[d].subset(good), logits[good], key=rung,
                                            seed=seed, day=d))
            frame = frame_of(cov[d].meta)
            keep = frame.finite.to_numpy() & np.isfinite(rep).all(axis=1)
            parts.append(frame[keep].assign(pred=logits[keep].argmax(axis=1)))
            feats.append(rep[keep])
        days = ccfg["features"][rung]["days"]
        table = pd.concat([cov[d].meta.reset_index(drop=True).assign(session=f"day{d}")
                           for d in by_day], ignore_index=True)
        out.append(stream(cohort, rung, "features", seed, pd.concat(parts, ignore_index=True),
                          np.concatenate(feats), totals[totals.day.isin(days)],
                          [models[(rung, seed)]], "in-sample reference" if 1 in days else "absent",
                          source=(table, np.concatenate([rep for rep, _ in by_day.values()]))))
    spec = ccfg["commands"]
    key = spec["condition"]
    for seed in spec["seeds"]:
        oof = load_oof(art, key, seed, selection)  # digest-checked against the selection
        temperature = selection["temperatures"][key][str(seed)]
        sources = [register(registry, art / f"oof_{key}_{seed}.joblib",
                            "M day-1 cross-fitted logits"), models[(key, seed)], sel]
        logits_by_day = {1: (oof.meta.reset_index(drop=True), np.asarray(oof.X, float))}
        logits_by_day.update({d: (cov[d].meta.reset_index(drop=True), outputs[(key, seed)][d][1])
                              for d in (2, 3)})
        # A move is the argmax class of its logits (beat.exp_c.decide: sign of the margin).
        table = pd.concat([meta.assign(session=f"day{d}", move=np.where(
            logits.argmax(axis=1) == 1, cfg["code_map"][1], cfg["code_map"][-1]))
            for d, (meta, logits) in logits_by_day.items()], ignore_index=True)
        for tau in taus:
            parts = []
            for d, (meta, logits) in logits_by_day.items():
                s = day_stream(meta, logits, temperature, tau, mcfg)
                rows = stream_rows(meta, logits, s)
                frame = frame_of(meta.iloc[rows])
                if not np.array_equal(frame.segment.to_numpy(), s.segment):
                    raise ValueError("Yang2025 segment rule differs from M's day_stream")
                parts.append(frame.assign(code=to_codes(s.output.astype(np.int64) - 1,
                                                        cfg["code_map"]),
                                          pred=logits[rows].argmax(axis=1)))
            frame = pd.concat(parts, ignore_index=True)
            out.append(stream(cohort, key, "commands", seed, frame, frame.code.to_numpy(np.int64),
                              totals, sources, "cross-fitted", tau, source=(table, None)))
    return out, pd.DataFrame(confusion), archives


# ------------------------------------------------------------------ writing and summaries

def write_export(cfg, ex):
    frame = ex["frame"]
    idx = np.lexsort((frame.chrono.to_numpy(), frame.day.to_numpy(), frame.subject.to_numpy()))
    frame = frame.iloc[idx].reset_index(drop=True)
    if frame.duplicated(["subject", "day", "chrono"]).any():
        raise ValueError(f"{ex['cohort']} {ex['rung']}: duplicate trials")
    frame["order"] = frame.groupby(["subject", "day"]).cumcount()
    outputs = np.asarray(ex["outputs"])[idx]
    arrays = dict(outputs=outputs, user=frame.subject.to_numpy(np.int64),
                  day=frame.day.to_numpy(np.int64), segment=frame.segment.to_numpy(np.int64),
                  order=frame.order.to_numpy(np.int64), target=frame.target.to_numpy(np.int64))
    if ex["kind"] == "commands":
        if not np.isin(outputs, np.arange(cfg["alphabet_size"])).all():
            raise ValueError("Command code outside the alphabet")
        arrays["C"] = np.int64(cfg["alphabet_size"])
        name = f"commands_{ex['rung']}_tau{ex['tau']:.2f}_seed{ex['seed']}.npz"
    else:
        if not np.isfinite(outputs).all():
            raise ValueError("Non-finite exported feature")
        arrays["outputs"] = outputs.astype(np.float64)
        name = f"features_{ex['rung']}_seed{ex['seed']}.npz"
    path = ROOT / cfg["inputs_dir"] / ex["cohort"] / name
    save_npz(path, arrays)
    keys = pd.DataFrame(dict(
        row=np.arange(len(frame)), user=arrays["user"], day=arrays["day"], order=arrays["order"],
        source_session=frame.session.astype(str), source_run=frame.run.astype(np.int64),
        source_trial=frame.trial.astype(np.int64)))
    keys_path = path.with_name(path.stem + ".trials.csv")
    write_csv(keys, keys_path)
    ex.update(frame=frame, outputs=arrays["outputs"], path=path, keys_path=keys_path)


def by_day_user(series):
    table = series.unstack(fill_value=0)
    return {str(d): {str(u): int(v) for u, v in row.items()} for d, row in table.iterrows()}


def manifest_entry(cfg, ex):
    ccfg, frame = cfg["cohorts"][ex["cohort"]], ex["frame"]
    totals = ex["totals"]
    kept = pd.MultiIndex.from_frame(frame[["subject", "day", "chrono"]])
    lost = totals[~pd.MultiIndex.from_frame(totals[["subject", "day", "chrono"]]).isin(kept)]
    if len(frame) + len(lost) != len(totals):
        raise ValueError(f"{ex['cohort']} {ex['rung']}: exported rows outside the stored trials")
    reason = np.where(lost.artifact, "artifact_flag", np.where(~lost.finite, "non_finite",
                                                               "other"))
    users = sorted(map(int, totals.subject.unique()))
    days = sorted(map(int, frame.day.unique()))
    grid = pd.MultiIndex.from_product([days, users], names=["day", "subject"])
    rows = frame.groupby(["day", "subject"]).size().reindex(grid, fill_value=0)
    dropped = lost.groupby(["day", "subject"]).size().reindex(grid, fill_value=0)
    cells = pd.MultiIndex.from_product([users, days, [1, 2]], names=["subject", "day", "segment"])
    seg = frame.groupby(["subject", "day", "segment"]).size().reindex(cells, fill_value=0)
    short = {}
    for k in cfg["k_grid"]:
        below = seg[seg < k]
        if len(below):
            affected = sorted(map(int, below.index.get_level_values("subject").unique()))
            short[str(k)] = dict(
                users=affected, share=len(affected) / len(users),
                above_max_excluded_share=len(affected) / len(users) > cfg["max_excluded_share"],
                segments=[[int(u), int(d), int(s), int(n)] for (u, d, s), n in below.items()])
    entry = dict(
        file=ex["path"].relative_to(ROOT).as_posix(), sha256=sha256_file(ex["path"]),
        bytes=ex["path"].stat().st_size, cohort=ex["cohort"], rung=ex["rung"], kind=ex["kind"],
        model_seed=ex["seed"], tau=ex["tau"], shape=list(ex["outputs"].shape),
        dtype=str(ex["outputs"].dtype), rows=len(frame), users=sorted(map(int, frame.subject.unique())),
        users_absent=sorted(set(users) - set(map(int, frame.subject.unique()))),
        days={str(d): ccfg["day_labels"][d] for d in days},
        segments={str(s): ccfg["segment_labels"][s] for s in (1, 2)},
        rows_per_user_day=by_day_user(rows), dropped_total=len(lost),
        dropped_per_user_day=by_day_user(dropped),
        dropped_by_reason={r: int((reason == r).sum()) for r in np.unique(reason)},
        drop_rule=DROPS[(ex["cohort"], ex["kind"])],
        day1_status=ex["day1_status"],
        min_rows_per_segment=int(seg.min()),
        segments_shorter_than_k=dict(
            note="[user, day, segment, rows] with fewer rows than k: no k-window there",
            **short),
        seconds_per_output=ccfg["seconds_per_output"], sources=ex["sources"],
        trial_keys=dict(file=ex["keys_path"].relative_to(ROOT).as_posix(),
                        sha256=sha256_file(ex["keys_path"]), bytes=ex["keys_path"].stat().st_size,
                        rows=len(frame), columns=list(TRIAL_KEYS["columns"]),
                        source_session=TRIAL_KEYS["source_session"][ex["cohort"]]))
    if ex["kind"] == "commands":
        entry.update(alphabet_size=cfg["alphabet_size"], code_labels=cfg["code_labels"],
                     code_map={str(k): v for k, v in cfg["code_map"].items()})
    return entry


def per_user(frame, days):
    rows = []
    for subject, g in frame[frame.day.isin(days)].groupby("subject"):
        row = dict(subject=int(subject), trials=len(g),
                   ba=float(balanced_accuracy_score(g.target, g.pred)))
        if "code" in g:
            right, left = g.target.to_numpy() == 1, g.target.to_numpy() == 0
            code = g.code.to_numpy()
            row.update(hold=float((code == 0).mean()),
                       correct=float((((code == 1) & left) | ((code == 2) & right)).mean()),
                       wrong=float((((code == 2) & left) | ((code == 1) & right)).mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def utility_row(cfg, ex):
    days = [d for d in cfg["cohorts"][ex["cohort"]]["deployment_days"]
            if d in set(ex["frame"].day)]
    pu = per_user(ex["frame"], days)
    ex["per_user"] = pu
    row = dict(cohort=ex["cohort"], rung=ex["rung"], kind=ex["kind"], tau=ex["tau"],
               model_seed=ex["seed"], days=",".join(map(str, days)), users=len(pu),
               trials=int(pu.trials.sum()), task_ba=pu.ba.mean())
    if ex["kind"] == "commands":
        row.update(hold_rate=pu.hold.mean(), correct_move_rate=pu.correct.mean(),
                   wrong_move_rate=pu.wrong.mean())
    return row


# ------------------------------------------------------------------ checks

def check(name, passed, **details):
    return dict(check=name, passed=bool(passed), **details)


def check_records(cfg, ex, records):
    """The written command export equals K's records (code-mapped, sorted by user, day, time)."""
    spec = cfg["cohorts"][ex["cohort"]]["commands"]
    part = select_records(records, ex["cohort"], spec, ex["seed"], ex["tau"]).sort_values(
        ["subject", "day", "run", "trial"])
    data = load_npz(ex["path"])
    expected = dict(user=part.subject, day=part.day, segment=part.segment, target=part.target,
                    outputs=to_codes(part.command, cfg["code_map"]))
    same_rows = len(part) == len(data["outputs"])
    mismatches = {f: int((data[f] != np.asarray(v)).sum()) if same_rows else None
                  for f, v in expected.items()}
    frame = ex["frame"]
    rule = int((frame.segment != frame.segment_rule).sum())
    label = int((frame.target != frame.label).sum())
    return check("command_export_equals_records", same_rows and not any(mismatches.values())
                 and rule == 0 and label == 0 and int(data["C"]) == cfg["alphabet_size"],
                 cohort=ex["cohort"], file=ex["path"].relative_to(ROOT).as_posix(),
                 rows_export=len(data["outputs"]), rows_records=len(part), mismatches=mismatches,
                 segment_vs_cohort_rule_mismatches=rule, target_vs_stored_label_mismatches=label)


def read_keys(path):
    return pd.read_csv(path, dtype={"source_session": str})


def check_keys(ex, exports):
    """Each sidecar key, looked up in the stored source trials, gives the export row's user, day,
    target and output (features: stored or recomputed vector; 2a/Lee commands: K's record after
    the code map; Yang2025 commands: every move equals the argmax of the trial's logits)."""
    keys, data = read_keys(ex["keys_path"]), load_npz(ex["path"])
    table, values = ex["source"]
    table = table.reset_index(drop=True)
    index = pd.MultiIndex.from_arrays([
        table.subject.to_numpy(np.int64), table.session.astype(str).to_numpy(),
        table.run.to_numpy(np.int64), table.trial.to_numpy(np.int64)])
    wanted = pd.MultiIndex.from_arrays([
        keys.user.to_numpy(np.int64), keys.source_session.to_numpy(),
        keys.source_run.to_numpy(np.int64), keys.source_trial.to_numpy(np.int64)])
    pos = index.get_indexer(wanted) if index.is_unique else np.full(len(keys), -1)
    found, n = pos >= 0, len(data["outputs"])
    p, out = pos[found], data["outputs"][found]
    aligned = len(keys) == n and np.array_equal(keys.row, np.arange(n)) and all(
        np.array_equal(keys[f], data[f]) for f in ("user", "day", "order"))
    mismatches = dict(day=int((table.day.to_numpy(np.int64)[p] != data["day"][found]).sum()),
                      target=int((table.label.to_numpy(np.int64)[p]
                                  != data["target"][found]).sum()))
    if values is not None:
        mismatches["outputs"] = int((values[p] != out).any(axis=1).sum())
        compared = "feature vector at the key (stored or recomputed from the stored model)"
    elif "code" in table:
        mismatches["outputs"] = int((table.code.to_numpy()[p] != out).sum())
        compared = "K's command record at the key after the code map"
    else:
        moves = out != 0
        mismatches["outputs"] = int((table.move.to_numpy()[p][moves] != out[moves]).sum())
        compared = "moves vs the argmax class of the trial's logits"
    details = dict(file=ex["keys_path"].relative_to(ROOT).as_posix(), rows=len(keys),
                   source_rows=len(table), source_keys_unique=bool(index.is_unique),
                   keys_not_found=int((~found).sum()), duplicate_keys=int(wanted.duplicated().sum()),
                   aligned_with_export=bool(aligned), outputs_compared=compared,
                   mismatches=mismatches)
    if ex["kind"] == "commands":
        feat = next(e for e in exports if e["kind"] == "features" and e["cohort"] == ex["cohort"]
                    and e["rung"] == ex["rung"] and e["seed"] == ex["seed"])
        cols = ["user", "day", "source_session", "source_run", "source_trial"]
        m = keys[cols + ["order"]].merge(read_keys(feat["keys_path"])[cols + ["order"]], on=cols,
                                         how="outer", suffixes=("", "_features"), indicator=True)
        side = m._merge.map(dict(both="both", left_only="commands_only",
                                 right_only="features_only"))
        details["common_support"] = dict(
            features_file=feat["keys_path"].relative_to(ROOT).as_posix(),
            rows_by_day={str(d): {k: int(v) for k, v in g.value_counts().items()}
                         for d, g in side.groupby(m.day)},
            shared_rows_with_different_order=int(
                (m[m._merge == "both"].order != m[m._merge == "both"].order_features).sum()))
    passed = (index.is_unique and found.all() and not wanted.has_duplicates and aligned
              and not any(mismatches.values()))
    return check("trial_keys_reproduce_source_rows", passed, cohort=ex["cohort"],
                 export=ex["path"].relative_to(ROOT).as_posix(), **details)


def compare_users(mine, ref, ba_col="ba"):
    merged = mine.merge(ref, on="subject", how="outer", suffixes=("", "_ref"), indicator=True)
    both = merged[merged._merge == "both"]
    diff = (both.ba - both[f"{ba_col}_ref"]).abs()
    return dict(users_compared=len(both), users_unmatched=int((merged._merge != "both").sum()),
                n_mismatches=int((both.trials != both.n).sum()),
                max_abs_ba_diff=float(diff.max()) if len(diff) else None,
                mean_ba=float(both.ba.mean()), mean_ba_ref=float(both[f"{ba_col}_ref"].mean()))


def check_task(cfg, exports):
    """2a and Lee2019 task BA per user against B's task_final.csv and L's task.csv."""
    out = []
    for cohort, col in (("bnci2014_001", "condition"), ("lee2019", "key")):
        ccfg = cfg["cohorts"][cohort]
        ref = pd.read_csv(ROOT / ccfg["task_reference"])
        if cohort == "bnci2014_001":
            ref = ref[ref.target == 0.0]
        keys = {rung: spec["key"] for rung, spec in ccfg["features"].items()}
        for ex in (e for e in exports if e["cohort"] == cohort):
            key = keys[ex["rung"]] if ex["kind"] == "features" else ccfg["commands"]["key"]
            r = ref[(ref[col] == key) & (ref.seed == ex["seed"])][["subject", "ba", "n"]]
            res = compare_users(ex["per_user"], r)
            name = f"{cohort} {ex['kind']} {ex['rung']} seed {ex['seed']}"
            if ex["kind"] == "commands":
                name += f" tau {ex['tau']}"
            if ex["kind"] == "commands" and cohort == "bnci2014_001":
                # K's deployment stream keeps artifact-flagged trials; B's task rows do not.
                masked = per_user(ex["frame"][~ex["frame"].artifact], ccfg["deployment_days"])
                res = dict(all_trials=res, quality_masked=compare_users(masked, r))
                passed = (res["quality_masked"]["max_abs_ba_diff"] <= 1e-12
                          and res["quality_masked"]["n_mismatches"] == 0
                          and res["quality_masked"]["users_unmatched"] == 0)
            else:
                passed = (res["max_abs_ba_diff"] <= 1e-12 and res["n_mismatches"] == 0
                          and res["users_unmatched"] == 0)
            out.append(check("task_ba_equals_study", passed, export=name,
                             reference=ccfg["task_reference"], **res))
    return out


def check_rates(cfg, exports):
    """2a and Lee2019 command rates per day against K's stream_summary.csv."""
    ss = pd.read_csv(ROOT / cfg["stream_summary"])
    out = []
    for ex in exports:
        if ex["kind"] != "commands" or ex["cohort"] == "yang2025":
            continue
        spec = cfg["cohorts"][ex["cohort"]]["commands"]
        for source, day in spec["sources"].items():
            g = ex["frame"][ex["frame"].day == day]
            right, code = g.target.to_numpy() == 1, g.code.to_numpy()
            mine = dict(trials=len(g), hold_rate=float((code == 0).mean()),
                        correct_rate=float((((code == 1) & ~right) | ((code == 2) & right)).mean()),
                        wrong_rate=float((((code == 2) & ~right) | ((code == 1) & right)).mean()))
            ref = ss[(ss.dataset == ex["cohort"]) & (ss.condition == spec["condition"])
                     & (ss.seed == ex["seed"]) & np.isclose(ss.tau, ex["tau"])
                     & (ss.source == source)].iloc[0]
            diffs = {k: abs(v - float(ref[k])) for k, v in mine.items()}
            out.append(check("command_rates_equal_stream_summary",
                             max(diffs.values()) <= 1e-12, cohort=ex["cohort"], seed=ex["seed"],
                             tau=ex["tau"], source=source, mine=mine, abs_diff=diffs,
                             reference=cfg["stream_summary"]))
    return out


def check_yang(cfg, exports, confusion):
    ccfg = cfg["cohorts"]["yang2025"]
    out = []
    dev = joblib.load(ROOT / ccfg["dev_p0_features"])
    dev_meta = dev["meta"].reset_index(drop=True)
    order = np.lexsort((dev_meta.trial.to_numpy(), dev_meta.subject.to_numpy()))
    for ex in (e for e in exports if e["cohort"] == "yang2025" and e["rung"] == "P0"):
        data = load_npz(ex["path"])
        day1 = data["day"] == 1
        X, ref = data["outputs"][day1], np.asarray(dev["X"], float)[order]
        same = X.shape == ref.shape and np.array_equal(data["user"][day1],
                                                       dev_meta.subject.to_numpy()[order])
        diff = float(np.abs(X - ref).max()) if same else None
        out.append(check("yang2025_day1_p0_equals_dev_features", same and diff == 0.0,
                         file=ex["path"].relative_to(ROOT).as_posix(), rows=int(day1.sum()),
                         rows_reference=len(ref), max_abs_diff=diff,
                         reference=ccfg["dev_p0_features"]))
    ref = pd.read_csv(ROOT / ccfg["task_reference"])
    cols = ["n", "tn", "fp", "fn", "tp"]
    for (key, seed), mine in confusion.groupby(["key", "seed"]):
        r = ref[(ref.key == key) & (ref.seed == seed)]
        m = mine.merge(r, on=["day", "subject"], how="outer", suffixes=("", "_ref"),
                       indicator=True)
        both = m[m._merge == "both"]
        bad = (both[cols].to_numpy() != both[[c + "_ref" for c in cols]].to_numpy()).any(axis=1)
        out.append(check("yang2025_day23_confusion_rows_reproduced",
                         not bad.any() and len(both) == len(m), key=key, seed=int(seed),
                         rows_compared=len(both), rows_unmatched=int((m._merge != "both").sum()),
                         rows_mismatching=int(bad.sum()),
                         max_abs_ba_diff=float((both.ba - both.ba_ref).abs().max()),
                         reference=ccfg["task_reference"]))
    streams = pd.read_csv(ROOT / ccfg["dev_streams"])
    final = pd.read_csv(ROOT / ccfg["final_utility"])
    commands = [e for e in exports if e["cohort"] == "yang2025" and e["kind"] == "commands"]
    for ex in commands:
        pu = per_user(ex["frame"], [1])
        g = ex["frame"][ex["frame"].day == 1]
        r = streams[(streams.key == ex["rung"]) & (streams.seed == ex["seed"])
                    & np.isclose(streams.tau, ex["tau"])].iloc[0]
        mine = dict(trials=len(g), hold_rate=float((g.code == 0).mean()),
                    correct=pu.correct.mean(), wrong=pu.wrong.mean(), hold=pu.hold.mean())
        diffs = {k: abs(v - float(r[k])) for k, v in mine.items()}
        out.append(check("yang2025_day1_commands_equal_dev_streams", max(diffs.values()) <= 1e-9,
                         seed=ex["seed"], tau=ex["tau"], mine=mine, abs_diff=diffs,
                         reference=ccfg["dev_streams"]))
    for tau in ccfg["commands"]["taus"]:
        parts = [e["per_user"].set_index("subject") for e in commands
                 if np.isclose(e["tau"], tau)]
        mean = pd.concat(parts).groupby(level=0).mean()
        mine = dict(hold=mean.hold.mean(), wrong=mean.wrong.mean(),
                    net=(mean.correct - mean.wrong).mean())
        r = final[(final.scenario == "cal_to_dep") & np.isclose(final.tau, tau)]
        diffs = {k: abs(v - float(r[r.statistic == f"{k}_undefended"]["mean"].iloc[0]))
                 for k, v in mine.items()}
        out.append(check("yang2025_day23_command_rates_equal_final_estimates",
                         max(diffs.values()) <= 1e-9, tau=tau, seeds=len(parts), mine=mine,
                         abs_diff=diffs, reference=ccfg["final_utility"],
                         note="seed mean of per-user rates over pooled days 2-3, then mean over "
                              "users (M's cal_to_dep *_undefended estimates)"))
    return out


# ------------------------------------------------------------------ entry point

def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/audit.yaml")
    args = parser.parse_args(argv)
    start = time.perf_counter()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    confine(cfg)
    registry = {}
    records = pd.read_csv(ROOT / cfg["command_records"])
    with threadpool_limits(limits=cfg["threads"]):
        exports = bnci2014_001(cfg, records, registry) + lee2019(cfg, records, registry)
        yang, confusion, archives = yang2025(cfg, registry)
        exports += yang
        for ex in exports:
            write_export(cfg, ex)
        entries = [manifest_entry(cfg, ex) for ex in exports]
        utility = pd.DataFrame([utility_row(cfg, ex) for ex in exports])
        checks = [check_records(cfg, ex, records) for ex in exports
                  if ex["kind"] == "commands" and ex["cohort"] != "yang2025"]
        checks += check_task(cfg, exports) + check_rates(cfg, exports)
        checks += check_yang(cfg, exports, confusion)
        checks += [check_keys(ex, exports) for ex in exports]
    for path, role in ((cfg["stream_summary"], "K stream summary (check reference)"),
                       *((c["task_reference"], "task figures (check reference)")
                         for c in cfg["cohorts"].values()),
                       *((cfg["cohorts"]["yang2025"][k], "M figures (check reference)")
                         for k in ("dev_p0_features", "dev_streams", "final_utility"))):
        register(registry, ROOT / path, role)
    write_csv(utility, ROOT / cfg["utility"])
    failed = [c for c in checks if not c["passed"]]
    write_json(ROOT / cfg["checks"], dict(
        passed=not failed, checks_total=len(checks), checks_failed=len(failed), checks=checks))
    plan = ROOT / "docs/plan_v3.md"
    config = register(registry, args.config, "export configuration")
    write_json(ROOT / cfg["manifest"], dict(
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        command=".venv\\Scripts\\python.exe scripts/audit_cohorts.py --config " + config,
        script=dict(path="scripts/audit_cohorts.py",
                    sha256=sha256_file(ROOT / "scripts/audit_cohorts.py")),
        config=dict(path=config, sha256=registry[config]["sha256"]),
        plan=dict(path="docs/plan_v3.md", git_blob=blob_sha1(plan.read_bytes()),
                  sha256=sha256_file(plan)),
        git_head=git("rev-parse", "HEAD"), versions=versions(), schema=SCHEMA,
        trial_keys_schema=TRIAL_KEYS, utility_definition=UTILITY, exports=entries, sources=dict(sorted(registry.items())),
        yang2025_archives=archives, checks_passed=not failed,
        seconds=time.perf_counter() - start))
    print(f"{len(entries)} exports, {len(checks)} checks, {len(failed)} failed, "
          f"{time.perf_counter() - start:.0f} s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
