"""Plan v3 stages S2-S3: BEAT's cohort exports, development audits, runtime gate and freeze.

    .venv\\Scripts\\python.exe scripts/audit_cohorts.py [COMMAND] --config configs/audit.yaml

Commands (docs/plan_v3.md §4; every beat_audit execution is logged in results/audit/attempts.jsonl):
  export        (default) the input files below; runs no audit
  parity        §5 A3: the tool's closed-set module on study L's masks, directions and seeds
  pilot         runtime gate: one Lee2019 fold per stream type (beat_audit.audit_timing), projected
                to the whole job manifest
  develop       development audits of 2a and Lee2019 (all rungs), then summary.csv
  summary       rebuild results/audit/summary.csv from the reports present
  select        name V1's comparator from the Lee2019 report by the configured rule
  jobs          write the draft Yang2025 job manifest
  freeze        build results/audit/freeze/ (the Lead's step after review); --dry-run: checks only
  frozen-batch  under an exclusive lock: verify the freeze bundle, run every Yang2025 cell of its
                jobs.json, reconcile the reports with it, decide V1/V2; always closes its attempt
  recover-batch --attempt ID: close a frozen attempt interrupted without an end record as a
                technical failure (required before a technical rerun)
beat_audit runs on a Yang2025 export only inside `frozen-batch`; every other route refuses.

The export command writes, without running any audit (§3 "Input", §4 "Rungs exported per cohort"):
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
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import math
import multiprocessing
import os
import subprocess
import sys
import time
import uuid
import warnings
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
# The export functions import the sealed study modules (beat.*) when they run. Nothing else
# imports beat, so the frozen batch executes only the bundled source (beat_audit/, this script,
# its config); `local_modules` checks that at run time.
sys.path.insert(0, str(ROOT))


def sha256_file(path):
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serialisable: {type(value).__name__}")


def write_json(path, value):
    """beat.common's byte format: indent 2, no NaN, LF line ends on every platform."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=_plain) + "\n",
                    encoding="utf-8", newline="\n")


def write_csv(frame, path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", **kwargs)


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
OWNED = ("artifacts/audit", "results/audit")
NOT_OWNED = ("results/audit/validation.json",)  # written by the audit tool's validation lane
DESTINATIONS = ("inputs_dir", "manifest", "utility", "checks", "results_dir", "summary",
                "attempts", "parity", "runtime", "v1_selection", "jobs_draft", "freeze_dir",
                "batch_lock")
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
    owned = [(ROOT / p).resolve() for p in OWNED]
    refused = [(ROOT / p).resolve() for p in NOT_OWNED]
    for name in DESTINATIONS:
        path = (ROOT / cfg[name]).resolve()
        if path in refused or not any(path.is_relative_to(base) for base in owned):
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
    from beat.exp_b import _task_rows
    from beat.exp_l import finite_trials, valid_covariances
    from beat.exp_m import CHECKSUMS, check_data, day_stream, load_cfg, load_oof, torch_threads
    from beat.yang2025 import load_trials, read_checksums
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


def export(cfg, args):
    from beat.exp_m import versions
    from beat.verify import blob_sha1
    start = time.perf_counter()
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


# ------------------------------------------------------------------ audits: jobs and settings

REDUCTIONS = ("drop_pair_logistic", "drop_gallery", "drop_k1", "permutations_2000")  # §4 order
FROZEN_KINDS = ("frozen_batch", "technical_rerun")
DEVELOPMENT = ("bnci2014_001", "lee2019")
V1_SELECTION_COHORT = "lee2019"     # plan §4: V1's comparator is named on Lee2019 only
# plan §4, V1 rule: an exact tie goes to the simpler comparator, in this order
SIMPLICITY = ("cosine", "lda_cosine", "cosine_rich", "lda_cosine_rich", "pair_logistic")
TIER = dict(bnci2014_001="exploratory", lee2019="exploratory",
            yang2025="pre-specified (non-blind)")
REFUSED_YANG = ("audit_cohorts: refused: beat_audit runs on a Yang2025 export only inside the "
                "frozen batch (`freeze`, then `frozen-batch` with a matching bundle); nothing was run")
BUNDLE_FILES = ("source.zip", "sources.json", "pip_freeze.txt", "hashes.json", "folds.json",
                "seeds.json", "predictions.json", "jobs.json")


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_manifest(cfg):
    return read_json(cfg["manifest"])


def rung_id(entry):
    if entry["kind"] == "features":
        return f"features_{entry['rung']}"
    return f"commands_{entry['rung']}_tau{entry['tau']:.2f}"


def audit_jobs(cfg, cohorts, manifest=None):
    """One beat_audit stream per cohort rung (commands: per tau), grouping all its model seeds."""
    manifest = manifest or read_manifest(cfg)
    acfg, jobs, users = cfg["audit"], {}, defaultdict(set)
    for e in manifest["exports"]:
        users[e["cohort"]].update(e["users"])
        if e["cohort"] not in cohorts:
            continue
        job = jobs.setdefault((e["cohort"], rung_id(e)), dict(
            cohort=e["cohort"], rung=rung_id(e), export_rung=e["rung"], kind=e["kind"],
            tau=e["tau"], users=e["users"], days=e["days"], segments=e["segments"],
            day1_status=e["day1_status"], seconds_per_output=e["seconds_per_output"]["value"],
            shape=e["shape"], inputs={}, sha256={}))
        if job["users"] != e["users"]:
            raise ValueError(f"{e['file']}: model seeds of one rung must share their users")
        job["inputs"][str(e["model_seed"])] = e["file"]
        job["sha256"][str(e["model_seed"])] = e["sha256"]
    base = Path(cfg["results_dir"])
    for (cohort, rung), job in jobs.items():
        spec = acfg["conditions"][cohort]
        names = acfg.get("rung_conditions", {}).get(cohort, {}).get(rung, list(spec))
        job["conditions"] = {name: spec[name] for name in names}
        job["fold_users"] = sorted(users[cohort])  # one fold assignment for every rung of a cohort
        order = [str(d) for d in acfg["day_order"][cohort]]
        if not set(job["days"]) <= set(order):
            raise ValueError(f"audit.day_order.{cohort}: list every exported day {list(job['days'])}")
        job["day_order"] = [d for d in order if d in job["days"]]
        out = (base / cohort / rung).resolve()
        if not out.is_relative_to(base):
            raise SystemExit(f"audit_cohorts: refused report directory {out}")
        job["out"] = rel(out)
    return sorted(jobs.values(), key=lambda j: (list(cohorts).index(j["cohort"]), j["rung"]))


def reduce_settings(settings, reductions):
    """Plan §4 runtime-gate reductions, applied cumulatively in the plan's order."""
    if list(reductions) != list(REDUCTIONS[:len(reductions)]):
        raise ValueError(f"audit.reductions: apply a prefix of {REDUCTIONS} in that order")
    s = dict(settings, k_grid=list(settings["k_grid"]), comparators=list(settings["comparators"]))
    if "drop_pair_logistic" in reductions:
        s["comparators"].remove("pair_logistic")
    if "drop_gallery" in reductions:
        s["gallery"] = False
    if "drop_k1" in reductions:
        s["k_grid"].remove(1)
    if "permutations_2000" in reductions:
        s["permutations"] = 2000
    return s


def audit_settings(cfg):
    return reduce_settings(cfg["audit"]["settings"], cfg["audit"]["reductions"])


def audit_config(settings, job):
    from beat_audit import AuditConfig
    return AuditConfig(**dict(settings, k_grid=tuple(settings["k_grid"]),
                              comparators=tuple(settings["comparators"])),
                       cohort=job["cohort"], seconds_per_output=job["seconds_per_output"],
                       cues=("0", "1"), day_order=tuple(job["day_order"]))


def attacks(kind, settings, k):
    """Comparators the tool runs on a cell; rich summaries are skipped at k = 1 (plan §3)."""
    if kind == "commands":
        return ["bayes_factor"]
    return [n for n in settings["comparators"] if not (k == 1 and n.endswith("_rich"))]


def conditions_of(spec):
    """Explicit physical sides: "d" selects every segment of day d, "d/s" segment s of day d."""
    from beat_audit import Condition, Segment

    def side(selectors):
        return tuple(Segment(*str(s).split("/", 1)) for s in selectors)

    return [Condition(name, c["relation"], tuple((side(p["a"]), side(p["b"])) for p in c["pairs"]))
            for name, c in spec.items()]


def fold_assignment(users, seed):
    """The tool's rule (plan §3): min(5, U // 3) folds of a seeded permutation of sorted labels."""
    labels = sorted(str(u) for u in users)
    groups = np.array_split(np.random.default_rng(seed).permutation(labels),
                            min(5, len(labels) // 3))
    return {str(u): int(f) for f, group in enumerate(groups) for u in group}


def workers(cfg):
    count = int(cfg["audit"]["max_parallel_jobs"])
    if not 1 <= count <= 2:
        raise ValueError("audit.max_parallel_jobs: plan §4 allows at most two jobs at once")
    return count


def run_pool(function, calls, count, failed=None):
    """At most `count` jobs at once, each in a fresh process (so each has its own peak memory),
    started in the order of `calls`. With `failed`, a job that raises or whose process dies
    becomes failed(call, exception) instead of ending the run."""
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=count, mp_context=context,
                             max_tasks_per_child=1) as pool:
        futures = [pool.submit(function, *call) for call in calls]
        results = []
        for call, future in zip(calls, futures):
            try:
                results.append(future.result())
            except (Exception, SystemExit) as exc:
                if failed is None:
                    raise
                results.append(failed(call, exc))
        return results


def stream_failure(call, exc):
    """An audit_job whose worker raised or died: a failed stream, never read as a non-detection."""
    job = call[1]
    return dict(cohort=job["cohort"], rung=job["rung"], report=job["out"] + "/report.json",
                status="failed", error=f"{type(exc).__name__}: {exc}", seconds=None,
                peak_rss_bytes=None, cells=[])


# ------------------------------------------------------------------ audits: attempts and guard

def git_head():
    try:
        return git("rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(cfg):
    from beat_audit.engine import source_hashes
    return dict(git_head=git_head(), script_sha256=sha256_file(ROOT / "scripts/audit_cohorts.py"),
                config=rel(cfg["_path"]), config_sha256=sha256_file(cfg["_path"]),
                beat_audit_sha256=source_hashes()["sha256"])


def log_attempt(cfg, record):
    path = Path(cfg["attempts"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_attempts(cfg):
    path = Path(cfg["attempts"])
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def start_attempt(cfg, kind, command, **fields):
    record = dict(attempt=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-"
                  + uuid.uuid4().hex[:8], kind=kind, command=command, **provenance(cfg), **fields)
    log_attempt(cfg, dict(record, event="start", utc=utc()))
    return record


def end_attempt(cfg, record, **fields):
    log_attempt(cfg, dict(attempt=record["attempt"], kind=record["kind"],
                          command=record["command"], event="end", utc=utc(),
                          script_sha256=sha256_file(ROOT / "scripts/audit_cohorts.py"), **fields))


def source_files(cfg):
    """Frozen source bytes (plan §4 Freeze): beat_audit/, this script, its config and the plan."""
    files = sorted(p for p in (ROOT / "beat_audit").rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts)
    files += [ROOT / "scripts/audit_cohorts.py", Path(cfg["_path"]), ROOT / "docs/plan_v3.md"]
    return {rel(p): p for p in files}


def read_bundle(cfg):
    path = Path(cfg["freeze_dir"]) / "bundle.json"
    if not path.exists():
        raise SystemExit(f"audit_cohorts: refused: no freeze bundle at {rel(path)}; run `freeze` "
                         "first; nothing was run")
    return dict(read_json(path), sha256=sha256_file(path))


def local_modules():
    """Project files executed in this process, outside the environment (pip freeze covers it)."""
    environment = [Path(p).resolve() for p in {sys.prefix, sys.base_prefix}]
    files = set()
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).is_file():  # not "<stdin>" or a frozen module
            path = Path(path).resolve()
            if path.is_relative_to(ROOT) and not any(path.is_relative_to(e) for e in environment):
                files.add(path.relative_to(ROOT).as_posix())
    return sorted(files)


def try_lock(path):
    """Exclusive non-blocking OS lock on `path`, released by the OS if the process dies; returns
    the open descriptor, or None while another holder has it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def unlock(fd):
    if os.name == "nt":
        import msvcrt
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    os.close(fd)


@contextmanager
def batch_lock(cfg):
    """One frozen batch or recovery at a time (plan §4 Attempts)."""
    fd = try_lock(cfg["batch_lock"])
    if fd is None:
        raise SystemExit(f"audit_cohorts: refused: a frozen batch or its recovery is running (it "
                         f"holds {rel(cfg['batch_lock'])}); nothing was run")
    try:
        yield
    finally:
        unlock(fd)


def batch_running(cfg):
    fd = try_lock(cfg["batch_lock"])
    if fd is None:
        return True
    unlock(fd)
    return False


def guard_yang(cfg, job, attempt):
    """Refuse beat_audit on a Yang2025 export on every route but the frozen batch; returns
    whether the job is a Yang2025 job (then frozen_mismatches must be empty before it runs)."""
    yang = (Path(cfg["inputs_dir"]) / "yang2025").resolve()
    paths = [(ROOT / f).resolve() for f in job["inputs"].values()]
    if job["cohort"] != "yang2025" and not any(p.is_relative_to(yang) for p in paths):
        return False
    if not attempt or attempt.get("kind") not in FROZEN_KINDS:
        raise SystemExit(REFUSED_YANG)
    return True


def frozen_mismatches(cfg, job, attempt):
    """A frozen job runs only while a frozen batch holds the lock, with the attempt's bundle, the
    bundled source (and no other project module) and the bundled inputs."""
    folder = Path(cfg["freeze_dir"])
    if not batch_running(cfg):
        return ["no running frozen batch holds the batch lock"]
    if not (folder / "bundle.json").is_file():
        return ["no freeze bundle"]
    bad = [] if attempt.get("bundle_sha256") == sha256_file(folder / "bundle.json") else [
        "bundle: the attempt names another bundle"]
    frozen = read_json(folder / "hashes.json")["exports"]
    sources, now = read_json(folder / "sources.json")["files"], source_files(cfg)
    bad += [f"input: {f}" for f in job["inputs"].values() if sha256_file(ROOT / f) != frozen.get(f)]
    bad += [f"source: {f}" for f in sorted(set(now) | set(sources))
            if f not in now or sha256_file(now[f]) != sources.get(f)]
    bad += [f"unbundled local module: {n}" for n in local_modules() if n not in now]
    return bad


def audit_job(cfg, job, attempt=None):
    """beat_audit on one stream (all model seeds); returns its timing, memory and cell statuses.
    Any failure after the refusal check, including the frozen checks, is a failed stream."""
    frozen = guard_yang(cfg, job, attempt)
    start = time.perf_counter()
    result = dict(cohort=job["cohort"], rung=job["rung"], report=job["out"] + "/report.json",
                  peak_rss_bytes=None)
    try:
        from beat_audit import Population, audit_stream
        from beat_audit.resources import peak_memory_bytes
        settings = audit_settings(cfg)
        bad = frozen_mismatches(cfg, job, attempt) if frozen else []
        if bad:
            raise ValueError(f"refused: the frozen batch's checks failed: {bad[:5]}")
        bad = [f for s, f in job["inputs"].items() if sha256_file(ROOT / f) != job["sha256"][s]]
        if bad:
            raise ValueError(f"inputs differ from the export manifest: {bad}")
        report = audit_stream(
            {seed: str(ROOT / f) for seed, f in job["inputs"].items()},
            conditions=conditions_of(job["conditions"]), config=audit_config(settings, job),
            population=Population(users=[str(u) for u in job["users"]],
                                  folds=fold_assignment(job["fold_users"], settings["fold_seed"])),
            kind=job["kind"], out=ROOT / job["out"])
        result.update(
            status="completed", tool_runtime_seconds=report["runtime_seconds"],
            hashes=dict(source=report["hashes"]["source"]["sha256"],
                        configuration=report["hashes"]["configuration"],
                        inputs=report["hashes"]["inputs"]),
            cells=[dict(condition=c["condition"], k=c["k"], status=c["status"],
                        excluded_users=c["excluded_users"], reason=c.get("reason"))
                   for c in report["cells"]], peak_rss_bytes=peak_memory_bytes())
    except Exception as exc:  # recorded as a failed stream, never read as a non-detection
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}", cells=[])
    result.update(seconds=time.perf_counter() - start)
    return result


def gate_matches(cfg):
    """Development and frozen runs use the reductions the runtime projection requires."""
    path = Path(cfg["runtime"])
    if not path.exists():
        raise SystemExit("audit_cohorts: refused: no runtime projection; run `pilot` first")
    required = read_json(path)["required_reductions"]
    if required is None or required != list(cfg["audit"]["reductions"]):
        raise SystemExit(f"audit_cohorts: refused: audit.reductions {cfg['audit']['reductions']} "
                         f"differ from the runtime gate's required reductions {required}")


# ------------------------------------------------------------------ parity (plan §5 A3)

def parity(cfg, args):
    """The tool's closed-set module on study L's original masks, directions, seeds and data."""
    from beat_audit.closed_set import enrolled_log_likelihood, identify
    pcfg = cfg["parity_check"]
    numbers = {n["id"]: n for n in yaml.safe_load(
        (ROOT / "docs/numbers.yaml").read_text(encoding="utf-8"))["numbers"]}
    summary = pd.read_csv(ROOT / pcfg["identity_summary"])
    record = start_attempt(cfg, "development", "parity", cohorts=["lee2019"],
                           inputs={rel(ROOT / pcfg["outputs"].format(seed=s)): sha256_file(
                               ROOT / pcfg["outputs"].format(seed=s)) for s in pcfg["seeds"]})
    start, rows, warned = time.perf_counter(), [], []
    with threadpool_limits(limits=cfg["threads"]):
        for seed in pcfg["seeds"]:
            packet = joblib.load(ROOT / pcfg["outputs"].format(seed=seed))
            T, E = packet["T"].reset_index(drop=True), packet["E"].reset_index(drop=True)
            tz, ez = np.asarray(packet["tz"], float), np.asarray(packet["ez"], float)
            online = E.run.to_numpy() > 2  # beat.exp_l.scenarios: offline = runs 1-2 of session 2
            for scenario, direction, a, ua, b, ub in (
                    ("auxiliary", "T_to_E", tz, T.subject, ez, E.subject),
                    ("main", "offline_to_online", ez[~online], E.subject[~online],
                     ez[online], E.subject[online]),
                    ("main", "online_to_offline", ez[online], E.subject[online],
                     ez[~online], E.subject[~online])):
                fa, fb = np.isfinite(a).all(axis=1), np.isfinite(b).all(axis=1)  # L's attack()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = identify(a[fa], ua.to_numpy()[fa], b[fb], ub.to_numpy()[fb],
                                      kind="features", seed=seed)
                warned += [dict(seed=seed, direction=direction, category=w.category.__name__,
                                message=str(w.message)[:200]) for w in caught]
                rows += [dict(seed=seed, scenario=scenario, direction=direction, attacker=name,
                              subject=int(u), recall=value)
                         for name, accuracy in result.items() for u, value in accuracy.items()]
    frame = pd.DataFrame(rows)
    # beat.exp_l.recalls: per user over seeds and directions with equal weight, then over users.
    per_user = frame.groupby(["scenario", "attacker", "seed", "direction", "subject"]).recall.mean(
        ).groupby(["scenario", "attacker", "subject"]).mean()
    ba = per_user.groupby(["scenario", "attacker"]).mean()
    checks = {}
    for name, spec in pcfg["registered"].items():
        entry = numbers[spec["id"]]
        where = entry["value"]["where"]
        mask = np.ones(len(summary), bool)
        for column, value in where.items():
            mask &= (summary[column] == value).to_numpy()
        registered = float(summary[mask][entry["value"]["column"]].iloc[0])
        mine = float(ba[(where["scenario"], where["attacker"])])
        checks[name] = dict(
            registered=registered, registered_text=entry["text"], registered_id=spec["id"],
            source=f"{entry['value']['file']} ({where})", tool=mine, abs_diff=abs(mine - registered),
            tolerance=pcfg["tolerance"], passed=abs(mine - registered) <= pcfg["tolerance"],
            per_seed={str(s): float(g.groupby(["direction", "subject"]).recall.mean().groupby(
                "subject").mean().mean()) for s, g in frame[
                (frame.scenario == where["scenario"]) & (frame.attacker == where["attacker"])
            ].groupby("seed")})
    others = {f"{scenario}/{attacker}": dict(tool=float(value), study=float(summary[
        (summary.kind == "representation") & (summary.key == "P0") & (summary.attacker == attacker)
        & (summary.scenario == scenario)].ba.iloc[0])) for (scenario, attacker), value in ba.items()}
    from beat.exp_k import bayes_scores  # sealed reference (imports torch); fixed inputs only
    windows = np.array([[[1, 0, 2], [0, 2, 1]], [[0, 0, 0], [1, 1, 1]]])
    enrollment = np.array([[[4, 2, 1], [2, 3, 1]], [[1, 3, 2], [4, 0, 1]]])
    alpha = np.array([[0.2, 0.5, 1.3], [0.8, 0.4, 0.6]])
    likelihood = float(np.abs(enrolled_log_likelihood(windows, enrollment, alpha) - bayes_scores(
        "cat_aware", {"aware": windows}, {"aware": enrollment}, alpha)).max())
    attackers = yaml.safe_load((ROOT / pcfg["study_config"]).read_text(encoding="utf-8"))[
        "attackers"]
    passed = all(c["passed"] for c in checks.values()) and likelihood <= 1e-12
    document = dict(
        passed=passed, checks=checks, enrolled_likelihood=dict(
            max_abs_diff_vs_beat_exp_k_bayes_scores=likelihood, tolerance=1e-12,
            inputs="fixed cue-aware tables of tests/test_audit_statistics.py", passed=likelihood <= 1e-12),
        all_attackers_informative=others,
        protocol=dict(
            data="study L's stored P0 representations, " + pcfg["outputs"] + ", seeds "
                 + ",".join(map(str, pcfg["seeds"])),
            within_day="session 2: offline runs 1-2 against online runs 3-4, both directions "
                       "(beat.exp_l.scenarios 'main')",
            cross_day="enrol all of session 1, identify all of session 2 (scenario 'auxiliary')",
            masks="every stored trial; non-finite rows dropped as in beat.exp_l.attack (none)",
            statistic="per-user recall averaged over seeds and directions, then over the 54 "
                      "users (beat.exp_l.recalls)",
            tool="beat_audit.closed_set.identify (standardised logistic C = 1, max_iter 1000; "
                 "RBF-SVM C = 1, gamma = scale; MLP 100 hidden units, 500 iterations), "
                 "random_state = model seed",
            study_attackers=attackers,
            settings_difference="logistic max_iter 1000 (tool) vs 2000 (L); L ran each fit on 1 "
                                f"thread, the tool here on {cfg['threads']}; L's MLP is 32 hidden "
                                "units, lbfgs, alpha 0.01, so the MLP rows are not a parity "
                                "comparison"),
        warnings=dict(count=len(warned), by_category=pd.Series(
            [w["category"] for w in warned], dtype=object).value_counts().to_dict(),
            first=warned[:5]),
        seconds=time.perf_counter() - start)
    write_json(cfg["parity"], document)
    end_attempt(cfg, record, status="completed" if passed else "failed",
                outputs=[rel(cfg["parity"])], seconds=document["seconds"])
    for name, c in checks.items():
        print(f"{name}: tool {c['tool']:.4f} registered {c['registered']:.4f} "
              f"diff {c['abs_diff']:.4f} passed {c['passed']}")
    return 0 if passed else 1


# ------------------------------------------------------------------ runtime gate (plan §4)

PILOT_TIMED = ("identify", "_gallery_rows", "measure", "predict", "_sequential_records",
               "matching_test", "bootstrap", "_tables", "make_windows", "_cell_population",
               "load_input", "file_hash", "source_hashes")
GALLERY_PARTS = ("_gallery_rows", "measure", "predict")
SETUP_PARTS = ("load_input", "file_hash", "source_hashes")  # once per stream


def pilot_job(cfg, job, settings):
    """Fold 0 of one Lee2019 stream through beat_audit.audit_timing, one call per k and
    condition with every comparator, as the full audit runs them.

    Each comparator's timing_seconds is its time on the cell. Wrappers around the engine's
    module functions only observe: they split that time, per comparator, into the parts that the
    plan's reductions remove (gallery, permutations) or that scale differently with input size,
    and time the cell's population record, the window summaries and the stream setup.
    """
    guard_yang(cfg, job, None)  # the pilot never runs on a Yang2025 export
    from beat_audit import Population, audit_timing, comparators, engine
    from beat_audit.resources import peak_memory_bytes
    timers, current = defaultdict(lambda: defaultdict(float)), [None]

    def timed(name, function):
        def wrapper(*a, **kw):
            begin = time.perf_counter()
            try:
                return function(*a, **kw)
            finally:
                timers[current[0]][name] += time.perf_counter() - begin
        return wrapper

    def attributed(function):
        def wrapper(*a, **kw):  # _analyse(streams, blocks, users, groups, resolved, name, ...)
            current[0] = a[5]
            try:
                return function(*a, **kw)
            finally:
                current[0] = None
        return wrapper

    for name in PILOT_TIMED:
        setattr(engine, name, timed(name, getattr(engine, name)))
    engine._analyse = attributed(engine._analyse)
    comparators.Comparator.fit = timed("comparator_fit", comparators.Comparator.fit)
    inputs = {seed: str(ROOT / f) for seed, f in sorted(job["inputs"].items())}
    population = Population(users=[str(u) for u in job["users"]],
                            folds=fold_assignment(job["fold_users"], settings["fold_seed"]))
    rows, begin = [], time.perf_counter()
    for k in settings["k_grid"]:
        for condition in conditions_of(job["conditions"]):
            timers.clear()
            t = time.perf_counter()
            report = audit_timing(inputs, folds=(0,), conditions=[condition],
                                  config=audit_config(dict(settings, k_grid=[k]), job),
                                  population=population, kind=job["kind"])
            call = time.perf_counter() - t
            cell, outside = report["cells"][0], timers[None]
            shared = dict(
                condition=condition.name, relation=condition.relation, k=k,
                pairs=len(condition.pairs), seeds=len(inputs), users=len(job["users"]),
                eligible=len(cell["analysed_users"]), status=cell["status"],
                reason=cell.get("reason"), call_seconds=call,
                windows_seconds=outside["make_windows"],
                population_seconds=outside["_cell_population"],
                setup_seconds=sum(outside[p] for p in SETUP_PARTS))
            for name in attacks(job["kind"], settings, k):
                value = cell["comparators"].get(name, {})
                rows.append(dict(shared, comparator=name, seconds=value.get("timing_seconds"),
                                 fold_users=sum(len(g["users"])
                                                for g in value.get("partner_auc", [])),
                                 parts=dict(timers[name])))
    return dict(cohort=job["cohort"], rung=job["rung"], kind=job["kind"], seeds=len(inputs),
                shape=job["shape"], fold=0, fold_users=max(r["fold_users"] for r in rows),
                rows=rows, seconds=time.perf_counter() - begin,
                peak_rss_bytes=peak_memory_bytes())


def side_rows(entry, selectors):
    """Mean rows per user on one physical side; a segment holds about half of its day."""
    per_day = {d: float(np.mean(list(v.values()))) for d, v in entry["rows_per_user_day"].items()}
    return sum(per_day[str(s).split("/")[0]] / (2 if "/" in str(s) else 1) for s in selectors)


def pair_rows(entry, spec):
    return float(np.mean([side_rows(entry, p["a"]) + side_rows(entry, p["b"])
                          for p in spec["pairs"]]))


def required_rows(entry, spec):
    """Rows per user over the distinct selectors that a condition's pairs require."""
    return side_rows(entry, sorted({s for p in spec["pairs"] for s in p["a"] + p["b"]}))


def cell_excluded(entry, spec, k):
    """Users without a k-window in a required segment (the export manifest's short segments)."""
    required = set()
    for pair in spec["pairs"]:
        for s in pair["a"] + pair["b"]:
            d, _, seg = str(s).partition("/")
            required |= {(int(d), int(seg))} if seg else {(int(d), 1), (int(d), 2)}
    short = entry["segments_shorter_than_k"].get(str(k), {}).get("segments", [])
    return {u for u, d, s, _ in short if (d, s) in required}


def pilot_type(job):
    if job["kind"] == "commands":
        return "commands"
    return "features_8d" if job["shape"][1] <= 8 else "features_full"


def project(jobs, entries, pilots, settings, limits, count, permutations):
    """Projected minutes and peak memory per stream; linear in rows per side, quadratic for the
    closed-set fits (SVM), proportional to model seeds, pairs, directions and folds."""
    out = []
    for job in jobs:
        pilot = pilots[pilot_type(job)]
        entry, pentry = entries[job["inputs"]["0"]], entries[pilot["job"]["inputs"]["0"]]
        users, seeds = len(job["users"]), len(job["inputs"])
        cells, seconds = [], 0.0
        for condition, spec in job["conditions"].items():
            for k in settings["k_grid"]:
                excluded = cell_excluded(entry, spec, k)
                # The cell's population record is made before the exclusion check, once per cell.
                row = next(r for r in pilot["rows"] if r["relation"] == spec["relation"]
                           and r["k"] == k)
                pspec = pilot["job"]["conditions"][row["condition"]]
                seconds += row["population_seconds"] * (
                    required_rows(entry, spec) * users * seeds) / (
                    required_rows(pentry, pspec) * row["users"] * row["seeds"])
                for name in attacks(job["kind"], settings, k):
                    cell = dict(condition=condition, k=k, comparator=name)
                    if len(excluded) > 0.1 * users:
                        out_seconds = 0.0
                    else:
                        row = next(r for r in pilot["rows"] if r["status"] == "ok"
                                   and r["relation"] == spec["relation"] and r["k"] == k
                                   and r["comparator"] == name)
                        pspec = pilot["job"]["conditions"][row["condition"]]
                        r = pair_rows(entry, spec) / pair_rows(pentry, pspec)
                        parts = row["parts"]
                        gallery = sum(parts.get(p, 0.0) for p in GALLERY_PARTS)
                        rest = row["seconds"] - sum(parts.values())
                        per_component = (rest * r + parts.get("_tables", 0.0) * r
                                         + parts.get("_sequential_records", 0.0) * r
                                         + parts.get("identify", 0.0) * r ** 2
                                         + (gallery * r if settings["gallery"] else 0.0)) / (
                            row["seeds"] * row["pairs"] * 2)
                        folds = min(5, (users - len(excluded)) // 3)
                        per_fold = (per_component * seeds * len(spec["pairs"]) * 2
                                    + parts.get("comparator_fit", 0.0) * r * seeds / row["seeds"]
                                    + parts.get("matching_test", 0.0) * settings["permutations"]
                                    / permutations + parts.get("bootstrap", 0.0))
                        out_seconds = folds * per_fold
                    cells.append(dict(cell, excluded=len(excluded), seconds=out_seconds))
                    seconds += out_seconds
        # Window summaries (mean, and rich when a rich comparator runs) once per k; loading and
        # hashing the inputs once per stream.
        windows = sum(max(r["windows_seconds"] for r in pilot["rows"] if r["k"] == k)
                      for k in settings["k_grid"])
        setup = min(r["setup_seconds"] for r in pilot["rows"])
        size = np.prod(job["shape"]) * seeds / (np.prod(pilot["job"]["shape"]) * pilot["seeds"])
        seconds += (windows + setup) * size
        out.append(dict(cohort=job["cohort"], rung=job["rung"], minutes=seconds / 60,
                        peak_gb=pilot["peak_rss_bytes"] / 2**30 * max(1.0, float(size)),
                        pilot=pilot_type(job), cells=cells))
    batch_hours = yang_batch_hours(out, count)
    over = over_limits(out, batch_hours, limits)
    return dict(streams=[{k: v for k, v in s.items() if k != "cells"} for s in out],
                cells=[dict(cohort=s["cohort"], rung=s["rung"], **c) for s in out
                       for c in s["cells"]],
                yang2025_batch_hours=batch_hours, total_hours_serial=sum(
                    s["minutes"] for s in out) / 60, exceeds=over, within_limits=not over)


def yang_batch_hours(streams, count):
    """The frozen batch's schedule: streams in jobs.json order (the order of `streams`, as
    audit_jobs sorts them), each started on the first of `count` jobs to become free, as
    run_pool does."""
    lanes = [0.0] * count
    for minutes in (s["minutes"] for s in streams if s["cohort"] == "yang2025"):
        lanes[lanes.index(min(lanes))] += minutes
    return max(lanes) / 60


def over_limits(streams, batch_hours, limits):
    over = [f"{s['cohort']}/{s['rung']}: {s['minutes']:.1f} min" for s in streams
            if s["minutes"] > limits["stream_minutes"]]
    over += [f"{s['cohort']}/{s['rung']}: {s['peak_gb']:.2f} GB" for s in streams
             if s["peak_gb"] > limits["stream_gb"]]
    if batch_hours > limits["yang2025_batch_hours"]:
        over.append(f"yang2025 batch: {batch_hours:.2f} h")
    return over


def gate_projection(cfg, manifest, pilots, reductions, permutations):
    """The pilot's projection of every cohort's job manifest under `reductions`."""
    acfg = cfg["audit"]
    jobs = audit_jobs(cfg, ["bnci2014_001", "lee2019", "yang2025"], manifest)
    entries = {e["file"]: e for e in manifest["exports"]}
    return dict(reductions=list(reductions), **project(
        jobs, entries, pilots, reduce_settings(acfg["settings"], reductions), acfg["limits"],
        workers(cfg), permutations))


def stored_pilots(cfg, document, manifest):
    """The pilot measurements of the runtime projection, as `project` reads them."""
    lee = {j["rung"]: j for j in audit_jobs(cfg, ["lee2019"], manifest)}
    return {p["type"]: dict(p, job=lee[p["stream"].split("/", 1)[1]],
                            peak_rss_bytes=p["peak_gb"] * 2**30) for p in document["pilot"]}


def projection_for(document, reductions):
    found = next((p for p in document["projections"] if p["reductions"] == list(reductions)),
                 None)
    if found is None:
        raise SystemExit(f"audit_cohorts: the runtime projection has no option with reductions "
                         f"{list(reductions)}; run `pilot` again")
    return found


def development_check(cfg, document):
    """The gate's projection against the last completed development run's observed stream times
    (which include report writing), each compared with the projection of the reductions that
    run used; the Yang2025 streams of the required reductions are rescaled by the largest
    observed/projected ratio among the development streams of the same type, never below 1,
    and checked against the configured limits."""
    attempts = read_attempts(cfg)
    attempt = next((r for r in reversed(attempts) if r["command"] == "develop"
                    and r["event"] == "end" and r["status"] == "completed"), None)
    if attempt is None:
        raise SystemExit("audit_cohorts: no completed development run in the attempts log")
    start = next((r for r in attempts if r["attempt"] == attempt["attempt"]
                  and r["event"] == "start"), None)
    if start is None:
        raise SystemExit(f"audit_cohorts: development run {attempt['attempt']} has no start "
                         "record with its reductions")
    used = start["reductions"]
    final = projection_for(document, document["required_reductions"])
    projected = {(s["cohort"], s["rung"]): s for s in projection_for(document, used)["streams"]}
    rows = []
    for r in attempt["results"]:
        p = projected[(r["cohort"], r["rung"])]
        rows.append(dict(cohort=r["cohort"], rung=r["rung"], type=p["pilot"],
                         observed_minutes=r["seconds"] / 60, projected_minutes=p["minutes"],
                         ratio=r["seconds"] / 60 / p["minutes"],
                         observed_peak_gb=r["peak_rss_bytes"] / 2**30,
                         projected_peak_gb=p["peak_gb"]))
    ratio = {}
    for row in rows:
        ratio[row["type"]] = max(ratio.get(row["type"], 1.0), row["ratio"])
    yang = [dict(s, minutes=s["minutes"] * ratio[s["pilot"]], factor=ratio[s["pilot"]])
            for s in final["streams"] if s["cohort"] == "yang2025"]
    batch = yang_batch_hours(yang, workers(cfg))
    over = over_limits(yang, batch, cfg["audit"]["limits"])
    return dict(
        attempt=attempt["attempt"], calibrated_on=list(used), reductions=final["reductions"],
        streams=rows, factor_by_type=ratio, yang2025_adjusted=yang, yang2025_batch_hours=batch,
        exceeds=over, within_limits=not over, rule=" ".join(development_check.__doc__.split()))


def runtime_check(cfg):
    """The development-calibrated gate (plan §4): while the adjusted projection exceeds a limit,
    the next reduction in the plan's order is applied, projected from the stored pilot and
    logged in the runtime projection; `audit.reductions` must then name the new prefix."""
    document = read_json(cfg["runtime"])
    check = development_check(cfg, document)
    while not check["within_limits"] and len(document["required_reductions"]) < len(REDUCTIONS):
        reductions = list(REDUCTIONS[:len(document["required_reductions"]) + 1])
        if not any(p["reductions"] == reductions for p in document["projections"]):
            manifest = read_manifest(cfg)
            document["projections"].append(gate_projection(
                cfg, manifest, stored_pilots(cfg, document, manifest), reductions,
                document["settings_measured"]["permutations"]))
        document.setdefault("development_reductions", []).append(dict(
            attempt=check["attempt"], exceeds=check["exceeds"], applied=reductions[-1]))
        document["required_reductions"] = reductions
        check = development_check(cfg, document)
    document["development_check"] = check
    document["status"] = ("within limits" if check["within_limits"] else
                          "blocked: the development-adjusted projection exceeds the limits after "
                          "every reduction")
    write_json(cfg["runtime"], document)
    return check


def check_runtime(cfg, args):
    result = runtime_check(cfg)
    print(f"required reductions {result['reductions']} (development run calibrated on "
          f"{result['calibrated_on']}); audit.reductions {cfg['audit']['reductions']}")
    for s in result["yang2025_adjusted"]:
        print(f"yang2025 {s['rung']:28s} {s['minutes']:6.1f} min (x{s['factor']:.2f})")
    print(f"yang2025 batch {result['yang2025_batch_hours']:.2f} h; exceeds {result['exceeds']}")
    return 0 if result["within_limits"] else 1


def pilot(cfg, args):
    """Runtime gate: one Lee2019 fold per stream type, projected to the whole job manifest."""
    acfg, manifest = cfg["audit"], read_manifest(cfg)
    settings = reduce_settings(acfg["settings"], [])  # the pilot measures the full procedure
    lee = {j["rung"]: j for j in audit_jobs(cfg, ["lee2019"], manifest)}
    chosen = {pilot_type(lee[r]): lee[r] for r in acfg["pilot_rungs"]}
    record = start_attempt(cfg, "development", "pilot", cohorts=["lee2019"],
                           streams=[j["rung"] for j in chosen.values()], fold=0, settings=settings)
    results = run_pool(pilot_job, [(cfg, job, settings) for job in chosen.values()],
                       workers(cfg))
    pilots = {pilot_type(job): dict(result, job=job) for job, result in zip(chosen.values(),
                                                                            results)}
    limits, count = acfg["limits"], workers(cfg)
    options, required = [], None
    for n in range(len(REDUCTIONS) + 1):
        options.append(gate_projection(cfg, manifest, pilots, REDUCTIONS[:n],
                                       settings["permutations"]))
        if options[-1]["within_limits"]:
            required = options[-1]["reductions"]
            break
    document = dict(
        pilot=[dict(stream=f"{p['cohort']}/{p['rung']}", type=t, fold=0,
                    fold_users=p["fold_users"], seeds=p["seeds"], seconds=p["seconds"],
                    peak_gb=p["peak_rss_bytes"] / 2**30, rows=p["rows"])
               for t, p in pilots.items()],
        settings_measured=settings, limits=limits, max_parallel_jobs=count,
        threads_per_job=settings["threads"], reduction_order=list(REDUCTIONS),
        required_reductions=required,
        status="within limits" if required is not None else "blocked: limits exceeded after "
                                                            "every reduction",
        projections=options,
        assumptions=[
            "Timings: beat_audit.audit_timing on fold 0, one call per k and condition with every "
            "comparator (each comparator's timing_seconds); observation-only wrappers around "
            "the engine's module functions split it, per comparator, into closed-set fits, "
            "gallery, permutation test, bootstrap and comparator fit, and time the cell's "
            "population record, the window summaries and the input loading and hashing.",
            "Population record: once per cell (also for cells that are not evaluable), linear "
            "in the rows of the required segments x users x model seeds. Window summaries: once "
            "per k; loading and hashing: once per stream; both linear in input size.",
            "Cost of a cell = folds x (model seeds x pairs x 2 directions x per-component cost + "
            "model seeds x comparator fit + permutation test + bootstrap), each from the pilot "
            "fold of the same stream type (full features, 8-d features or commands), relation, "
            "k and comparator.",
            "Per-component cost scales linearly with the rows per user on the two sides of a "
            "pair, the closed-set fits quadratically (RBF-SVM); fold sizes are taken as in the "
            "pilot (an overestimate for 2a's folds of three users).",
            "Peak memory scales with input size (rows x dimensions x model seeds) when larger "
            "than the pilot's; report writing is not part of the pilot and is checked against "
            "the development runs' observed times.",
            "Yang2025 batch: its streams in jobs.json order, each started on the first of the "
            "allowed parallel jobs to become free, as frozen-batch runs them."],
        seconds=sum(p["seconds"] for p in pilots.values()))
    write_json(cfg["runtime"], document)
    end_attempt(cfg, record, status="completed", outputs=[rel(cfg["runtime"])],
                required_reductions=required,
                peak_rss_bytes={t: p["peak_rss_bytes"] for t, p in pilots.items()})
    final = options[-1]
    for s in final["streams"]:
        print(f"{s['cohort']:13s} {s['rung']:28s} {s['minutes']:7.1f} min {s['peak_gb']:.2f} GB")
    print(f"reductions {required}; yang2025 batch {final['yang2025_batch_hours']:.2f} h; "
          f"exceeds {final['exceeds']}")
    return 0 if required is not None else 1


# ------------------------------------------------------------------ development runs and summary

def develop(cfg, args):
    cohorts = args.cohort or list(DEVELOPMENT)
    if not set(cohorts) <= set(DEVELOPMENT):
        raise SystemExit(REFUSED_YANG)
    gate_matches(cfg)
    jobs = [j for j in audit_jobs(cfg, cohorts) if not args.rung or j["rung"] in args.rung]
    jobs.sort(key=lambda j: np.prod(j["shape"]) * len(j["inputs"]), reverse=True)
    record = start_attempt(cfg, "development", "develop", cohorts=cohorts,
                           streams=[f"{j['cohort']}/{j['rung']}" for j in jobs],
                           settings=audit_settings(cfg), reductions=cfg["audit"]["reductions"],
                           inputs={f: s for j in jobs for f, s in zip(j["inputs"].values(),
                                                                      j["sha256"].values())})
    results = run_pool(audit_job, [(cfg, job) for job in jobs], workers(cfg),
                       failed=stream_failure)
    failed = [r for r in results if r["status"] != "completed"
              or any(c["status"] == "failed" for c in r["cells"])]
    end_attempt(cfg, record, status="failed" if failed else "completed", results=results)
    write_summary(cfg)
    if not failed:
        runtime_check(cfg)
    for r in results:
        print(f"{r['cohort']:13s} {r['rung']:28s} {r['status']:9s} "
              f"{(r['seconds'] or 0) / 60:6.1f} min {(r['peak_rss_bytes'] or 0) / 2**30:.2f} GB")
    return 1 if failed else 0


def role_of(cfg, cohort, rung, condition, k, comparator):
    if cohort != "yang2025":
        return "development"
    preds = cfg["predictions"]
    for name in ("V1", "V2"):
        p = preds[name]
        if (rung, condition, k, comparator) == (p["rung"], p["condition"], p["k"],
                                                p["comparator"]):
            return name
    v1, e = preds["V1"], preds["estimate"]
    if (rung, k, comparator) == (v1["rung"], v1["k"], v1["comparator"]) and condition in (
            e["same_day"], e["cross_day"]):
        return "pre-specified estimate (paired contrast)"
    return "estimate without verdict"


def summary_rows(cfg, job, report):
    settings = report["configuration"]["settings"]
    note = cfg["audit"]["day1_notes"].get(job["cohort"], {}).get(job["rung"], "")
    rows = []
    for cell in report["cells"]:
        skipped = cell.get("skipped_comparators", {})  # rich summaries at k = 1 (plan §3)
        names = list(cell["comparators"]) or attacks(job["kind"], settings, cell["k"])
        for name in names + [n for n in skipped if n not in names]:
            v = cell["comparators"].get(name, {})
            m, seq, gallery = v.get("metrics", {}), v.get("sequential", {}), v.get("gallery")
            interval = v.get("interval", [None, None])
            rows.append(dict(
                cohort=job["cohort"], rung=job["rung"], kind=job["kind"], tau=job["tau"],
                model_seeds=" ".join(report["model_seeds"]), condition=cell["condition"],
                relation=cell["relation"], days=" ".join(cell["days"]), k=cell["k"],
                comparator=name, status="skipped" if name in skipped else cell["status"],
                reason=skipped.get(name, cell.get("reason", "")),
                users=len(cell["analysed_users"]) + len(cell["excluded_users"]),
                users_analysed=len(cell["analysed_users"]),
                excluded_users=" ".join(cell["excluded_users"]), auc=v.get("auc"),
                ci_low=interval[0], ci_high=interval[1], interval_unreliable=v.get("unreliable"),
                permutation_p=v.get("p"), permutation_draws=v.get("draws"),
                holm_p_over_k=v.get("holm_p"), wording=v.get("wording", ""),
                strongest=cell.get("strongest") == name if v else None,
                strongest_comparator=cell.get("strongest"),
                smallest_detected_k=report["smallest_detected_k"].get(cell["condition"]),
                eer=m.get("eer"), tpr_at_fpr_01=m.get("tpr_at_fpr_01"),
                tpr_at_fpr_10=m.get("tpr_at_fpr_10"), d_sys=m.get("D_sys"), cllr=m.get("Cllr"),
                cllr_min=m.get("Cllr_min"), deployed_fmr=m.get("development_threshold_FMR"),
                deployed_fnmr=m.get("development_threshold_FNMR"),
                seq_share_decided=seq.get("share_decided"),
                seq_median_commands=seq.get("median_commands_to_decision"),
                seq_fmr=seq.get("FMR"), seq_fnmr=seq.get("FNMR"),
                seq_undecided_mated=seq.get("mated", {}).get("undecided"),
                seq_undecided_nonmated=seq.get("nonmated", {}).get("undecided"),
                gallery_check=("" if not v else "not run" if gallery is None else
                               "passed" if all(f["supported"] for f in gallery["folds"]) else
                               "independence model not supported"),
                closed_set=" ".join(f"{a}={x['accuracy']:.4f}"
                                    for a, x in v.get("closed_set", {}).items()),
                seconds_per_output=job["seconds_per_output"],
                day1_note=note if "1" in cell["days"] else "", tier=TIER[job["cohort"]],
                role=role_of(cfg, job["cohort"], job["rung"], cell["condition"], cell["k"], name),
                population_hash=cell.get("population_hash"),
                report=f"{job['out']}/report.json"))
    return rows


def write_summary(cfg):
    """summary.csv of the development cohorts. It never opens a Yang2025 report: those are
    summarised only by the frozen batch that verified them (close_batch)."""
    rows, manifest = [], read_manifest(cfg)
    for job in audit_jobs(cfg, list(DEVELOPMENT), manifest):
        path = ROOT / job["out"] / "report.json"
        if path.exists():
            rows += summary_rows(cfg, job, read_json(path))
    write_csv(pd.DataFrame(rows), cfg["summary"])
    return rows


def summary(cfg, args):
    print(f"{len(write_summary(cfg))} summary rows")
    return 0


# ------------------------------------------------------------------ V1 selection and job manifest

def select_v1(cfg):
    """V1's comparator: the configured rule applied to the Lee2019 development report."""
    sel = cfg["predictions"]["V1"]["selection"]
    if sel["cohort"] not in DEVELOPMENT:
        raise SystemExit(f"audit_cohorts: refused: V1's comparator is selected on a development "
                         f"cohort {list(DEVELOPMENT)}, not {sel['cohort']}; nothing was read")
    v1 = cfg["predictions"]["V1"]
    source = dict(cohort=V1_SELECTION_COHORT, rung=v1["rung"], condition=v1["condition"],
                  k=v1["k"])
    given = {name: sel.get(name) for name in source}
    if given != source:
        raise SystemExit(f"audit_cohorts: refused: V1's comparator is selected on {source} "
                         f"(V1's own cell on {V1_SELECTION_COHORT}), not {given}; "
                         "nothing was read")
    base = list(cfg["audit"]["settings"]["comparators"])
    order = [n for n in SIMPLICITY if n in base]
    if set(base) - set(SIMPLICITY) or list(sel["order"]) != order:
        raise SystemExit(f"audit_cohorts: refused: predictions.V1.selection.order must list "
                         f"every configured feature comparator once in the simplicity order "
                         f"{order}, not {list(sel['order'])}; nothing was read")
    job = next((j for j in audit_jobs(cfg, [sel["cohort"]]) if j["rung"] == sel["rung"]), None)
    path = ROOT / job["out"] / "report.json" if job else None
    if path is None or not path.exists():
        raise SystemExit(f"audit_cohorts: no development report for {sel['cohort']} "
                         f"{sel['rung']}; run `develop` first")
    cell = next((c for c in read_json(path)["cells"]
                 if c["condition"] == sel["condition"] and c["k"] == sel["k"]), None)
    if cell is None or cell["status"] != "ok":
        raise SystemExit(f"audit_cohorts: {sel['cohort']} {sel['condition']} k={sel['k']} is not "
                         "evaluable; V1's comparator cannot be named by the rule")
    surviving = audit_settings(cfg)["comparators"]
    missing = [n for n in order if n in surviving and n not in cell["comparators"]]
    if missing:
        raise SystemExit(f"audit_cohorts: the development report lacks the surviving "
                         f"comparators {missing}; V1's comparator cannot be named by the rule")
    values = {n: cell["comparators"][n]["auc"] for n in order if n in surviving}
    best = max(values.values())
    chosen = next(n for n in order if values.get(n) == best)
    return dict(rule=sel["rule"], cohort=sel["cohort"], rung=sel["rung"],
                condition=sel["condition"], k=sel["k"],
                not_compared={n: f"removed by the runtime gate's reductions "
                                 f"{cfg['audit']['reductions']}"
                              for n in order if n not in values},
                mean_per_user_auc=values,
                intervals={n: cell["comparators"][n]["interval"] for n in values},
                comparator=chosen, report=rel(path), report_sha256=sha256_file(path))


def select(cfg, args):
    result = select_v1(cfg)
    named = cfg["predictions"]["V1"]["comparator"]
    result.update(configured=named, agrees=named == result["comparator"])
    write_json(cfg["v1_selection"], result)
    print(json.dumps({k: result[k] for k in ("mean_per_user_auc", "comparator", "configured")}))
    return 0 if named is None or result["agrees"] else 1


def folds_document(cfg, jobs):
    settings = audit_settings(cfg)
    assignment = fold_assignment(jobs[0]["fold_users"], settings["fold_seed"])
    return dict(
        cohort=jobs[0]["cohort"], users=sorted(assignment, key=int), assignment=assignment,
        folds=[sorted((u for u, f in assignment.items() if f == g), key=int)
               for g in sorted(set(assignment.values()))],
        rule="min(5, U // 3) folds of numpy default_rng(fold_seed).permutation of the sorted "
             "string labels (plan §3); passed to beat_audit as explicit Population folds, so a "
             "user excluded from a cell leaves the others' folds unchanged",
        sampling_seeds=dict(
            fold_seed=settings["fold_seed"], seed=settings["seed"],
            derived="beat_audit.engine: comparator, pair-training and closed-set seed = seed + "
                    "1009 x fold + 101 x model-seed index; comparison and gallery draws = that "
                    "seed + 37 x pair + 17 x direction; matching permutations = seed; subject "
                    "bootstrap and contrasts = seed + 1"))


def sampling_seeds(document):
    """Every sampling seed of the batch as a number, by beat_audit.engine's derivation; the
    seeds do not depend on condition, k or comparator."""
    settings, seed = document["settings"], document["settings"]["seed"]
    folds = sorted(set(document["folds"]["assignment"].values()))
    streams = []
    for job in document["streams"]:
        pairs = {name: len(spec["pairs"]) for name, spec in job["conditions"].items()}
        streams.append(dict(rung=job["rung"], pairs_per_condition=pairs, components=[
            dict(fold=f, model_seed=name, attack_seed=seed + 1009 * f + 101 * i,
                 draws=[dict(pair=p, direction=d,
                             seed=seed + 1009 * f + 101 * i + 37 * p + 17 * j)
                        for p in range(max(pairs.values()))
                        for j, d in enumerate(("A->B", "B->A"))])
            for f in folds for i, name in enumerate(sorted(job["inputs"]))]))
    return dict(
        fold_seed=settings["fold_seed"], seed=seed, matching_permutations=seed,
        bootstrap_and_contrasts=seed + 1,
        attack_seed="comparator fit, pair-training draws and closed-set fits of one fold and "
                    "model seed (index i of the model-seed label among the stream's labels "
                    "sorted as strings, the order in which beat_audit.engine fits them)",
        draw_seed="comparison-table and gallery draws of one pair and direction",
        fold="position of the fold among a cell's folds (the labels of folds.json while every "
             "fold keeps an analysed user)",
        derived=document["folds"]["sampling_seeds"]["derived"], streams=streams)


def expected_cells(cfg, jobs):
    settings = audit_settings(cfg)
    cells = []
    for job in jobs:
        for condition, spec in job["conditions"].items():
            for k in settings["k_grid"]:
                for name in attacks(job["kind"], settings, k):
                    cells.append(dict(
                        cohort=job["cohort"], rung=job["rung"], kind=job["kind"],
                        tau=job["tau"], model_seeds=sorted(job["inputs"], key=int),
                        condition=condition, relation=spec["relation"],
                        # direction labels as in the components of beat_audit's report
                        components=[dict(model_seed=seed, day_pair=f"{'+'.join(p['a'])} vs "
                                         f"{'+'.join(p['b'])}", side_a=p["a"], side_b=p["b"],
                                         direction=direction)
                                    for seed in sorted(job["inputs"], key=int)
                                    for p in spec["pairs"] for direction in ("A->B", "B->A")],
                        k=k, comparator=name,
                        role=role_of(cfg, job["cohort"], job["rung"], condition, k, name)))
    return cells


def jobs_document(cfg, manifest=None, draft=True):
    jobs = audit_jobs(cfg, ["yang2025"], manifest)
    cells = expected_cells(cfg, jobs)
    return dict(draft=draft, cohort="yang2025", settings=audit_settings(cfg),
                reductions=list(cfg["audit"]["reductions"]),
                max_parallel_jobs=workers(cfg), folds=folds_document(cfg, jobs),
                not_run_by_design="rich-summary comparators at k = 1 (plan §3: the rich summary "
                                  "equals the mean there; the tool skips them)",
                predictions=cfg["predictions"], streams=jobs, cells=cells, cell_count=len(cells))


def jobs(cfg, args):
    document = jobs_document(cfg)
    write_json(cfg["jobs_draft"], document)
    print(f"{len(document['streams'])} Yang2025 streams, {document['cell_count']} cells; "
          f"V1 comparator {cfg['predictions']['V1']['comparator']}")
    return 0


# ------------------------------------------------------------------ freeze and frozen batch

def zip_sources(files, path):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, file in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, file.read_bytes())


def pip_freeze():
    return subprocess.check_output([sys.executable, "-m", "pip", "freeze"], cwd=ROOT, text=True)


def frozen_hashes(manifest):
    """Every exported input (with its key sidecar), the archives, final models and day-1 files."""
    def role(prefix):
        return {p: s["sha256"] for p, s in sorted(manifest["sources"].items())
                if any(r.startswith(prefix) for r in s["roles"])}
    exports = {}
    for e in manifest["exports"]:
        exports[e["file"]] = e["sha256"]
        exports[e["trial_keys"]["file"]] = e["trial_keys"]["sha256"]
    return dict(exports=exports,
                yang2025_archives={a["path"]: a["sha256"] for a in manifest["yang2025_archives"]},
                final_models=role("M final model"),
                day1_crossfitted=role("M day-1 cross-fitted logits"))


def hash_mismatches(groups):
    return [f"{group}: {path}" for group, files in groups.items() for path, sha in files.items()
            if not (ROOT / path).is_file() or sha256_file(ROOT / path) != sha]


def build_bundle(cfg, manifest, selection, runtime):
    folder = Path(cfg["freeze_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    files = source_files(cfg)
    zip_sources(files, folder / "source.zip")
    write_json(folder / "sources.json", dict(
        files={n: sha256_file(p) for n, p in files.items()}, zip="source.zip",
        zip_sha256=sha256_file(folder / "source.zip")))
    (folder / "pip_freeze.txt").write_text(pip_freeze(), encoding="utf-8", newline="\n")
    write_json(folder / "hashes.json", frozen_hashes(manifest))
    document = jobs_document(cfg, manifest, draft=False)
    write_json(folder / "folds.json", document["folds"])
    write_json(folder / "seeds.json", sampling_seeds(document))
    write_json(folder / "predictions.json", dict(cfg["predictions"], v1_selection=selection,
                                                 runtime_gate=runtime))
    write_json(folder / "jobs.json", document)
    write_json(folder / "bundle.json", dict(
        created_utc=utc(), git_head=git_head(), tier=TIER["yang2025"],
        note="Freeze of plan v3 §4: commit with tag freeze-5, pushed before `frozen-batch`",
        files={n: sha256_file(folder / n) for n in BUNDLE_FILES}))


def verify_bundle(cfg, bundle):
    """Every hash of the bundle against the current bytes; returns the mismatches."""
    folder = Path(cfg["freeze_dir"])
    bad = [f"bundle file: {n}" for n, sha in bundle["files"].items()
           if not (folder / n).is_file() or sha256_file(folder / n) != sha]
    bad += [f"bundle file missing from index: {n}" for n in BUNDLE_FILES if n not in bundle["files"]]
    if bad:
        return bad
    frozen, now = read_json(folder / "sources.json")["files"], source_files(cfg)
    bad += [f"source: {n}" for n in sorted(set(frozen) | set(now))
            if n not in now or n not in frozen or sha256_file(now[n]) != frozen[n]]
    bad += [f"unbundled local module: {n}" for n in local_modules() if n not in now]
    if pip_freeze() != (folder / "pip_freeze.txt").read_text(encoding="utf-8"):
        bad.append("environment: pip freeze differs")
    bad += hash_mismatches(read_json(folder / "hashes.json"))
    if not bad and json.loads(json.dumps(jobs_document(cfg, draft=False))) != read_json(
            folder / "jobs.json"):
        bad.append("jobs.json differs from the job manifest of the current configuration")
    return bad


def freeze(cfg, args):
    """Build results/audit/freeze/ (plan §4 Freeze); refuses on any inconsistency."""
    folder, attempts = Path(cfg["freeze_dir"]), read_attempts(cfg)
    if (folder / "bundle.json").exists() and not args.replace:
        raise SystemExit(f"audit_cohorts: refused: {rel(folder)} holds a bundle; use --replace "
                         "only if no frozen batch has used it")
    if any(a["kind"] in FROZEN_KINDS for a in attempts):
        raise SystemExit("audit_cohorts: refused: a frozen batch has run; a new freeze would be "
                         "a deviation (plan §4 Attempts)")
    touched = [a["attempt"] for a in attempts if "yang2025" in a.get("cohorts", [])]
    if touched:
        raise SystemExit(f"audit_cohorts: refused: attempts on Yang2025 before the freeze: {touched}")
    manifest = read_manifest(cfg)
    hashes = frozen_hashes(manifest)
    counts = {group: len(files) for group, files in hashes.items()}
    if counts != cfg["freeze_expect"]:
        raise SystemExit(f"audit_cohorts: refused: hashed inputs {counts} differ from "
                         f"freeze_expect {cfg['freeze_expect']}")
    bad = hash_mismatches(hashes)
    if bad:
        raise SystemExit(f"audit_cohorts: refused: {len(bad)} inputs differ from the export "
                         f"manifest: {bad[:5]}")
    unlisted = unlisted_predictions(cfg, jobs_document(cfg, manifest, draft=False))
    if unlisted:
        raise SystemExit(f"audit_cohorts: refused: prediction or estimate cells not listed "
                         f"exactly once in the job manifest: {unlisted}")
    gate_matches(cfg)
    check = development_check(cfg, read_json(cfg["runtime"]))
    if not check["within_limits"]:
        raise SystemExit(f"audit_cohorts: refused: the runtime projection adjusted by development "
                         f"run {check['attempt']} exceeds the limits: {check['exceeds']}")
    selection = select_v1(cfg)
    if cfg["predictions"]["V1"]["comparator"] != selection["comparator"]:
        raise SystemExit(f"audit_cohorts: refused: V1 comparator in the config "
                         f"({cfg['predictions']['V1']['comparator']}) differs from the rule's "
                         f"{selection['comparator']}")
    if args.dry_run:
        print(f"freeze dry run: every check passed; hashed inputs {counts}; V1 comparator "
              f"{selection['comparator']}; nothing was written")
        return 0
    build_bundle(cfg, manifest, selection, dict(
        file=rel(cfg["runtime"]), sha256=sha256_file(cfg["runtime"]),
        required_reductions=read_json(cfg["runtime"])["required_reductions"],
        development_check={k: check[k] for k in ("attempt", "factor_by_type",
                                                 "yang2025_batch_hours", "within_limits")}))
    print(f"freeze bundle {rel(folder)}: {counts}; commit it with tag freeze-5 and push before "
          "`frozen-batch`")
    return 0


def holm_decision(pvalues, alpha):
    """Holm over the family; a non-evaluable prediction (None) stays in it with p = 1."""
    names = list(pvalues)
    p = np.array([1.0 if pvalues[n] is None else float(pvalues[n]) for n in names])
    adjusted, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return {n: dict(p=float(p[i]), evaluable=pvalues[n] is not None, holm_p=float(adjusted[i]),
                    supported=bool(adjusted[i] <= alpha)) for i, n in enumerate(names)}


def find_cell(report, condition, k):
    if report is None:
        return None
    return next((c for c in report["cells"] if c["condition"] == condition and c["k"] == k), None)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def cell_key(cell):
    return {k: cell[k] for k in ("rung", "condition", "k", "comparator", "role")}


def physical(selectors, job):
    """A side's (day, segment) selections: "d" is every exported segment of day d, "d/s" one."""
    out = []
    for selector in map(str, selectors):
        day, _, segment = selector.partition("/")
        out += [(day, segment)] if segment else [(day, s) for s in job["segments"]]
    return out


def stream_problems(job, settings, report, frozen):
    """A report's model seeds, configuration and recorded hashes against jobs.json and the
    bundle (source files of beat_audit/ and the exported inputs)."""
    from dataclasses import asdict
    from beat_audit.io import digest
    hashes, configuration = report.get("hashes") or {}, report.get("configuration") or {}
    executed = json.loads(json.dumps(dict(
        settings=audit_config(settings, job).to_dict(), kind=job["kind"],
        conditions=[asdict(c) for c in conditions_of(job["conditions"])])))
    source = {Path(n).name: sha for n, sha in frozen["sources"].items()
              if Path(n).parent.as_posix() == "beat_audit" and n.endswith(".py")}
    problems = []
    if report.get("model_seeds") != sorted(job["inputs"]):  # the engine's (string) order
        problems.append(f"model seeds {report.get('model_seeds')} differ from "
                        f"{sorted(job['inputs'])}")
    if hashes.get("inputs") != {s: frozen["exports"].get(f) for s, f in job["inputs"].items()}:
        problems.append("input hashes differ from the bundle")
    if hashes.get("source") != dict(sha256=digest(source), files=source):
        problems.append("beat_audit source hashes differ from the bundle")
    if not configuration or hashes.get("configuration") != digest(configuration):
        problems.append("configuration hash does not match the recorded configuration")
    if {k: configuration.get(k) for k in executed} != executed:
        problems.append("configuration differs from jobs.json")
    return problems


def cell_status(cell, job, found, assignment):
    """(status, reason) of one expected cell: ok only with its resolved day pairs, its population
    and every seed/pair/direction component of jobs.json once in every fold."""
    from beat_audit.protocol import canonical_pairs
    if found is None:
        return "failed", "cell missing from the stream's report"
    spec = job["conditions"][cell["condition"]]
    pairs = canonical_pairs([(physical(p["a"], job), physical(p["b"], job)) for p in spec["pairs"]])
    if found.get("resolved_pairs") != pairs:
        return "failed", "resolved day pairs differ from jobs.json"
    analysed = found.get("analysed_users") or []
    if sorted(analysed + (found.get("excluded_users") or []), key=int) != sorted(
            map(str, job["users"]), key=int):
        return "failed", "population differs from jobs.json"
    if found.get("status") == "not_evaluable":
        if found.get("reason"):
            return "not_evaluable", found["reason"]
        return "failed", "not evaluable without a reason"
    if found.get("status") != "ok":
        return "failed", found.get("reason") or f"status {found.get('status')!r}"
    value = (found.get("comparators") or {}).get(cell["comparator"])
    if value is None:
        return "failed", "comparator missing from the cell"
    if not (finite(value.get("auc")) and finite(value.get("p")) and 0 <= value["p"] <= 1):
        return "failed", "no finite AUC and permutation p"
    index = {(tuple(p["a"]), tuple(p["b"])): i for i, p in enumerate(spec["pairs"])}
    expected = Counter((c["model_seed"], index.get((tuple(c["side_a"]), tuple(c["side_b"]))),
                        c["direction"]) for c in cell["components"])
    folds = {assignment.get(u) for u in analysed}
    seen = Counter((op.get("fold"), op.get("model_seed"), op.get("pair"), op.get("direction"))
                   for op in value.get("operating_points") or [])
    if (not folds or None in folds or max(expected.values()) > 1 or max(seen.values(), default=0)
            > 1 or set(seen) != {(f, *c) for f in range(len(folds)) for c in expected}):
        return "failed", "seed/pair/direction components differ from jobs.json"
    if (value.get("comparison_bounds") or {}).get("components") != len(expected):
        return "failed", "component count differs from jobs.json"
    return "ok", None


def reconcile(document, reports, frozen, errors=None):
    """Every expected cell of jobs.json against the batch's reports (plan §4 Attempts). A stream
    counts only if its report records the manifest's model seeds, cells and configuration and
    the bundle's source and input hashes; a cell's status is cell_status. Whatever is not
    established is failed. Streams that do not count are added to `errors` (rung: reason)."""
    errors = {} if errors is None else errors
    jobs = {j["rung"]: j for j in document["streams"]}
    listed = Counter((c["rung"], c["condition"], c["k"], c["comparator"])
                     for c in document["cells"])
    for rung, job in jobs.items():
        report = reports.get(rung)
        if rung in errors or report is None:
            errors.setdefault(rung, "no report")
            continue
        problems = stream_problems(job, document["settings"], report, frozen)
        reported = Counter((c.get("condition"), c.get("k")) for c in report.get("cells") or [])
        wanted = {(c["condition"], c["k"]) for c in document["cells"] if c["rung"] == rung}
        problems += [f"cell {c} reported {n} times" for c, n in reported.items() if n > 1]
        problems += [f"cell {c} not in jobs.json" for c in sorted(set(reported) - wanted, key=str)]
        if problems:
            errors[rung] = "report contract: " + "; ".join(problems)
    rows = []
    for cell in document["cells"]:
        found = find_cell(reports.get(cell["rung"]), cell["condition"], cell["k"])
        if listed[cell["rung"], cell["condition"], cell["k"], cell["comparator"]] > 1:
            status, reason = "failed", "cell listed more than once in jobs.json"
        elif cell["rung"] not in jobs:
            status, reason = "failed", "stream not in jobs.json"
        elif cell["rung"] in errors:
            status, reason = "failed", errors[cell["rung"]]
        else:
            status, reason = cell_status(cell, jobs[cell["rung"]], found,
                                         document["folds"]["assignment"])
        rows.append(dict(cell_key(cell), status=status, reason=reason, excluded_users=(
            found["excluded_users"] if found and status != "failed" else None)))
    return rows


def prediction_cells(cfg):
    """jobs.json keys (rung, condition, k, comparator) of V1, V2 and both cells of the
    pre-specified contrast (V1's comparator and k)."""
    preds = cfg["predictions"]
    e, v1 = preds["estimate"], preds["V1"]
    cells = {n: (preds[n]["rung"], preds[n]["condition"], preds[n]["k"], preds[n]["comparator"])
             for n in ("V1", "V2")}
    cells.update({f"estimate {side}": (e["rung"], e[side], v1["k"], v1["comparator"])
                  for side in ("same_day", "cross_day")})
    return cells


def unlisted_predictions(cfg, document):
    """Prediction and estimate cells that the job manifest does not list exactly once."""
    listed = Counter((c["rung"], c["condition"], c["k"], c["comparator"])
                     for c in document["cells"])
    return [f"{name} {key}" for name, key in prediction_cells(cfg).items() if listed[key] != 1]


def estimate_status(cfg, reports, statuses):
    """The pre-specified paired contrast: ok with its value, not_evaluable with a reason (also
    when one of its cells is not evaluable: the tool pairs ok cells only), else failed. An ok
    value needs finite, ordered interval endpoints and a mean equal to the same-day minus
    cross-day AUC of its two cells, as the tool computes it."""
    preds = cfg["predictions"]
    e, k, name = preds["estimate"], preds["V1"]["k"], preds["V1"]["comparator"]
    out = dict(e, k=k, comparator=name, value=None,
               note="paired same-day minus cross-day contrast; an estimate, no verdict")
    rows = {(s["rung"], s["condition"], s["k"], s["comparator"]): s for s in statuses}
    keys = prediction_cells(cfg)
    cells = [rows.get(keys[f"estimate {side}"]) for side in ("same_day", "cross_day")]
    if None in cells:
        return dict(out, status="failed", reason="contrast cells not in jobs.json")
    reasons = "; ".join(f"{c['condition']}: {c['reason']}" for c in cells if c["status"] != "ok")
    if any(c["status"] == "failed" for c in cells):
        return dict(out, status="failed", reason=reasons)
    if reasons:
        return dict(out, status="not_evaluable", reason=reasons)
    found = [c for c in (reports.get(e["rung"]) or {}).get("contrasts") or []
             if (c.get("same_day"), c.get("cross_day"), c.get("k")) == (
                 e["same_day"], e["cross_day"], k)]
    if len(found) != 1:
        return dict(out, status="failed", reason=f"{len(found)} contrast rows in the report, "
                                                 "expected 1")
    row = found[0]
    if row.get("status") == "not_evaluable" and row.get("reason"):
        return dict(out, status="not_evaluable", reason=row["reason"])
    value = (row.get("comparators") or {}).get(name) if row.get("status") == "ok" else None
    if value is None:
        return dict(out, status="failed", reason="contrast value missing from the report")
    interval = value.get("interval")
    if not (isinstance(interval, list) and len(interval) == 2 and all(map(finite, interval))
            and interval[0] <= interval[1]):
        return dict(out, status="failed", reason=f"contrast interval {interval!r}: two finite, "
                                                 "ordered endpoints required")
    aucs = [((find_cell(reports.get(e["rung"]), e[side], k) or {}).get("comparators") or {})
            .get(name, {}).get("auc") for side in ("same_day", "cross_day")]
    if not (finite(value.get("mean")) and all(map(finite, aucs)) and math.isclose(
            value["mean"], aucs[0] - aucs[1], rel_tol=0, abs_tol=1e-12)):
        return dict(out, status="failed", reason=f"contrast mean {value.get('mean')!r} is not "
                                                 f"the same-day minus cross-day AUC {aucs}")
    return dict(out, status="ok", reason=None, value=value)


def decision_document(cfg, reports, statuses, attempt, problems=()):
    """V1 and V2 as a Holm family of two from the reconciled cells. A failed or missing
    prediction is a technical failure, never a non-detection; a batch with any failed cell,
    failed prediction, failed estimate or batch problem gives no verdict."""
    preds = cfg["predictions"]
    rows = {(s["rung"], s["condition"], s["k"], s["comparator"]): s for s in statuses}
    keys = prediction_cells(cfg)
    estimate = estimate_status(cfg, reports, statuses)
    failed = [s for s in statuses if s["status"] == "failed"]
    found = {}
    for name in ("V1", "V2"):
        spec = preds[name]
        row = rows.get(keys[name], dict(status="failed", reason="prediction cell not in jobs.json"))
        cell = (find_cell(reports.get(spec["rung"]), spec["condition"], spec["k"])
                if row["status"] == "ok" else None)
        value = cell["comparators"][spec["comparator"]] if cell else {}
        found[name] = dict(
            rung=spec["rung"], condition=spec["condition"], k=spec["k"],
            comparator=spec["comparator"], status=row["status"], reason=row.get("reason"),
            auc=value.get("auc"), interval=value.get("interval"), p=value.get("p"),
            draws=value.get("draws"), excluded_users=row.get("excluded_users"),
            population_hash=cell and cell.get("population_hash"))
    technical = bool(failed or problems or estimate["status"] == "failed"
                     or any(v["status"] == "failed" for v in found.values()))
    family = holm_decision({n: v["p"] for n, v in found.items()}, preds["alpha"])
    predictions = {}
    for name, value in found.items():
        entry = dict(value, **family[name])
        if value["status"] == "failed":
            entry.update(supported=False,
                         wording=f"technical failure: {value['reason']}; no verdict")
        elif technical:
            entry.update(supported=False, wording="no verdict: the batch is a technical failure "
                                                  "(plan §4 Attempts)")
        elif value["status"] == "not_evaluable":
            entry["wording"] = (f"not evaluable ({value['reason']}); counts as p = 1 in the Holm "
                                "family")
        else:
            entry["wording"] = ("linkage detected" if entry["supported"] else
                                "not detected: insufficient evidence, not absence")
        predictions[name] = entry
    return dict(
        attempt=attempt["attempt"], kind=attempt["kind"], tier=TIER["yang2025"],
        technical_failure=technical, failed_cells=len(failed), problems=list(problems),
        alpha=preds["alpha"],
        family="Holm over V1 and V2; a non-evaluable prediction counts as p = 1",
        predictions=predictions, estimate=estimate)


def close_batch(cfg, document, record, attempt, results, problems, **fields):
    """Always ends a frozen attempt: every expected cell's status (a stream without a completed
    result fails all its cells), decision.json and the end record."""
    out, problems, decision = Path(cfg["results_dir"]) / "yang2025", list(problems), None
    statuses = []
    try:
        reports, errors = {}, {}
        try:
            folder = Path(cfg["freeze_dir"])
            frozen = dict(exports=read_json(folder / "hashes.json")["exports"],
                          sources=read_json(folder / "sources.json")["files"])
            by_rung = {r["rung"]: r for r in results}
            for job in document["streams"]:
                result = by_rung.get(job["rung"])
                if result is None or result["status"] != "completed":
                    errors[job["rung"]] = (result or {}).get("error") or "no result: " + (
                        "; ".join(problems) or "the worker returned nothing")
                    continue
                reports[job["rung"]] = read_json(ROOT / job["out"] / "report.json")
            statuses = reconcile(document, reports, frozen, errors)
        except Exception as exc:
            problems.append(f"reconciliation: {type(exc).__name__}: {exc}")
            reports, statuses = {}, [dict(cell_key(c), status="failed", reason=problems[-1],
                                          excluded_users=None) for c in document["cells"]]
        try:  # only the reports this batch read and whose stream passed the report contract
            rows = [r for job in document["streams"]
                    if job["rung"] in reports and job["rung"] not in errors
                    for r in summary_rows(cfg, job, reports[job["rung"]])]
            write_csv(pd.DataFrame(rows), out / "summary.csv")
        except Exception as exc:
            problems.append(f"summary: {type(exc).__name__}: {exc}")
        decision = decision_document(cfg, reports, statuses, attempt, problems)
        write_json(out / "cells.json", dict(attempt=record["attempt"], kind=record["kind"],
                                            cells=statuses))
        write_json(out / "decision.json", decision)
    except Exception as exc:
        problems.append(f"closing: {type(exc).__name__}: {exc}")
    finally:
        technical = decision is None or decision["technical_failure"] or bool(problems)
        end_attempt(cfg, record, status="technical_failure" if technical else "completed",
                    results=results, problems=problems,
                    cell_status=dict(Counter(s["status"] for s in statuses)),
                    outputs=[rel(out / n) for n in ("decision.json", "cells.json", "summary.csv")],
                    **fields)
    return decision


def frozen_attempts(attempts):
    """Start and end record of every frozen attempt, by attempt id."""
    events = defaultdict(dict)
    for a in attempts:
        if a["kind"] in FROZEN_KINDS:
            events[a["attempt"]][a["event"]] = a
    return dict(events)


def batch_kind(attempts, bundle):
    """frozen_batch first; technical_rerun only when every earlier attempt used this bundle and
    ended in a recorded technical failure (plan §4 Attempts)."""
    events = frozen_attempts(attempts)
    for attempt, e in events.items():
        start, end = e.get("start"), e.get("end")
        if start is None or start.get("bundle_sha256") != bundle["sha256"]:
            raise SystemExit(f"audit_cohorts: refused: frozen attempt {attempt} used another "
                             "bundle or has no start record; nothing was run")
        if end is None:
            raise SystemExit(f"audit_cohorts: refused: frozen attempt {attempt} has no end record "
                             "(interrupted); close it as a technical failure with `recover-batch "
                             f"--attempt {attempt}` before a repeat; nothing was run")
        if end.get("status") != "technical_failure":
            raise SystemExit(f"audit_cohorts: refused: frozen attempt {attempt} ended "
                             f"{end.get('status')!r}; a repeat follows only a recorded technical "
                             "failure (the first completed batch is the pre-specified result); "
                             "nothing was run")
    return "technical_rerun" if events else "frozen_batch"


def frozen_batch(cfg, args):
    """The one pre-specified Yang2025 batch: under the batch lock, refuses unless every hash
    matches the bundle and the attempts log allows it; once started, always closes its attempt."""
    with batch_lock(cfg):
        bundle = read_bundle(cfg)
        bad = verify_bundle(cfg, bundle)
        if bad:
            raise SystemExit(f"audit_cohorts: refused: {len(bad)} mismatches against the freeze "
                             f"bundle (first: {bad[:5]}); nothing was run")
        kind = batch_kind(read_attempts(cfg), bundle)
        document = read_json(Path(cfg["freeze_dir"]) / "jobs.json")
        record = start_attempt(cfg, kind, "frozen-batch", cohorts=["yang2025"],
                               bundle_sha256=bundle["sha256"],
                               streams=[j["rung"] for j in document["streams"]],
                               cells=document["cell_count"])
        attempt = dict(attempt=record["attempt"], kind=kind, bundle_sha256=bundle["sha256"])
        results, problems = [], []
        try:
            results = run_pool(audit_job, [(cfg, job, attempt) for job in document["streams"]],
                               document["max_parallel_jobs"], failed=stream_failure)
        except BaseException as exc:  # closed below, then raised
            problems.append(f"batch: {type(exc).__name__}: {exc}")
            raise
        finally:
            decision = close_batch(cfg, document, record, attempt, results, problems)
    if decision is None:
        return 1
    print(json.dumps({n: {k: v[k] for k in ("status", "auc", "p", "holm_p", "supported")}
                      for n, v in decision["predictions"].items()}))
    return 1 if decision["technical_failure"] else 0


def recover_batch(cfg, args):
    """Close a frozen attempt that was interrupted without an end record as a technical failure:
    its cells failed, decision.json without a verdict and a logged end record."""
    with batch_lock(cfg):  # a running batch holds it, so nothing running can be closed
        events = frozen_attempts(read_attempts(cfg))
        open_attempts = sorted(a for a, e in events.items() if "end" not in e)
        if args.attempt not in open_attempts:
            raise SystemExit(f"audit_cohorts: refused: {args.attempt!r} is not an open frozen "
                             f"attempt (open: {open_attempts}); nothing was changed")
        start = events[args.attempt]["start"]
        attempt = {k: start[k] for k in ("attempt", "kind", "bundle_sha256")}
        close_batch(cfg, read_json(Path(cfg["freeze_dir"]) / "jobs.json"), start, attempt, [],
                    ["interrupted: the attempt has no end record"],
                    recovered=dict(command="recover-batch", utc=utc(),
                                   note="closed while no frozen batch held the batch lock"))
    print(f"frozen attempt {args.attempt} closed as a technical failure")
    return 0


# ------------------------------------------------------------------ entry point

COMMANDS = {"export": export, "parity": parity, "pilot": pilot, "runtime-check": check_runtime,
            "develop": develop,
            "summary": summary, "select": select, "jobs": jobs, "freeze": freeze,
            "frozen-batch": frozen_batch, "recover-batch": recover_batch}


def read_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    confine(cfg)
    cfg["_path"] = path
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", nargs="?", default="export", choices=list(COMMANDS))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/audit.yaml")
    parser.add_argument("--cohort", nargs="+", help="develop: 2a and/or Lee2019 only")
    parser.add_argument("--rung", nargs="+", help="develop: report directories, e.g. features_P0")
    parser.add_argument("--replace", action="store_true",
                        help="freeze: replace a bundle that no frozen batch has used")
    parser.add_argument("--dry-run", action="store_true",
                        help="freeze: run every check of the freeze and write nothing")
    parser.add_argument("--attempt", help="recover-batch: the interrupted frozen attempt's id")
    args = parser.parse_args(argv)
    return COMMANDS[args.command](read_config(args.config), args)


if __name__ == "__main__":
    sys.exit(main())
