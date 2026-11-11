"""Plan v3 stage S4 (exploratory): per-user decoders and the transmission ladder.

docs/plan_v3.md §4 "S4 protocol" and §5 A10. For one cohort, every transmission alternative (rung)
is put on one common trial support, audited by beat_audit with the frozen settings on days that no
encoder or decoder was fitted on, and merged with its task utility through the tool's own
`beat-audit compare` (its refusals and its data-processing guard apply). Everything is exploratory.

  python scripts/audit_ladder.py prepare --cohort yang2025 [--set calibration_day]
  python scripts/audit_ladder.py run --cohort yang2025 [--set calibration_day] [--rung NAME ...]
  python scripts/audit_ladder.py table

Reads the frozen S2/S3 files (configs/audit.yaml, results/audit/export_manifest.json and
utility.csv, the exports and the study outputs) and never writes them. Writes only
artifacts/audit/ladder/** and results/audit/ladder*, and appends `exploratory_s4` records to
results/audit/attempts.jsonl.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
import warnings
import zipfile

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import yaml

ROOT = Path(__file__).resolve().parents[1]
# The project's beat_audit, not an installed copy (the venv holds an older non-editable install).
sys.path.insert(0, str(ROOT))
CONFIG = "configs/audit.yaml"                  # frozen at freeze-5: read only
MANIFEST = "results/audit/export_manifest.json"
EXPORT_UTILITY = "results/audit/utility.csv"
INPUTS = "artifacts/audit/ladder"
RESULTS = "results/audit/ladder"
TABLE = "results/audit/ladder.csv"
ATTEMPTS = "results/audit/attempts.jsonl"
KIND = "exploratory_s4"
LABEL = "exploratory"
KEY = ["user", "source_session", "source_run", "source_trial"]
LAYOUT = KEY + ["day", "segment", "target", "order"]
UNAVAILABLE = "not evaluated: source rung unavailable"
MODEL_SEEDS = (0, 1, 2)                        # the frozen job manifest's model seeds
LADDER_K = 20
THREADS = 4
FEATURE_COMPARATORS = ["cosine", "lda_cosine", "cosine_rich", "lda_cosine_rich"]
PER_USER = dict(C=1.0, max_iter=1000, taus=(0.5, 0.6, 0.8), folds=5, cv_seed=20261115)
UTILITY_BOOTSTRAP = dict(replicates=2000, seed=20261116, percentiles=(2.5, 97.5))
METRICS = ("task_ba", "hold_rate", "correct_move_rate", "wrong_move_rate")
SETS = ("principal", "calibration_day")
UTILITY_BASIS = (
    "later-day trials of the common support, per user (mean over model seeds 0-2), then mean over "
    "users with a 95% percentile bootstrap over users; task_ba = mean of the two class recalls of "
    "the argmax decision of the rung's decoder; rates over all trials (hold, move toward the cue, "
    "move away from it); a feature rung's receiver acts on the argmax of every trial (no hold)")

# Sides: "d" = every segment of day d, "d/s" = segment s of day d (export labels).
COHORTS = {
    "yang2025": dict(calibration_day=1, later_days=(2, 3), sets=dict(
        principal=dict(days=(2, 3), conditions=[
            ("same-day-d2-d3", "same-day", [(["2/1"], ["2/2"]), (["3/1"], ["3/2"])],
             "trials 1-100 against 101-200 of day 2 and of day 3"),
            ("cross-day-d2-d3", "cross-day", [(["2"], ["3"])], "day 2 against day 3")]),
        # day 1 is the P2-RC encoder's training day: that rung was never exported for it
        calibration_day=dict(days=(1, 2, 3), without=["features_P2-RC_d8"], conditions=[
            ("cross-day-d1-d23", "cross-day", [(["1"], ["2", "3"])],
             "day 1 (calibration) against days 2-3")]))),
    # Lee2019 and 2a have one later session; the tool needs two days in an input, so session 1
    # rides along on the common support, but no principal condition selects it (comparators,
    # eligibility and population masks use only the condition's segments).
    "lee2019": dict(calibration_day=1, later_days=(2,), sets=dict(
        principal=dict(days=(1, 2), conditions=[
            ("same-day-session2", "same-day", [(["2/1"], ["2/2"])],
             "offline against online phase of session 2")]),
        calibration_day=dict(days=(1, 2), conditions=[
            ("cross-day-session1-session2", "cross-day", [(["1"], ["2"])],
             "session 1 (calibration) against session 2")]))),
    "bnci2014_001": dict(calibration_day=1, later_days=(2,), sets=dict(
        principal=dict(days=(1, 2), conditions=[
            ("same-day-sessionE", "same-day", [(["2/1"], ["2/2"])],
             "runs 1-3 against runs 4-6 of session E")]),
        calibration_day=dict(days=(1, 2), conditions=[
            ("cross-day-sessionT-sessionE", "cross-day", [(["1"], ["2"])],
             "session T (calibration) against session E")]))),
}


# ------------------------------------------------------------------ small helpers

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str) + "\n", encoding="utf-8")


def write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def save_npz(path, arrays):
    """Compressed NPZ with fixed member timestamps: equal arrays give equal bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in arrays.items():
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with archive.open(info, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(value), allow_pickle=False)


def frozen_settings():
    """The frozen audit settings after the runtime gate's reductions (configs/audit.yaml)."""
    cfg = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    settings = dict(cfg["audit"]["settings"])
    for reduction in cfg["audit"]["reductions"]:
        if reduction != "drop_pair_logistic":
            raise SystemExit(f"audit_ladder: unexpected frozen reduction {reduction}")
        settings["comparators"] = [c for c in settings["comparators"] if c != "pair_logistic"]
    if settings["comparators"] != FEATURE_COMPARATORS:
        raise SystemExit(f"audit_ladder: frozen comparators {settings['comparators']} differ "
                         f"from {FEATURE_COMPARATORS}")
    return cfg, settings


def seeds_text(settings):
    return (f"model seeds {','.join(map(str, MODEL_SEEDS))}; fold seed {settings['fold_seed']}; "
            f"attack seed {settings['seed']} + 1009 x fold + 101 x model-seed position; matching "
            f"permutations {settings['seed']}; bootstrap {settings['seed'] + 1}; per-user "
            f"decoder CV seed {PER_USER['cv_seed']}; utility bootstrap {UTILITY_BOOTSTRAP['seed']}")


# ------------------------------------------------------------------ rungs and exports

def rungs(cohort, manifest, set_name="principal"):
    taus = sorted({e["tau"] for e in manifest["exports"]
                   if e["cohort"] == cohort and e["kind"] == "commands"})
    out = [dict(name="features_P0", kind="features", tau=None, derived_from=None,
                transmits="full features (P0)", decoder="shared P0 decoder (study final model)"),
           dict(name="features_P2-RC_d8", kind="features", tau=None, derived_from=None,
                transmits="recentred 8-d bottleneck (P2-RC d = 8)",
                decoder="shared P2-RC decoder (study final model)")]
    out += [dict(name=f"commands_shared_tau{t:.2f}", kind="commands", tau=t,
                 derived_from="features_P2-RC_d8", transmits=f"shared-decoder commands, tau {t:.2f}",
                 decoder="shared P2-RC decoder (study final model)") for t in taus]
    out += [dict(name=f"commands_per_user_tau{t:.2f}", kind="commands", tau=t,
                 derived_from="features_P0", transmits=f"per-user-decoder commands, tau {t:.2f}",
                 decoder="per-user standardised logistic regression (C = 1), calibration-day P0")
            for t in PER_USER["taus"]]
    without = COHORTS[cohort]["sets"][set_name].get("without", [])
    out = [r for r in out if r["name"] not in without]
    names = {r["name"] for r in out}
    for r in out:
        if r["derived_from"] not in names:
            r["derived_from"] = None
    return out


def export_entry(manifest, cohort, rung, kind, seed, tau=None):
    found = [e for e in manifest["exports"] if e["cohort"] == cohort and e["rung"] == rung
             and e["kind"] == kind and e["model_seed"] == seed
             and (tau is None or np.isclose(e["tau"], tau))]
    if len(found) != 1:
        raise ValueError(f"{cohort} {kind} {rung} seed {seed} tau {tau}: {len(found)} exports")
    return found[0]


def load_export(path, keys_path):
    """An export with its trial-key sidecar as a key frame (one row per export row)."""
    arrays = load_npz(path)
    keys = pd.read_csv(keys_path, dtype={"source_session": str})
    n = len(arrays["user"])
    if (len(keys) != n or not np.array_equal(keys.row.to_numpy(), np.arange(n))
            or any(not np.array_equal(keys[c].to_numpy(), arrays[c]) for c in ("user", "day", "order"))):
        raise ValueError(f"{keys_path}: the trial keys do not describe the export row by row")
    frame = keys[KEY].assign(day=arrays["day"], segment=arrays["segment"], target=arrays["target"],
                             order=arrays["order"], row=np.arange(n))
    if frame.duplicated(KEY).any():
        raise ValueError(f"{keys_path}: duplicate trial keys")
    return frame, arrays


def verified_export(entry):
    for name, sha in ((entry["file"], entry["sha256"]),
                      (entry["trial_keys"]["file"], entry["trial_keys"]["sha256"])):
        if sha256_file(ROOT / name) != sha:
            raise SystemExit(f"audit_ladder: {name} differs from the export manifest")
    return load_export(ROOT / entry["file"], ROOT / entry["trial_keys"]["file"])


def common_support(frames, days):
    """Trials present in every export on `days`, joined on (user, source session, run, trial),
    never on (user, day, order); day, segment and cue must agree. Rows follow the first export's
    time order, which every export must share; `order` is renumbered within user and day."""
    names = list(frames)
    columns = KEY + ["day", "segment", "target", "order"]
    first = frames[names[0]]
    support = first.loc[first.day.isin(days), columns]
    for name in names[1:]:
        merged = support.merge(frames[name][columns], on=KEY, how="inner",
                               suffixes=("", "_other"), validate="one_to_one")
        for field in ("day", "segment", "target"):
            bad = (merged[field] != merged[f"{field}_other"]).to_numpy()
            if bad.any():
                raise ValueError(f"{name}: {field} differs from {names[0]} for trial "
                                 f"{tuple(merged.loc[bad, KEY].iloc[0])}")
        merged = merged.sort_values(["user", "day", "order"])
        if (merged.groupby(["user", "day"]).order_other.diff() <= 0).any():
            raise ValueError(f"{name}: time order differs from {names[0]}")
        support = merged[columns]
    support = support.sort_values(["user", "day", "order"]).reset_index(drop=True)
    support["order"] = support.groupby(["user", "day"]).cumcount()
    return support


def layout_sha(frame):
    """sha256 of the ordered source-trial keys and layout (user, day, segment, target, order)."""
    return hashlib.sha256(frame[LAYOUT].to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def aligned_rows(frame, support):
    """Rows of an export holding the support's trials, in support order."""
    rows = support[KEY].merge(frame[KEY + ["row"]], on=KEY, how="left", validate="one_to_one").row
    if rows.isna().any():
        raise ValueError("a support trial is missing from an export")
    return rows.to_numpy(np.int64)


def write_input(folder, name, seed, support, outputs, kind):
    """One ladder input on the support (shared user/day/segment/order/target layout)."""
    arrays = dict(outputs=outputs, user=support.user.to_numpy(np.int64),
                  day=support.day.to_numpy(np.int64), segment=support.segment.to_numpy(np.int64),
                  order=support.order.to_numpy(np.int64), target=support.target.to_numpy(np.int64))
    if kind == "commands":
        arrays["outputs"] = np.asarray(outputs, np.int64)
        arrays["C"] = np.int64(3)
    else:
        arrays["outputs"] = np.asarray(outputs, np.float64)
    path = Path(folder) / f"{name}_seed{seed}.npz"
    save_npz(path, arrays)
    keys = support[KEY].assign(row=np.arange(len(support)), day=arrays["day"], order=arrays["order"])
    write_csv(keys[["row", "user", "day", "order"] + KEY[1:]], path.with_name(path.stem + ".trials.csv"))
    return path


# ------------------------------------------------------------------ decoders and utility

def fit_user_decoder(X, y):
    """Standardised logistic regression (C = 1); scaler and model see only the given trials."""
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=PER_USER["C"], max_iter=PER_USER["max_iter"])
    return scaler, model.fit(scaler.transform(X), y)


def decoder_margins(decoder, X):
    scaler, model = decoder
    return model.decision_function(scaler.transform(X))


def per_user_margins(user, day, X, y, calibration_day):
    """Log-odds of the right-hand cue for every row. Later-day rows: one decoder per user fitted
    on all of that user's calibration-day rows. Calibration-day rows: cross-fitted in 5
    stratified folds of the user's own calibration day (fixed seed). No later-day row reaches a
    scaler or a model."""
    margins, fits = np.full(len(y), np.nan), []
    for u in np.unique(user):
        own = user == u
        cal = np.flatnonzero(own & (day == calibration_day))
        later = np.flatnonzero(own & (day > calibration_day))
        if not len(cal):
            fits.append(dict(user=int(u), calibration_trials=0, decoder="none: no calibration day"))
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            decoder = fit_user_decoder(X[cal], y[cal])
            if len(later):
                margins[later] = decoder_margins(decoder, X[later])
            splitter = StratifiedKFold(n_splits=PER_USER["folds"], shuffle=True,
                                       random_state=PER_USER["cv_seed"])
            for train, test in splitter.split(X[cal], y[cal]):
                margins[cal[test]] = decoder_margins(
                    fit_user_decoder(X[cal[train]], y[cal[train]]), X[cal[test]])
        fits.append(dict(user=int(u), calibration_trials=len(cal), later_trials=len(later),
                         iterations=int(decoder[1].n_iter_[0]),
                         convergence_warnings=sum(issubclass(w.category, ConvergenceWarning)
                                                  for w in caught)))
    return margins, fits


def commands(margins, tau):
    """Maximum-probability threshold as beat.exp_c.decide: move when max p >= tau, toward the
    right (code 2) when the margin is positive, else left (1); otherwise hold (0)."""
    move = expit(np.abs(margins)) >= tau
    return np.where(move, np.where(margins > 0, 2, 1), 0).astype(np.int64)


def user_rates(user, target, decision, code):
    """Per-user task BA of the argmax decision and command rates over all of the user's trials."""
    rows = []
    for u in np.unique(user):
        m = user == u
        t, c = target[m], code[m]
        rows.append(dict(
            user=int(u), trials=int(m.sum()),
            task_ba=balanced_accuracy_score(t, decision[m]), hold_rate=float((c == 0).mean()),
            correct_move_rate=float((((c == 1) & (t == 0)) | ((c == 2) & (t == 1))).mean()),
            wrong_move_rate=float((((c == 2) & (t == 0)) | ((c == 1) & (t == 1))).mean())))
    return pd.DataFrame(rows).set_index("user")


def utility_summary(per_seed):
    """Mean over model seeds per user, then mean over users with a bootstrap over users."""
    per_user = sum(per_seed) / len(per_seed)
    values = per_user[list(METRICS)].to_numpy(float)
    draws = np.random.default_rng(UTILITY_BOOTSTRAP["seed"]).integers(
        0, len(values), size=(UTILITY_BOOTSTRAP["replicates"], len(values)))
    boot = values[draws].mean(axis=1)
    low, high = np.percentile(boot, UTILITY_BOOTSTRAP["percentiles"], axis=0)
    out = dict(users=len(per_user), trials=int(per_user.trials.sum()))
    for i, metric in enumerate(METRICS):
        out.update({metric: float(values[:, i].mean()), f"{metric}_low": float(low[i]),
                    f"{metric}_high": float(high[i])})
    out["per_user"] = per_user.reset_index().to_dict(orient="records")
    return out


def p0_decisions(cfg, cohort, seed, frame, arrays):
    """Argmax (1 = right) of the study's shared P0 decoder for every row of a P0 export."""
    ccfg = cfg["cohorts"][cohort]
    import joblib
    if cohort == "yang2025":
        # The sealed FrozenModel (beat.models): logits = classifier(scaler(tangent vector)); the
        # export holds the unstandardised tangent vectors (beat.models.FrozenModel.transform).
        model = joblib.load(ROOT / ccfg["models"].format(key="P0", seed=seed))
        x = model.features.scale.transform(arrays["outputs"]).astype(np.float32)
        return (model.predictor.decision_function(x) > 0).astype(np.int64)
    path = ROOT / ccfg["outputs"].format(key=ccfg["features"]["P0"]["key"], seed=seed)
    if cohort == "lee2019":
        packet = joblib.load(path)
        meta = pd.concat([packet["T"], packet["E"]], ignore_index=True)
        logits = np.concatenate([packet["tl"], packet["el"]])
    else:
        data = load_npz(path)
        meta = pd.DataFrame({c: data[c] for c in ("subject", "session", "run", "trial")})
        logits = data["logits"]
    stored = pd.DataFrame(dict(
        user=meta.subject.to_numpy(np.int64), source_session=meta.session.astype(str).to_numpy(),
        source_run=meta.run.to_numpy(np.int64), source_trial=meta.trial.to_numpy(np.int64),
        decision=logits.argmax(axis=1)))
    joined = frame[KEY].merge(stored, on=KEY, how="left", validate="one_to_one")
    if joined.decision.isna().any():
        raise ValueError(f"{cohort} P0 seed {seed}: export trials without stored logits")
    return joined.decision.to_numpy(np.int64)


def shared_decisions(frame, codes, support):
    """Argmax of the shared P2-RC decoder from its tau = 0.5 commands (no hold at 0.5)."""
    c = codes[aligned_rows(frame, support)]
    if (c == 0).any():
        raise ValueError("tau 0.5 shared commands hold on the support: not an argmax decision")
    return c - 1


# ------------------------------------------------------------------ prepare

def required_segments(pairs, schema):
    need = set()
    for a, b in pairs:
        for selector in list(a) + list(b):
            day, _, segment = selector.partition("/")
            need |= {(d, s) for d, s in schema if d == day and (not segment or s == segment)}
    return sorted(need)


def eligibility(layouts, conditions, k_grid):
    """Users with a k-window in every required segment of every input, per condition and k;
    the common population is their intersection over rungs (equal by construction here)."""
    out = {}
    for name, _, pairs, _ in conditions:
        out[name] = {}
        for k in k_grid:
            sets = []
            for arrays in layouts:
                labels = pd.DataFrame({c: np.asarray(arrays[c]).astype(str)
                                       for c in ("user", "day", "segment")})
                counts = labels.groupby(["user", "day", "segment"]).size()
                schema = sorted({(d, s) for _, d, s in counts.index})
                need = required_segments(pairs, schema)
                sets.append(sorted(u for u in labels.user.unique() if all(
                    counts.get((u, d, s), 0) >= k for d, s in need)))
            common = sorted(set.intersection(*map(set, sets)))
            if any(s != common for s in sets):
                raise ValueError(f"{name} k={k}: rungs differ in eligibility; check the support")
            out[name][str(k)] = common
    return out


def prepare(args):
    cohort, set_name = args.cohort, args.set
    spec, cset = COHORTS[cohort], COHORTS[cohort]["sets"][set_name]
    cfg, settings = frozen_settings()
    manifest = read_json(ROOT / MANIFEST)
    ladder = rungs(cohort, manifest, set_name)
    later, cal_day = list(spec["later_days"]), spec["calibration_day"]
    shared_taus = sorted({r["tau"] for r in ladder if r["name"].startswith("commands_shared")})
    sources, decisions, margins_cache, fits, checks = {}, {}, {}, {}, []
    export_utility = pd.read_csv(ROOT / EXPORT_UTILITY)
    for seed in MODEL_SEEDS:
        p0 = export_entry(manifest, cohort, "P0", "features", seed)
        frame, arrays = verified_export(p0)
        sources[("features_P0", seed)] = (frame, arrays["outputs"])
        decisions[("P0", seed)] = (frame, p0_decisions(cfg, cohort, seed, frame, arrays))
        tau05 = verified_export(export_entry(manifest, cohort, "P2-RC_d8", "commands", seed, 0.5))
        decisions[("P2-RC_d8", seed)] = tau05
        rc_frame, rc = verified_export(export_entry(manifest, cohort, "P2-RC_d8", "features", seed))
        if "features_P2-RC_d8" in {r["name"] for r in ladder}:
            sources[("features_P2-RC_d8", seed)] = (rc_frame, rc["outputs"])
        # both shared decoders reproduce the export phase's task BA on the exports' own rows
        p0_own, rc_own = frame[frame.day.isin(later)], rc_frame[rc_frame.day.isin(later)]
        for rung, own, dec in (
                ("P0", p0_own, decisions[("P0", seed)][1][p0_own.index.to_numpy()]),
                ("P2-RC_d8", rc_own, shared_decisions(tau05[0], tau05[1]["outputs"], rc_own))):
            ba = user_rates(own.user.to_numpy(), own.target.to_numpy(), dec, dec + 1).task_ba.mean()
            stored = export_utility[(export_utility.cohort == cohort) & (export_utility.rung == rung)
                                    & (export_utility.kind == "features")
                                    & (export_utility.model_seed == seed)].task_ba.iloc[0]
            checks.append(dict(check=f"{rung} decoder reproduces utility.csv task_ba", seed=seed,
                               recomputed=float(ba), stored=float(stored),
                               passed=bool(abs(ba - stored) <= 1e-12)))
            if not checks[-1]["passed"]:
                raise SystemExit(f"audit_ladder: {checks[-1]}")
        for tau in shared_taus:
            cframe, carr = tau05 if np.isclose(tau, 0.5) else verified_export(
                export_entry(manifest, cohort, "P2-RC_d8", "commands", seed, tau))
            sources[(f"commands_shared_tau{tau:.2f}", seed)] = (cframe, carr["outputs"])
        if p0["sha256"] not in margins_cache:
            margins_cache[p0["sha256"]] = per_user_margins(
                frame.user.to_numpy(), frame.day.to_numpy(), arrays["outputs"],
                frame.target.to_numpy(), cal_day)
        margins, fits[seed] = margins_cache[p0["sha256"]]
        finite = np.isfinite(margins)
        derived = frame[finite].assign(row=np.arange(finite.sum()))
        decisions[("per_user", seed)] = (derived, margins[finite])
        for tau in PER_USER["taus"]:
            sources[(f"commands_per_user_tau{tau:.2f}", seed)] = (
                derived, commands(margins[finite], tau))
    support = common_support({f"{n} seed {s}": f for (n, s), (f, _) in sources.items()},
                             list(cset["days"]))
    folder = ROOT / INPUTS / cohort / set_name
    inputs, layouts = {}, []
    for r in ladder:
        inputs[r["name"]] = {}
        for seed in MODEL_SEEDS:
            frame, outputs = sources[(r["name"], seed)]
            path = write_input(folder, r["name"], seed, support,
                               outputs[aligned_rows(frame, support)], r["kind"])
            layouts.append(load_npz(path))
            inputs[r["name"]][str(seed)] = dict(file=rel(path), sha256=sha256_file(path))
    population = eligibility(layouts, cset["conditions"], settings["k_grid"])
    on = support[support.day.isin(later)].reset_index(drop=True)
    u, t = on.user.to_numpy(), on.target.to_numpy()
    utility = {}
    for r in ladder:
        per_seed = []
        for seed in MODEL_SEEDS:
            if r["name"] == "features_P0":
                frame, d = decisions[("P0", seed)]
                dec = d[aligned_rows(frame, on)]
                code = dec + 1
            elif r["name"] == "features_P2-RC_d8" or r["name"].startswith("commands_shared"):
                frame, arr = decisions[("P2-RC_d8", seed)]
                dec = shared_decisions(frame, arr["outputs"], on)
                if r["kind"] == "features":
                    code = dec + 1
                else:
                    cframe, cout = sources[(r["name"], seed)]
                    code = cout[aligned_rows(cframe, on)]
            else:
                frame, m = decisions[("per_user", seed)]
                margin = m[aligned_rows(frame, on)]
                dec = (margin > 0).astype(np.int64)
                code = commands(margin, r["tau"])
            per_seed.append(user_rates(u, t, dec, code))
        utility[r["name"]] = utility_summary(per_seed)
    out = ROOT / RESULTS / cohort / set_name
    summary = dict(rows=len(support), users=int(support.user.nunique()),
                   rows_per_day={str(d): int(n) for d, n in support.groupby("day").size().items()},
                   later_day_rows=len(on), sha256_of_keys=layout_sha(support))
    plan = dict(
        label=LABEL, cohort=cohort, set=set_name, days=list(cset["days"]), later_days=later,
        calibration_day=cal_day, conditions=[dict(name=n, relation=r, pairs=p, sides=d)
                                             for n, r, p, d in cset["conditions"]],
        settings=settings, seeds=seeds_text(settings), model_seeds=list(MODEL_SEEDS),
        seconds_per_output=cfg["cohorts"][cohort]["seconds_per_output"]["value"],
        users=sorted(str(x) for x in support.user.unique()), support=summary, rungs=ladder,
        inputs=inputs, per_user_decoder=dict(PER_USER, note=(
            "the per-user decoder is deterministic and P0 features are equal across model seeds, "
            "so its three model-seed inputs hold equal commands; they keep the population hash "
            "(masks keyed by model seed) equal to the other rungs'")),
        join="(user, source_session, source_run, source_trial); time order of the first export")
    write_json(out / "plan.json", plan)
    write_json(out / "population.json", dict(
        rule="users eligible in every rung at the row's condition and k (docs/plan_v3.md S4)",
        support=summary, eligible=population))
    write_json(out / "utility.json", dict(basis=UTILITY_BASIS, bootstrap=UTILITY_BOOTSTRAP,
                                          checks=checks, rungs=utility))
    write_json(out / "decoders.json", dict(PER_USER, fits={str(s): f for s, f in fits.items()}))
    print(f"{cohort} {set_name}: support {summary['rows']} rows, {summary['users']} users "
          f"{summary['rows_per_day']}; utility on {len(on)} later-day rows")
    for name, value in utility.items():
        print(f"  {name:28s} BA {value['task_ba']:.4f}  hold {value['hold_rate']:.4f}  "
              f"wrong {value['wrong_move_rate']:.4f}")
    return [rel(out / n) for n in ("plan.json", "population.json", "utility.json", "decoders.json")]


# ------------------------------------------------------------------ run

def condition_objects(conditions):
    from beat_audit.api import Condition, Segment

    def side(selectors):
        return tuple(Segment(*s.split("/")) if "/" in s else Segment(s) for s in selectors)

    return [Condition(c["name"], c["relation"], tuple((side(a), side(b)) for a, b in c["pairs"]))
            for c in conditions]


def audit_rung(plan, rung, out):
    """One rung through beat_audit with the frozen settings, on the plan's population."""
    from beat_audit.api import AuditConfig, Population, audit_stream
    s = plan["settings"]
    config = AuditConfig(
        k_grid=tuple(s["k_grid"]), fold_seed=s["fold_seed"], seed=s["seed"],
        permutations=s["permutations"], bootstraps=s["bootstraps"], threads=s["threads"],
        cohort=plan["cohort"], comparators=tuple(s["comparators"]), gallery=s["gallery"],
        sequential=s["sequential"], closed_set=s["closed_set"],
        seconds_per_output=plan["seconds_per_output"],
        day_order=tuple(str(d) for d in plan["days"]))
    inputs = {seed: ROOT / item["file"] for seed, item in plan["inputs"][rung["name"]].items()}
    for seed, item in plan["inputs"][rung["name"]].items():
        if sha256_file(inputs[seed]) != item["sha256"]:
            raise SystemExit(f"audit_ladder: {item['file']} changed since prepare")
    return audit_stream(inputs, conditions=condition_objects(plan["conditions"]), config=config,
                        population=Population(users=plan["users"]), kind=rung["kind"], out=out)


def run(args):
    out = ROOT / RESULTS / args.cohort / args.set
    plan = read_json(out / "plan.json")
    names = [r["name"] for r in plan["rungs"]]
    if args.rung and set(args.rung) - set(names):
        raise SystemExit(f"audit_ladder: unknown rungs {sorted(set(args.rung) - set(names))}")
    runs_path, written = out / "runs.json", []
    runs = read_json(runs_path) if runs_path.exists() else {}
    for rung in plan["rungs"]:
        if args.rung and rung["name"] not in args.rung:
            continue
        started = time.perf_counter()
        report = audit_rung(plan, rung, out / rung["name"])
        cells = pd.Series([c["status"] for c in report["cells"]]).value_counts().to_dict()
        runs[rung["name"]] = dict(
            seconds=time.perf_counter() - started, tool_runtime_seconds=report["runtime_seconds"],
            cells=cells, report=rel(out / rung["name"] / "report.json"),
            report_sha256=sha256_file(out / rung["name"] / "report.json"),
            utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        write_json(runs_path, runs)
        written.append(runs[rung["name"]]["report"])
        print(f"{args.cohort} {args.set} {rung['name']}: {runs[rung['name']]['seconds']:.0f} s, "
              f"cells {cells}", flush=True)
    return written + [rel(runs_path)]


# ------------------------------------------------------------------ tables

def check_populations(reports, eligible=None):
    """Every rung's cell at a condition and k has the same analysed users and population hash
    (and, when given, the precomputed common population); returns {(condition, k): hash}."""
    hashes = {}
    for name, report in reports.items():
        for cell in report["cells"]:
            key = (cell["condition"], cell["k"])
            users = sorted(cell["analysed_users"])
            if eligible is not None and users != sorted(eligible[cell["condition"]][str(cell["k"])]):
                raise ValueError(f"{name} {key}: analysed users differ from the common population")
            if key not in hashes:
                hashes[key] = (cell["population_hash"], users, name)
            elif hashes[key][:2] != (cell["population_hash"], users):
                raise ValueError(f"{name} {key}: population hash differs from {hashes[key][2]}")
    return {key: value[0] for key, value in hashes.items()}


def comparison_digest(plan, name, report):
    """sha256 of what comparing rungs relies on beyond the tool's population hash (which binds
    users, folds and row indices only): the ordered source-trial keys and layout of every
    model-seed input the report hashed, read with its trial-key sidecar; the population and fold
    assignment of every cell; the complete condition specification with days, day order and
    analysis settings. Returns the digest and each seed's layout hash."""
    layouts = {}
    for seed, item in plan["inputs"][name].items():
        path = ROOT / item["file"]
        if sha256_file(path) != report["hashes"]["inputs"][seed]:
            raise ValueError(f"{name} seed {seed}: {item['file']} is not the input its report hashed")
        layouts[seed] = layout_sha(load_export(path, path.with_name(path.stem + ".trials.csv"))[0])
    configuration = report["configuration"]
    cells = sorted(([c["condition"], c["relation"], c["k"], c["days"], c["resolved_pairs"],
                     c["population"], c["population_hash"]] for c in report["cells"]),
                   key=lambda c: (c[0], c[2]))
    value = dict(layouts=layouts, cells=cells, cohort=report["cohort"], days=report["days"],
                 model_seeds=report["model_seeds"], conditions=configuration["conditions"],
                 day_order=configuration["day_order"], settings=configuration["settings"])
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(), layouts


def check_comparison(plan, reports):
    """Refuses unless every rung has one comparison digest and the inputs' layout is the plan's
    common support (support.sha256_of_keys from prepare); returns {rung: digest}."""
    digests = {name: comparison_digest(plan, name, report) for name, report in reports.items()}
    first = next(iter(digests))
    for name, (value, _) in digests.items():
        if value != digests[first][0]:
            raise ValueError(f"{name}: comparison digest differs from {first}'s (trial keys, "
                             "layout, folds or conditions)")
    if set(digests[first][1].values()) != {plan["support"]["sha256_of_keys"]}:
        raise ValueError("the inputs' trial keys and layout are not the plan's common support")
    return {name: value for name, (value, _) in digests.items()}


def shown(path):
    """Project-relative when under the project (the README case runs from its root)."""
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def run_compare(reports, utility, derived, out_csv):
    """`beat-audit compare` (the tool's own CLI) on the ladder's reports; returns its rows."""
    out_csv = Path(out_csv)
    util_path = out_csv.with_name(out_csv.stem + "_utility.json")
    write_json(util_path, {name: {m: utility[name][m] for m in METRICS} for name in reports})
    command = ["python", "-m", "beat_audit", "compare",
               *[f"{name}={shown(path)}" for name, path in reports.items()],
               "--utility", shown(util_path), "--out", shown(out_csv),
               *[f"--derived-from={c}={f}" for c, f in derived.items()]]
    done = subprocess.run([sys.executable] + command[1:], cwd=ROOT, capture_output=True,
                          text=True)
    if done.returncode != 0:
        raise SystemExit(f"audit_ladder: beat-audit compare refused: {done.stderr.strip()}")
    return pd.read_csv(out_csv, keep_default_na=False), command


def guard_outcomes(rows, derived, conditions):
    """Per declared derivation and requested cell: command AUC, feature AUC and whether the guard
    fired (the tool's coverage rows for analyses not requested are left out)."""
    out = []
    for command, feature in derived.items():
        for _, crow in rows[(rows.alternative == command) & rows.condition.isin(conditions)].iterrows():
            frow = rows[(rows.alternative == feature) & (rows.condition == crow.condition)
                        & (rows.k == crow.k)].iloc[0]
            evaluable = crow.status == "ok" and frow.status == "ok"
            out.append(dict(command=command, feature=feature, condition=crow.condition, k=crow.k,
                            command_auc=crow.auc if evaluable else "",
                            feature_auc=frow.auc if evaluable else "",
                            flagged=bool(crow.data_processing_guard),
                            outcome=("not evaluable" if not evaluable else "flagged"
                                     if crow.data_processing_guard else "not flagged")))
    return pd.DataFrame(out)


def fmt(value, digits=3):
    if value in ("", None) or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{float(value):.{digits}f}"


def markdown(cohort, set_name, plan, rows, guards, command, out_csv, digest, unavailable):
    conditions = {c["name"]: c["sides"] for c in plan["conditions"]}
    lines = [f"# Transmission ladder, {cohort} ({set_name.replace('_', '-')} contrasts; "
             "exploratory)", "",
             f"Rendered from `{shown(out_csv)}`, the table written by `beat-audit compare`; the "
             "tool checked cohort, days, condition, k, physical sides and the population hash of "
             "every report before merging.", "",
             "- Contrasts: " + "; ".join(f"`{n}` = {d}" for n, d in conditions.items()),
             f"- Population: {len(plan['users'])} users on a common support of "
             f"{plan['support']['rows']} trials; every row of a condition and k shares one "
             "population hash.",
             f"- Comparison digest `{digest}`, equal for every rung of this table: it binds the "
             "ordered source-trial keys and layout (user, day, segment, order, target) of every "
             "model-seed input, equal to the plan's common support, the population and fold "
             "assignment of every cell, and the complete condition specification with the "
             "analysis settings; the script refuses a contrast set whose rungs differ.",
             "- Attacker: open-set linkage of transmitted outputs only; comparators fitted on the "
             "development users of each fold and scored on the fold's unseen target users (feature "
             "rungs: cosine, LDA, rich-summary cosine and LDA, the strongest named post hoc; command "
             "rungs: cue-aware Bayes factor).",
             f"- Utility: {UTILITY_BASIS}.", f"- Seeds: {plan['seeds']}."]
    gaps = sorted(set(rows.condition) - set(conditions))
    if gaps:
        lines.append(f"- The CSV also carries the tool's coverage rows {', '.join(gaps)} (not "
                     "evaluable: not requested on these days; see the contrasts above).")
    lines.append("")
    for condition in conditions:
        lines += [f"## {condition}, k = {LADDER_K}", "",
                  "| alternative | derived from | task BA | hold | wrong move | comparator | "
                  "mean AUC [95% interval] | wording | data-processing guard |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in rows[(rows.condition == condition) & (rows.k == LADDER_K)].iterrows():
            auc = (f"{fmt(r.auc)} [{fmt(r.interval_low)}, {fmt(r.interval_high)}]"
                   if r.status == "ok" else r.status)
            guard = (UNAVAILABLE if r.alternative in unavailable
                     else r.data_processing_guard or "not flagged")
            lines.append(f"| {r.alternative} | {r.derived_from} | {fmt(r.task_ba)} | "
                         f"{fmt(r.hold_rate)} | {fmt(r.wrong_move_rate)} | {r.comparator} | {auc} | "
                         f"{r.wording} | {guard} |")
        ks = sorted(rows.k.unique())
        lines += ["", f"Mean AUC by k ({condition}; blank = not evaluable):", "",
                  "| alternative | " + " | ".join(f"k = {k}" for k in ks) + " |",
                  "|---|" + "---|" * len(ks)]
        for alt in rows.alternative.unique():
            part = rows[(rows.alternative == alt) & (rows.condition == condition)].set_index("k")
            lines.append(f"| {alt} | " + " | ".join(
                fmt(part.auc[k]) if part.status[k] == "ok" else "" for k in ks) + " |")
        lines.append("")
    lines += ["## Data-processing guard, every declared derivation", ""]
    for (command_alt, feature), part in guards.groupby(["command", "feature"], sort=False):
        flagged = part[part.outcome == "flagged"]
        cells = ", ".join(f"{r.condition} k = {r.k}" for _, r in flagged.iterrows())
        lines.append(f"- `{command_alt}` declared derived from `{feature}`: "
                     + (f"flagged at {cells} (the feature attacker is too weak there; the feature "
                        "rung's exposure is at least the command rung's)" if len(flagged) else
                        f"not flagged at any of {int((part.outcome != 'not evaluable').sum())} "
                        "evaluable cells"))
    for command_alt, feature in unavailable.items():
        lines.append(f"- `{command_alt}` derives from `{feature}`, which is not a rung on these "
                     f"days, so the derivation cannot be declared: {UNAVAILABLE}")
    lines += ["", "Command line (from the project root):", "", "```", " ".join(command), "```", ""]
    return "\n".join(lines)


CONTRAST_FIELDS = ("condition", "sides", "status", "comparator", "comparator_selection", "auc",
                   "auc_low", "auc_high", "p", "holm_p", "wording", "guard")


def contrast_columns(prefix, cell, guard, note=""):
    if cell is None:
        return {f"{prefix}_{f}": note if f == "status" else "" for f in CONTRAST_FIELDS}
    ok = cell["status"] == "ok"
    best = cell["comparators"][cell["strongest"]] if ok else {}
    return {f"{prefix}_condition": cell["condition"], f"{prefix}_sides": cell["sides"],
            f"{prefix}_status": cell["status"],
            f"{prefix}_comparator": cell.get("strongest", ""),
            f"{prefix}_comparator_selection": (
                "" if not ok else "only command comparator" if cell["strongest"] == "bayes_factor"
                else "highest mean AUC of the four frozen feature comparators on these target "
                     "users (post hoc; all four in the report)"),
            f"{prefix}_auc": best.get("auc", ""),
            f"{prefix}_auc_low": best["interval"][0] if ok else "",
            f"{prefix}_auc_high": best["interval"][1] if ok else "",
            f"{prefix}_p": best.get("p", ""), f"{prefix}_holm_p": best.get("holm_p", ""),
            f"{prefix}_wording": cell.get("wording", cell.get("reason", "")),
            f"{prefix}_guard": guard}


def cohort_set(cohort, set_name):
    """Plan, utility, population and the available reports of one cohort's contrast set."""
    out = ROOT / RESULTS / cohort / set_name
    if not (out / "plan.json").exists():
        return None
    plan = read_json(out / "plan.json")
    reports = {r["name"]: out / r["name"] / "report.json" for r in plan["rungs"]
               if (out / r["name"] / "report.json").exists()}
    return dict(plan=plan, utility=read_json(out / "utility.json")["rungs"],
                eligible=read_json(out / "population.json")["eligible"], reports=reports)


def table(args):
    from beat_audit.engine import source_hashes
    written, rows, missing = [], [], []
    tool_sha = source_hashes()["sha256"]
    for cohort in COHORTS:
        sets = {s: cohort_set(cohort, s) for s in SETS}
        main = sets["principal"]
        if main is None or len(main["reports"]) != len(main["plan"]["rungs"]):
            missing.append(f"{cohort}: principal ladder not complete")
            continue
        by_set = {}
        for set_name, part in sets.items():
            if part is None or not part["reports"]:
                if set_name != "principal":
                    missing.append(f"{cohort}: {set_name} contrasts not run")
                continue
            plan = part["plan"]
            ran = [r for r in plan["rungs"] if r["name"] in part["reports"]]
            if len(ran) != len(plan["rungs"]):
                missing.append(f"{cohort} {set_name}: not run for "
                               f"{[r['name'] for r in plan['rungs'] if r not in ran]}")
            reports = {n: read_json(p) for n, p in part["reports"].items()}
            for name, report in reports.items():
                if report["hashes"]["source"]["sha256"] != tool_sha:
                    raise SystemExit(f"audit_ladder: {name} ran on another beat_audit source")
                if report["hashes"]["inputs"] != {s: i["sha256"] for s, i in plan["inputs"][name].items()}:
                    raise SystemExit(f"audit_ladder: {name} report inputs differ from the plan")
            hashes = check_populations(reports, part["eligible"])
            try:
                digests = check_comparison(plan, reports)
            except ValueError as exc:
                raise SystemExit(f"audit_ladder: refused {cohort} {set_name}: {exc}") from exc
            names = {r["name"] for r in ran}
            derived = {r["name"]: r["derived_from"] for r in ran
                       if r["derived_from"] and r["derived_from"] in names}
            # the principal plan declares every rung's source; a set without that rung (Yang2025's
            # calibration day has no 8-d export) cannot declare the derivation
            declared = {r["name"]: r["derived_from"] for r in main["plan"]["rungs"]}
            unavailable = {r["name"]: declared[r["name"]] for r in ran
                           if declared.get(r["name"]) and declared[r["name"]] not in names}
            suffix = "" if set_name == "principal" else f"_{set_name}"
            out_csv = ROOT / f"{RESULTS}_compare_{cohort}{suffix}.csv"
            compared, command = run_compare(part["reports"], part["utility"], derived, out_csv)
            guards = guard_outcomes(compared, derived, [c["name"] for c in plan["conditions"]])
            write_csv(guards, out_csv.with_name(out_csv.stem + "_guard.csv"))
            md = out_csv.with_suffix(".md")
            md.write_text(markdown(cohort, set_name, plan, compared, guards, command, out_csv,
                                   next(iter(digests.values())), unavailable), encoding="utf-8")
            written += [rel(p) for p in (out_csv, out_csv.with_name(out_csv.stem + "_utility.json"),
                                         out_csv.with_name(out_csv.stem + "_guard.csv"), md)]
            by_set[set_name] = dict(plan=plan, reports=reports, hashes=hashes, digests=digests,
                                    guards=guards, compared=compared)
        rows += ladder_rows(cohort, main["utility"], by_set)
    frame = pd.DataFrame(rows)
    write_csv(frame, ROOT / TABLE)
    written.append(TABLE)
    for note in missing:
        print(f"not in the table: {note}")
    print(f"{TABLE}: {len(frame)} rows")
    return written


def set_cell(part, rung, relation):
    """(cell at the ladder's k with its sides, guard outcome, note) of one rung and relation."""
    if part is None:
        return None, "", "contrast set not run"
    condition = next((c for c in part["plan"]["conditions"] if c["relation"] == relation), None)
    if condition is None:
        return None, "", f"no {relation} contrast in this set"
    if rung["name"] not in {r["name"] for r in part["plan"]["rungs"]}:
        return None, "", "rung not available on these days"
    if rung["name"] not in part["reports"]:
        return None, "", "not run"
    cell = next(c for c in part["reports"][rung["name"]]["cells"]
                if c["condition"] == condition["name"] and c["k"] == LADDER_K)
    guards = part["guards"]
    mine = guards[(guards.command == rung["name"]) & (guards.condition == condition["name"])
                  & (guards.k == LADDER_K)] if len(guards) else guards
    names = {r["name"] for r in part["plan"]["rungs"]}
    guard = (mine.outcome.iloc[0] if len(mine)
             else UNAVAILABLE if rung["derived_from"] and rung["derived_from"] not in names
             else "no derivation declared")
    return dict(cell, sides=condition["sides"]), guard, ""


def ladder_rows(cohort, utility, by_set):
    """One row per rung (and tau): utility, then each contrast at the ladder's k."""
    main, calibration = by_set["principal"], by_set.get("calibration_day")
    plan = main["plan"]
    hashes = {main["hashes"][(c["name"], LADDER_K)] for c in plan["conditions"]}
    if len(hashes) != 1:
        raise SystemExit(f"audit_ladder: {cohort} principal contrasts differ in population at "
                         f"k = {LADDER_K}")
    population_hash = hashes.pop()
    rows = []
    for rung in plan["rungs"]:
        u = utility[rung["name"]]
        row = dict(label=LABEL, cohort=cohort, rung=rung["name"], transmits=rung["transmits"],
                   decoder=rung["decoder"], kind=rung["kind"],
                   tau="" if rung["tau"] is None else f"{rung['tau']:.2f}",
                   derived_from=rung["derived_from"] or "",
                   days=",".join(map(str, plan["later_days"])),
                   calibration_day=plan["calibration_day"], k=LADDER_K,
                   model_seeds=",".join(map(str, MODEL_SEEDS)), seeds=plan["seeds"],
                   population_hash=population_hash,
                   comparison_digest=main["digests"][rung["name"]], users=len(plan["users"]),
                   utility_users=u["users"], utility_trials=u["trials"])
        for m in METRICS:
            row.update({m: u[m], f"{m}_low": u[f"{m}_low"], f"{m}_high": u[f"{m}_high"]})
        row["utility_basis"] = UTILITY_BASIS
        for relation in ("same-day", "cross-day"):
            cell, guard, note = set_cell(main, rung, relation)
            row.update(contrast_columns(relation.replace("-", "_"), cell, guard, note))
        cell, guard, note = set_cell(calibration, rung, "cross-day")
        row.update(contrast_columns("calibration_day_cross_day", cell, guard, note))
        row["calibration_day_population_hash"] = cell["population_hash"] if cell else ""
        row["calibration_day_comparison_digest"] = (
            calibration["digests"][rung["name"]] if cell else "")
        row["calibration_day_note"] = (
            "includes the calibration day: P0 and P2-RC features are in-sample there (tangent "
            "reference or encoder fitted on it); commands are cross-fitted (shared decoder: the "
            "study's cross-fitting; per-user decoder: 5 stratified folds)") if cell else note
        row["report"] = rel(ROOT / RESULTS / cohort / "principal" / rung["name"] / "report.json")
        rows.append(row)
    return rows


# ------------------------------------------------------------------ provenance

def log_attempt(path, record):
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def git_head():
    done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return done.stdout.strip()


COMMANDS = {"prepare": prepare, "run": run, "table": table}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--cohort", choices=sorted(COHORTS))
    parser.add_argument("--set", choices=SETS, default="principal")
    parser.add_argument("--rung", nargs="+")
    args = parser.parse_args(argv)
    if args.command != "table" and args.cohort is None:
        parser.error("--cohort is required")
    from beat_audit.engine import source_hashes
    if not Path(source_hashes.__code__.co_filename).resolve().is_relative_to(ROOT):
        raise SystemExit("audit_ladder: beat_audit must be imported from the project")
    attempt = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-"
               + uuid.uuid4().hex[:8])
    base = dict(attempt=attempt, kind=KIND, command=f"ladder-{args.command}", label=LABEL)
    log_attempt(ROOT / ATTEMPTS, dict(
        base, event="start", utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        cohorts=[args.cohort] if args.cohort else sorted(COHORTS), set=args.set,
        rungs=args.rung, script=rel(Path(__file__)), script_sha256=sha256_file(Path(__file__)),
        beat_audit_sha256=source_hashes()["sha256"], git_head=git_head(), config=CONFIG,
        config_sha256=sha256_file(ROOT / CONFIG)))
    started, status, outputs, error = time.perf_counter(), "failed", [], None
    try:
        with threadpool_limits(limits=THREADS):
            outputs = COMMANDS[args.command](args)
        status = "completed"
        return 0
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        log_attempt(ROOT / ATTEMPTS, dict(
            base, event="end", status=status, seconds=time.perf_counter() - started,
            outputs=outputs, error=error,
            utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))


if __name__ == "__main__":
    raise SystemExit(main())
