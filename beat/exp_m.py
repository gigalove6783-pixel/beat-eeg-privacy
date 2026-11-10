"""Sealed confirmatory replication M on Yang2025 (WBCIC-SHU two-class MI, 51 new users).

    python -m beat.exp_m dev                      # recording day 1 only (pre-seal)
    python -m beat.exp_m streams                  # day-1 command streams and triviality checks
    python -m beat.exp_m final --seal-commit HASH --result-dir results/exp_m \
        --artifact-dir artifacts/exp_m             # after seal 4: days 2-3

Day 1 is the calibration day ("T"); days 2 and 3 are deployment days ("E"), opened only by
``final`` after verify_seal. The pipeline mirrors Experiment L (beat/exp_l.py): causal
8-30 Hz reduced trials, leave-one-run-out cross-fitting on day 1, epoch budgets by
``choose_epochs``, NLL temperature on cross-fitted logits, Experiment C's command rule.
Command-stream statistics use Experiment K's Bayes attackers, groups, EQ and randomized
response (beat/exp_k.py, a seal-4 file); the decision rule is configs/exp_m.yaml's, which
supersedes configs/exp_k.yaml's declared decision rule for this study.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from joblib import delayed
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from .common import digest, holm, rng_for, t_greater, t_interval, write_csv, write_json
from .config import ROOT
from .data import LeakageError, Trials, require_training
from .exp_b import _task_rows, choose_epochs
from .exp_c import decide, fit_temperature
from .exp_k import (Counts, Stream, apply_channel, apply_crossfit, bayes_scores, chance,
                    confusion_counts, crossfit_matrices, fit_eq, make_groups, net_progress_from,
                    recalls, rr_matrix, task_information, window_starts, wrong_weight)
from .exp_l import (_fit_attack, _oof, _pilot, _prepare_fold, budget_key, finite_trials,
                    parallel, training_config, valid_covariances)
from .models import fit_defended
from .verify import check_seals
from .yang2025 import (KINDS, load_trials, member_path, post_seal_allowlist, read_checksums,
                       sha256_file)

RESULTS = ROOT / "results/exp_m"
ARTIFACTS = ROOT / "artifacts/exp_m"
CONFIG = ROOT / "configs/exp_m.yaml"
SEAL_FILES = ("docs/preregistration_m.md", "configs/exp_m.yaml", "configs/exp_k.yaml",
              "beat/exp_m.py", "beat/yang2025.py", "beat/exp_k.py", "beat/common.py",
              "scripts/fetch_yang2025.py", "results/exp_m/selection.json",
              "checksums/yang2025_day1.sha256")
# Inputs hashed into selection.json besides exp_m's import closure.
SELECTION_INPUTS = ("configs/exp_m.yaml", "configs/exp_k.yaml", "scripts/fetch_yang2025.py")
CHECKSUMS = {1: "checksums/yang2025_day1.sha256", 2: "checksums/yang2025_day23.sha256",
             3: "checksums/yang2025_day23.sha256"}
PACKAGES = ("numpy", "scipy", "sklearn", "pandas", "torch", "pyriemann", "joblib")


def load_cfg(path=None):
    return yaml.safe_load(Path(path or CONFIG).read_text(encoding="utf-8"))


def analysis_config(cfg):
    return {k: v for k, v in cfg.items()
            if k not in ("workers", "lead_status") and not k.startswith("_")}


def development_config(cfg):
    return {k: cfg[k] for k in ("data_dir", "subjects", "seeds", "conditions", "training",
                                "temperature")}


def versions():
    out = dict(python=platform.python_version())
    for name in PACKAGES:
        out[name] = __import__(name).__version__
    return out


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def day1_only():
    """Process audit guard: any open of a day-2/3 reduced or raw file raises."""
    state = {"active": True, "opened": [], "blocked": []}

    def audit(event, args):
        if not state["active"] or event != "open" or not isinstance(args[0], (str, Path)):
            return
        path = str(args[0]).replace("\\", "/").lower()
        name = path.rsplit("/", 1)[-1]
        if (name.startswith(("day2_", "day3_")) and name.endswith(".npz")) or \
                "_ses-02_" in name or "_ses-03_" in name:
            state["blocked"].append(path)
            raise LeakageError("Pre-seal stages may not open recording days 2-3")
        if name.startswith("day1_") and name.endswith(".npz"):
            state["opened"].append(path)

    sys.addaudithook(audit)
    try:
        yield state
    finally:
        state["active"] = False


# ------------------------------------------------------------------ decoders (day 1)

def settings(cfg):
    for key, value in cfg["conditions"].items():
        yield dict(key=key, **value)


def _timed(function, *args):
    start = time.perf_counter()
    out = function(*args)
    return out, time.perf_counter() - start


def _time_full_fit(train, setting, cfg, seed, budget):
    torch_threads(cfg)
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        fit_defended(train, setting["condition"], training_config(cfg, setting["d"]), seed,
                     budget, setting["target"])
        return time.perf_counter() - start


def torch_threads(cfg):
    import torch
    torch.set_num_threads(cfg["torch_threads"])


def oof_digest(oof):
    return hashlib.sha256(np.ascontiguousarray(oof.X, dtype=np.float64).tobytes()
                          + oof.meta[["subject", "trial"]].to_numpy(np.int64).tobytes()).hexdigest()


def run_dev(signals, cfg, result_dir=RESULTS, artifact_dir=ARTIFACTS, *, time_p3=True):
    """Cross-fitted day-1 decoders: budgets, out-of-fold logits, temperatures, P3 runtime."""
    require_training(signals)
    start = time.perf_counter()
    out, art = Path(result_dir), Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    (art / "folds").mkdir(parents=True, exist_ok=True)
    torch_threads(cfg)
    with threadpool_limits(limits=1):
        train = finite_trials(valid_covariances(signals))
    folds = sorted(map(int, np.unique(train.run)))
    if folds != list(range(1, cfg["runs"] + 1)):
        raise ValueError("Development requires the five day-1 runs")
    foldpath = lambda rc, fold: art / "folds" / f"{int(rc)}_{fold}.joblib"
    families = sorted({(s["rc"], s["d"]) for s in settings(cfg) if s["condition"] != "P0"})
    timings = []
    t0 = time.perf_counter()
    parallel(cfg, (delayed(_prepare_fold)(
        train.subset(train.run != fold), train.subset(train.run == fold), rc, foldpath(rc, fold))
        for rc in sorted({rc for rc, _ in families} | {False}) for fold in folds))
    timings.append(dict(stage="fold_features", seconds=time.perf_counter() - t0))
    t0 = time.perf_counter()
    pilots = parallel(cfg, (delayed(_pilot)(foldpath(rc, fold), rc, d, seed, fold, cfg)
                            for rc, d in families for seed in cfg["seeds"] for fold in folds))
    timings.append(dict(stage="epoch_pilot", seconds=time.perf_counter() - t0))
    history = pd.DataFrame([r for part in pilots for r in part])
    write_csv(history, out / "dev_epoch_history.csv")
    budgets = choose_epochs(history)
    t0 = time.perf_counter()
    jobs = [(s, seed, fold) for s in settings(cfg) for seed in cfg["seeds"] for fold in folds]
    packets = parallel(cfg, (delayed(_timed)(
        _oof, foldpath(s["rc"], fold), s, seed, budgets[budget_key(s["rc"], s["d"])], cfg)
        for s, seed, fold in jobs))
    timings.append(dict(stage="oof_fits", seconds=time.perf_counter() - t0))
    for (s, seed, fold), (_, seconds) in zip(jobs, packets):
        timings.append(dict(stage="oof_fit", key=s["key"], seed=seed, fold=fold,
                            seconds=seconds, train_trials=int((train.run != fold).sum())))
    temperatures, taskrows, digests = {}, [], {}
    for s in settings(cfg):
        key = s["key"]
        temperatures[key], digests[key] = {}, {}
        for seed in cfg["seeds"]:
            selected = [p for (job, (p, _)) in zip(jobs, packets)
                        if job[0]["key"] == key and job[1] == seed]
            logits = np.concatenate([p["logits"] for p in selected])
            meta = pd.concat([p["meta"] for p in selected], ignore_index=True)
            order = np.lexsort((meta.trial.to_numpy(), meta.subject.to_numpy()))
            oof = Trials(logits[order], meta.iloc[order].reset_index(drop=True))
            temperatures[key][str(seed)] = fit_temperature(oof, **cfg["temperature"])
            taskrows.extend(_task_rows(oof, oof.X, key=key, seed=seed))
            joblib.dump(oof, art / f"oof_{key}_{seed}.joblib")
            digests[key][str(seed)] = oof_digest(oof)
    write_csv(pd.DataFrame(taskrows), out / "dev_task.csv")
    p3 = {}
    if time_p3:
        p3_settings = [s for s in settings(cfg) if s["condition"] == "P3"]
        for s in p3_settings:
            budget = budgets[budget_key(s["rc"], s["d"])]
            seconds = parallel(dict(cfg, workers=1), [delayed(_time_full_fit)(
                train, s, cfg, cfg["seeds"][0], budget)])[0]
            fold_seconds = [t["seconds"] for t in timings
                            if t.get("key") == s["key"] and "fold" in t]
            p3[s["key"]] = dict(
                full_day1_fit_seconds=seconds, epochs=budget, trials=len(train),
                oof_fold_fit_seconds_mean=float(np.mean(fold_seconds)),
                oof_fold_fit_seconds_total=float(np.sum(fold_seconds)),
                projected_final_fit_seconds=seconds * len(cfg["seeds"]),
                note="one torch thread; the final M2 cost adds attacker fits on 8-d outputs")
    write_csv(pd.DataFrame(timings), out / "dev_timing.csv")
    selection = dict(
        config_hash=digest(analysis_config(cfg)),
        development_config_hash=digest(development_config(cfg)), budgets=budgets,
        temperatures=temperatures, oof_sha256=digests,
        subjects=sorted(map(int, np.unique(train.subject))), day=1, session="T", folds=folds,
        n_trials=len(signals), n_retained=len(train), versions=versions(),
        p3_runtime=p3, development_seconds=time.perf_counter() - start)
    write_json(out / "dev_decoders.json", selection)
    return selection


# ------------------------------------------------------------------ command streams

def make_stream(meta, logits, temperature, tau, segment=None):
    """K Stream (outputs 0 left, 1 hold, 2 right) of finite trials sorted by user, trial."""
    meta = meta.reset_index(drop=True)
    logits = np.asarray(logits, dtype=float)
    keep = np.isfinite(logits).all(axis=1) & meta.finite.to_numpy(bool)
    order = np.lexsort((meta.trial.to_numpy(), meta.subject.to_numpy()))
    order = order[keep[order]]
    commands = decide(logits[order], temperature, tau).astype(np.int16) + 1
    seg = np.ones(len(order), np.int8) if segment is None else np.asarray(segment)[order]
    return Stream(meta.subject.to_numpy(np.int64)[order], seg.astype(np.int8),
                  meta.label.to_numpy(np.int8)[order], commands.astype(np.int8))


def tau_key(tau):
    return f"{float(tau):.2f}"


def identity_channel(n):
    return np.repeat(np.eye(3)[None], n, axis=0)


def stream_task(stream, users, tau):
    """Per-user task statistics; utility = P(correct) - w P(wrong), w = tau / (1 - tau)."""
    return task_information(confusion_counts(stream, users), identity_channel(len(users)),
                            wrong_weight(tau))


def load_oof(art, key, seed, dev):
    oof = joblib.load(Path(art) / f"oof_{key}_{seed}.joblib")
    if oof_digest(oof) != dev["oof_sha256"][key][str(seed)]:
        raise LeakageError(f"Out-of-fold logits changed: {key} {seed}")
    return oof


def day_stream(meta, logits, temperature, tau, cfg):
    """Stream with segment 1 = trials 1-100 and 2 = the rest (same-day halves)."""
    segment = np.where(meta.trial.to_numpy() <= cfg["dev"]["segment_trials"], 1, 2)
    return make_stream(meta, logits, temperature, tau, segment)


def half(stream, h):
    return stream.subset(stream.segment == h)


def tau_grid(cfg):
    lo, hi, step = cfg["tau_rule"]["grid"]
    return np.round(np.arange(lo, hi + step / 2, step), 2)


def choose_tau(cfg, dev, art):
    """tau* = grid value whose day-1 cross-fitted hold rate of the rule condition (share of
    held trials, mean over model seeds) is closest to the target; ties -> smaller tau. Uses
    no identity outcome."""
    rule = cfg["tau_rule"]
    key, rows = rule["condition"], []
    for seed in cfg["seeds"]:
        oof = load_oof(art, key, seed, dev)
        for tau in tau_grid(cfg):
            stream = make_stream(oof.meta, oof.X, dev["temperatures"][key][str(seed)], tau)
            rows.append(dict(seed=seed, tau=float(tau), hold_rate=float((stream.output == 1).mean())))
    table = pd.DataFrame(rows).groupby("tau", as_index=False).hold_rate.mean()
    table["distance"] = (table.hold_rate - rule["target_hold_rate"]).abs().round(12)
    best = table.sort_values(["distance", "tau"]).iloc[0]
    return float(best.tau), float(best.hold_rate), table


def calibration_groups(stream, users, size, prior):
    """K's grouping: users by prior-weighted calibration net progress (exp_k.cell_job)."""
    counts = confusion_counts(stream, users)
    util = net_progress_from(np.einsum("y,uyc->uyc", np.asarray(prior), counts / np.maximum(
        counts.sum(-1, keepdims=True), 1)))[3]
    return make_groups(users, util, size)


def covered(stream, k):
    """Rows inside the stream's non-overlapping k-command windows (per user and segment)."""
    starts = window_starts(stream, k)
    return stream.subset(np.sort((starts[:, None] + np.arange(k)).ravel()))


def nanmean(arrays):
    a = np.stack(arrays)
    count = np.isfinite(a).sum(axis=0)
    return np.where(count > 0, np.nansum(a, axis=0) / np.maximum(count, 1), np.nan)


def attack(enrol, test, users, ks, attackers, alpha, groupings, rng):
    """Bayes attackers enrolled on `enrol` and tested on the rows of `test` covered by
    max(ks)-command windows, so every k scores the same rows. Returns {(attacker, k):
    (S, recall)}: S[u, v] = mean over u's windows of the percentile rank of enrolled user v
    (share of the other N-1 users scoring below; ties count one half), so diag(S) is the
    true user's rank (0.5 = chance); recall = K's per-user top-1 N-way and within-group
    recall, plus "D_<label>" = within-group recall - 1/|own group|."""
    enrolled = Counts(enrol).enrol(users)
    test = covered(test, max(ks))
    counts, chances = Counts(test), chance(users, groupings)
    out = {}
    for k in ks:
        starts = window_starts(test, k)
        win, truth = counts.windows(starts, k), test.subject[starts]
        idx = np.searchsorted(users, truth)
        n = np.bincount(idx, minlength=len(users)).astype(float)[:, None]
        for attacker in attackers:
            scores = bayes_scores(attacker, win, enrolled, alpha)
            S = np.zeros((len(users), len(users)))
            np.add.at(S, idx, (rankdata(scores, axis=1) - 1) / (len(users) - 1))
            S = np.where(n > 0, S / np.maximum(n, 1), np.nan)
            rec = recalls(scores, users, truth, groupings, rng)
            rec.update({f"D_{g}": rec[g] - chances[g] for g in groupings})
            out[(attacker, k)] = (S, rec)
    return out


def mean_attacks(parts):
    """Average attack results (nan-aware per user) over scenario pairs, seeds or draws."""
    return {key: (nanmean([p[key][0] for p in parts]),
                  {m: nanmean([p[key][1][m] for p in parts]) for m in parts[0][key][1]})
            for key in parts[0]}


def scenario_attack(pairs, users, ks, attackers, alpha, groupings, rng):
    return mean_attacks([attack(e, t, users, ks, attackers, alpha, groupings, rng)
                         for e, t in pairs])


def pool_days(tests):
    """Test days pooled per user (subject-major, day order kept); segment = day index, so
    windows and transitions never cross a day boundary."""
    subject = np.concatenate([t.subject for t in tests])
    order = np.argsort(subject, kind="stable")
    segment = np.concatenate([np.full(len(t), i + 1, np.int8) for i, t in enumerate(tests)])
    return Stream(subject[order], segment[order],
                  np.concatenate([t.target for t in tests])[order],
                  np.concatenate([t.output for t in tests])[order])


def scenarios(day1, later):
    """Cross-day day 1 -> days 2-3 (test days pooled, windows within a day), day 2 <-> 3,
    and the same-day halves of days 2 and 3: {name: [(enrolment, test), ...]}."""
    return {"cal_to_dep": [(day1, pool_days([later[2], later[3]]))],
            "dep_to_dep": [(later[2], later[3]), (later[3], later[2])],
            "same_day_halves": [(half(later[d], a), half(later[d], b)) for d in (2, 3)
                                for a, b in ((1, 2), (2, 1))]}


def user_independent(stream, rng):
    """Null stream: each command redrawn from the pooled P(output | target) of the stream."""
    pooled = confusion_counts(Stream(np.ones(len(stream), np.int64), stream.segment,
                                     stream.target, stream.output), np.array([1]))[0]
    p = (pooled / pooled.sum(-1, keepdims=True))[stream.target.astype(int)]
    return stream.with_output((rng.random(len(stream))[:, None] > np.cumsum(p, axis=1)[:, :2])
                              .sum(axis=1))


# ------------------------------------------------------------------ claim statistics

def impute(vectors, chances, max_missing):
    """Missing-cell rule: users with a missing value in any vector get chance in every
    vector, up to max_missing users; more -> (None, count) (claim not evaluable)."""
    missing = np.zeros(len(vectors[0]), bool)
    for v in vectors:
        missing |= ~np.isfinite(v)
    if missing.sum() > max_missing:
        return None, int(missing.sum())
    return [np.where(missing, c, v) for v, c in zip(vectors, chances)], int(missing.sum())


def t_value(v):
    v = np.asarray(v, float)
    sd = v.std(ddof=1)
    return float(v.mean() / (sd / np.sqrt(len(v)))) if sd > 0 else float(np.sign(v.mean()) * np.inf)


def interval(v):
    v = np.asarray(v, float)
    lo, hi = t_interval(v)
    return dict(mean=float(np.nanmean(v)), ci_low=lo, ci_high=hi,
                missing=int((~np.isfinite(v)).sum()))


def m1_test(r, n_users, cfg):
    """r (attackers, U) cross-day recall: p = min(1, J min_j T(r_j > factor/N))."""
    threshold = cfg["m1"]["threshold_factor"] / n_users
    vectors, missing = impute(list(r), [1.0 / n_users] * len(r), cfg["missing_cells_max"])
    if vectors is None:
        return dict(evaluable=False, missing=missing, p=1.0, threshold=threshold)
    p = float(min(1.0, len(vectors) * np.min([t_greater(v, threshold) for v in vectors])))
    return dict(evaluable=True, missing=missing, p=p, threshold=threshold,
                strongest=float(np.max([v.mean() for v in vectors])))


def m2_test(battery, head, n_users, cfg):
    """battery (attackers, U), head (U): attacker j is eligible if its observed mean gap
    (battery - head) >= margin; p = min(1, J min_{eligible j} T(gap_j > 0)), 1 if none."""
    chances = [1.0 / n_users] * (len(battery) + 1)
    vectors, missing = impute(list(battery) + [head], chances, cfg["missing_cells_max"])
    if vectors is None:
        return dict(evaluable=False, missing=missing, p=1.0)
    gaps = np.stack(vectors[:-1]) - vectors[-1]
    observed = gaps.mean(axis=1)
    eligible = observed >= cfg["m2"]["margin"]
    p = float(min(1.0, len(gaps) * np.min(t_greater(gaps[eligible], 0.0)))) \
        if eligible.any() else 1.0
    return dict(evaluable=True, missing=missing, p=p, gaps=observed.tolist(),
                best_gap=float(observed.max()), head=float(vectors[-1].mean()),
                battery=[float(v.mean()) for v in vectors[:-1]])


def m3_test(S_star, S_base, cfg, rng):
    """Mean percentile rank of the true user at K* must exceed 0.5 and its own k_base value:
    p = max of the two one-sided one-sample/paired t-tests; label-permutation p (enrolled
    identities permuted, same permutation at both k) as a sensitivity check."""
    r_star, r_base = np.diag(S_star), np.diag(S_base)
    vectors, missing = impute([r_star, r_base], [0.5, 0.5], cfg["missing_cells_max"])
    if vectors is None:
        return dict(evaluable=False, missing=missing, p=1.0)
    r_star, r_base = vectors
    p1, p2 = float(t_greater(r_star, 0.5)), float(t_greater(r_star - r_base, 0.0))
    excess = np.nan_to_num(S_star - 0.5)
    gain = np.nan_to_num(S_star - S_base)
    n = len(r_star)
    perms = np.argsort(rng.random((cfg["m3"]["permutations"], n)), axis=1)
    rows = np.arange(n)
    null1, null2 = excess[rows, perms].mean(axis=1), gain[rows, perms].mean(axis=1)
    q1 = (1 + (null1 >= (r_star - 0.5).mean() - 1e-12).sum()) / (1 + len(perms))
    q2 = (1 + (null2 >= (r_star - r_base).mean() - 1e-12).sum()) / (1 + len(perms))
    return dict(evaluable=True, missing=missing, p=max(p1, p2), p_vs_chance=p1,
                p_vs_k_base=p2, t_vs_chance=t_value(r_star - 0.5),
                t_vs_k_base=t_value(r_star - r_base), rank_k_star=interval(r_star),
                rank_k_base=interval(r_base), permutation_p=float(max(q1, q2)),
                permutation_p_components=[float(q1), float(q2)])


def m3_power(S_star, S_base, cfg, rng, family_size):
    """Rank-based power of M3 when days 2-3 retain a share of the day-1 excess (both vs
    chance and vs k_base), keeping the day-1 between-user SD: users resampled with
    replacement, power = P(max of the two one-sided p <= level)."""
    r_star, r_base = np.diag(S_star), np.diag(S_base)
    ok = np.isfinite(r_star) & np.isfinite(r_base)
    e1, e2 = r_star[ok] - 0.5, (r_star - r_base)[ok]
    n, pw = len(e1), cfg["power"]
    idx = rng.integers(0, n, size=(pw["n_sim"], n))
    levels = {"alpha": cfg["alpha"], f"alpha/{family_size}": cfg["alpha"] / family_size}
    rows = []
    for retention in pw["retentions"]:
        x1 = retention * e1.mean() + e1 - e1.mean()
        x2 = retention * e2.mean() + e2 - e2.mean()
        p = np.maximum(t_greater(x1[idx], 0.0), t_greater(x2[idx], 0.0))
        rows.append(dict(retention=retention, excess_vs_chance=float(x1.mean()),
                         excess_vs_k_base=float(x2.mean()), sd_vs_chance=float(e1.std(ddof=1)),
                         sd_vs_k_base=float(e2.std(ddof=1)), users=n,
                         **{f"power_{name}": float((p <= level).mean())
                            for name, level in levels.items()}))
    return rows


# ------------------------------------------------------------------ defences (estimates)

def eq_fit(day1, users, group, cfg, tau):
    """EQ on day 1: matrices from all of day 1 (applied to later days) and cross-fitted
    matrices (applied to the day-1 enrolment stream itself)."""
    est, w = cfg["estimates"], wrong_weight(tau)
    M = fit_eq(confusion_counts(day1, users), group, est["eq_variant"], cfg["target_prior"],
               cfg["eq_smoothing"], w)[0]
    fold, mats = crossfit_matrices(day1, users, group, est["eq_variant"], cfg["target_prior"],
                                   cfg["eq_smoothing"], w, est["eq_crossfit_folds"])
    return M, fold, mats


def own_groups(users):
    """Every user its own group: K's LP then only recalibrates each user's commands to the
    utility optimum, without equalization (the utility baseline of EQ's privacy cost)."""
    return np.arange(len(users))


def day1_utilities(day1, users, group, cfg, tau):
    """Per-user expected day-1 weighted utility of the undefended stream, of
    recalibration only and of EQ (both cross-fitted, as the day-1 enrolment streams)."""
    w = wrong_weight(tau)
    out = dict(undefended=stream_task(day1, users, tau)["utility"])
    for name, g in (("recalibration", own_groups(users)), ("eq", group)):
        _, fold, mats = eq_fit(day1, users, g, cfg, tau)
        joint = sum(np.einsum("uyc,uco->uyo", confusion_counts(day1.subset(fold == f), users),
                              mats[f]) for f in range(len(mats)))
        correct, wrong = net_progress_from(joint / joint.sum((1, 2), keepdims=True))[:2]
        out[name] = correct - w * wrong
    return out


def rr_matched_q(recalibration, eq, tau):
    """q such that k-ary RR over 3 outputs applied to the recalibration-only stream has EQ's
    mean weighted utility: U(q) = (1 - q) U_r + q (1 - w) / 3, so
    q = (U_r - U_eq) / (U_r - (1 - w) / 3); clipped to [0, 1]."""
    ur, ue = float(np.nanmean(recalibration)), float(np.nanmean(eq))
    return float(np.clip((ur - ue) / (ur - (1 - wrong_weight(tau)) / 3), 0.0, 1.0))


def defences(day1, users, group, cfg, rr_matched, tau):
    """[(name, q, channel for later days, enrolment draw for day 1)]. EQ and
    recalibration-only enrol on their cross-fitted day-1 streams; rr_matched is RR(q)
    applied after recalibration."""
    n, out = len(users), []
    for name, g in (("eq", group), ("recal", own_groups(users))):
        M, fold, mats = eq_fit(day1, users, g, cfg, tau)
        out.append((name, None, M,
                    lambda r, fold=fold, mats=mats: apply_crossfit(day1, fold, mats, users, r)))
    for q in cfg["estimates"]["rr_q"]:
        R = np.repeat(rr_matrix(q)[None], n, axis=0)
        out.append((f"rr_q{q:g}", q, R, lambda r, R=R: apply_channel(day1, R, users, r)))
    recal, recal_draw = out[1][2], out[1][3]
    R = np.repeat(rr_matrix(rr_matched)[None], n, axis=0)
    out.append(("rr_matched", rr_matched, recal @ R,
                lambda r: apply_channel(recal_draw(r), R, users, r)))
    return out


def utility_vectors(pairs, M, users, tau):
    """Per-user expected task statistics of the scenario's test streams, undefended and
    through channel M (mean over pairs)."""
    w, out = wrong_weight(tau), {}
    for _, test in pairs:
        counts = confusion_counts(test, users)
        und = task_information(counts, identity_channel(len(users)), w)
        dfd = task_information(counts, M, w)
        for stat in ("utility", "net", "wrong", "hold"):
            out.setdefault(f"{stat}_undefended", []).append(und[stat])
            out.setdefault(f"{stat}_defended", []).append(dfd[stat])
    return {k: nanmean(v) for k, v in out.items()}


def run_streams(signals, cfg, result_dir=RESULTS, artifact_dir=ARTIFACTS):
    """Day-1 development of the claims: tau*, stream summaries, within-day identification
    (first vs second 100 trials, both directions), the M1-M3 gates and triviality checks,
    the day-1 utilities of EQ and recalibration only, the q at which RR after recalibration
    matches EQ's day-1 utility, and the M3 power table."""
    out, art = Path(result_dir), Path(artifact_dir)
    dev = json.loads((out / "dev_decoders.json").read_text(encoding="utf-8"))
    users = np.asarray(cfg["subjects"], dtype=np.int64)
    rng = rng_for(cfg["root_seed"], "dev_streams")
    tau_star, hold, table = choose_tau(cfg, dev, art)
    write_csv(table, out / "dev_tau_grid.csv")
    if cfg.get("tau_star") is not None and tau_key(cfg["tau_star"]) != tau_key(tau_star):
        raise ValueError(f"configs/exp_m.yaml tau_star differs from the tau rule ({tau_star})")
    taus = [tau_star, cfg["tau_rule"]["secondary_tau"]]
    m3c, est, alpha = cfg["m3"], cfg["estimates"], cfg["dirichlet_alpha"]
    attackers, label = cfg["bayes_attackers"], f"g{cfg['estimates']['g']}"
    ks = sorted(set(m3c["k_curve"]) | {m3c["k_star"], m3c["k_base"]} | set(est["ks"]))
    rows, ident, utilities = [], {}, {}
    for s in settings(cfg):
        key = s["key"]
        for seed in cfg["seeds"]:
            oof = load_oof(art, key, seed, dev)
            temperature = dev["temperatures"][key][str(seed)]
            for tau in taus:
                stream = day_stream(oof.meta, oof.X, temperature, tau, cfg)
                task = stream_task(stream, users, tau)
                rows.append(dict(key=key, seed=seed, tau=tau, temperature=temperature,
                                 trials=len(stream), hold_rate=float((stream.output == 1).mean()),
                                 **{stat: float(np.nanmean(task[stat])) for stat in
                                    ("correct", "wrong", "hold", "net", "utility", "bits")}))
                if key != m3c["condition"]:
                    continue
                parts = []
                for a, b in ((1, 2), (2, 1)):  # groups from the enrolment half only
                    group = calibration_groups(half(stream, a), users, est["g"],
                                               cfg["target_prior"])
                    parts.append(attack(half(stream, a), half(stream, b), users, ks, attackers,
                                        alpha, {label: group}, rng))
                ident.setdefault(tau_key(tau), []).append(mean_attacks(parts))
                group = calibration_groups(stream, users, est["g"], cfg["target_prior"])
                utilities.setdefault(tau_key(tau), []).append(
                    day1_utilities(stream, users, group, cfg, tau))
    streams = pd.DataFrame(rows)
    write_csv(streams, out / "dev_streams.csv")
    ident = {t: mean_attacks(parts) for t, parts in ident.items()}
    summary = []
    for t, res in ident.items():
        for (attacker, k), (S, rec) in res.items():
            rank = np.diag(S)
            summary.append(dict(tau=float(t), attacker=attacker, k=k,
                                rank_t=t_value(rank[np.isfinite(rank)] - 0.5),
                                **{f"rank_{x}": v for x, v in interval(rank).items()},
                                **{f"top1_{x}": v for x, v in interval(rec["nway"]).items()},
                                **{f"{label}_excess_{x}": v
                                   for x, v in interval(rec[f"D_{label}"]).items()}))
    summary = pd.DataFrame(summary)
    write_csv(summary, out / "dev_identification_summary.csv")
    a3, kst, kb = m3c["attacker"], m3c["k_star"], m3c["k_base"]
    res = ident[tau_key(tau_star)]
    m3 = m3_test(res[(a3, kst)][0], res[(a3, kb)][0], cfg, rng)
    null_parts = []
    for seed in cfg["seeds"]:
        oof = load_oof(art, m3c["condition"], seed, dev)
        stream = user_independent(day_stream(
            oof.meta, oof.X, dev["temperatures"][m3c["condition"]][str(seed)], tau_star, cfg), rng)
        null_parts += [attack(half(stream, a), half(stream, b), users, [kb, kst], [a3], alpha,
                              {}, rng) for a, b in ((1, 2), (2, 1))]
    null = mean_attacks(null_parts)
    m3_null = m3_test(null[(a3, kst)][0], null[(a3, kb)][0], cfg, rng)
    gate3 = m3c["gate"]
    in_range = gate3["tau_range"][0] - 1e-9 <= tau_star <= gate3["tau_range"][1] + 1e-9
    m3_pass = bool(in_range and m3.get("t_vs_chance", -np.inf) >= gate3["min_t"]
                   and m3_null["p"] > cfg["alpha"])
    m2 = m2_within_day(signals, cfg, dev, users, rng)
    m2_pass = bool(m2["real"].get("best_gap", -np.inf) >= cfg["m2"]["gate"]["min_within_day_gap"]
                   and m2["null"]["p"] > cfg["alpha"])
    gates = dict(
        tau=dict(tau_star=tau_star, hold_rate=hold, target=cfg["tau_rule"]["target_hold_rate"],
                 condition=cfg["tau_rule"]["condition"], secondary_tau=taus[1]),
        M1=m1_dev(cfg, users, rng),
        M2=dict(m2, passed=m2_pass, status="confirmatory" if m2_pass else "descriptive",
                scenario="within-day, day 1 (first vs second 100 trials, both directions)"),
        M3=dict(real=m3, null=m3_null, tau_in_range=bool(in_range), passed=m3_pass,
                status="confirmatory" if m3_pass else "estimate", attacker=a3, k_star=kst,
                k_base=kb, scenario="within-day, day 1 (first vs second 100 trials, both "
                                    "directions)"))
    family = 2 + int(m2_pass)  # M1, M2 if active, and M3
    power = pd.DataFrame(m3_power(res[(a3, kst)][0], res[(a3, kb)][0], cfg, rng, family))
    write_csv(power, out / "dev_m3_power.csv")
    rr = {}
    for t, parts in utilities.items():
        u = {name: nanmean([p[name] for p in parts]) for name in parts[0]}
        rr[t] = dict(q=rr_matched_q(u["recalibration"], u["eq"], float(t)),
                     wrong_weight=wrong_weight(float(t)),
                     day1_utility={name: interval(v) for name, v in u.items()},
                     eq_privacy_cost=interval(u["recalibration"] - u["eq"]),
                     eq_loss_vs_undefended=interval(u["undefended"] - u["eq"]),
                     recalibration_gain=interval(u["recalibration"] - u["undefended"]))
    return dict(streams=streams, summary=summary, gates=gates, rr_matched=rr,
                power=power, tau_grid=table)


def head_recall(model, trials, users):
    pred = model.identity_classes_[model.adversary_logits(trials).argmax(axis=1)]
    return (pd.Series(pred == trials.subject).groupby(trials.subject).mean().reindex(users)
            .to_numpy(float))


def m2_within_day(signals, cfg, dev, users, rng):
    """M2 gate on day 1: the M2 condition fitted on one half; the independent battery
    (trained on that half's representation) vs the co-trained identity head on the other
    half, both directions, every seed. The null permutes the battery's training identities."""
    s = cfg["conditions"][cfg["m2"]["condition"]]
    budget = dev["budgets"][budget_key(s["rc"], s["d"])]
    torch_threads(cfg)
    with threadpool_limits(limits=1):
        train = finite_trials(valid_covariances(signals))
    seg = np.where(train.meta.trial.to_numpy() <= cfg["dev"]["segment_trials"], 1, 2)
    names = ("logistic", "rbf_svm", "mlp")
    real, null, heads, seconds = [], [], [], []
    for seed in cfg["seeds"]:
        for a, b in ((1, 2), (2, 1)):
            A, B = train.subset(seg == a), train.subset(seg == b)
            t0 = time.perf_counter()
            with threadpool_limits(limits=1):
                model = fit_defended(A, s["condition"], training_config(cfg, s["d"]), seed,
                                     budget, s["target"])
                rep_a, rep_b = model(A)[0], model(B)[0]
            seconds.append(time.perf_counter() - t0)
            heads.append(head_recall(model, B, users))
            for target, labels in ((real, A.subject), (null, rng.permutation(A.subject))):
                v = attack_vectors(rep_a, labels, [(rep_b, B.subject)], users, cfg, seed)
                target.append(np.stack([v[name] for name in names]))
    head = nanmean(heads)
    return dict(real=m2_test(nanmean(real), head, len(users), cfg),
                null=m2_test(nanmean(null), head, len(users), cfg), attackers=list(names),
                condition=cfg["m2"]["condition"], epochs=budget,
                half_day_fit_seconds=float(np.mean(seconds)))


# ------------------------------------------------------------------ full-feature identification

def attack_vectors(enrol_X, enrol_users, tests, users, cfg, seed):
    """Per-user recall of the fixed L battery (logistic, RBF SVM, MLP) trained on one
    enrolment set and averaged equally over the test sets."""
    lcfg = dict(attackers=cfg["attackers"])
    with threadpool_limits(limits=1):
        models = _fit_attack(enrol_X, enrol_users, lcfg, seed)
    out = {}
    for name, model in models.items():
        recalls = []
        for X, truth in tests:
            pred = model.predict(X)
            recalls.append(pd.Series(pred == truth).groupby(truth).mean()
                           .reindex(users).to_numpy(float))
        out[name] = np.mean(recalls, axis=0)
        if name == "mlp":
            out["mlp_convergence"] = model.convergence_
    return out


def m1_dev(cfg, users, rng):
    """Within-day full-feature identification on day 1 and a permuted-identity null."""
    path = Path(cfg.get("_artifact_dir", ARTIFACTS)) / "dev_p0_features.joblib"
    feats = joblib.load(path)
    X, meta = feats["X"], feats["meta"]
    seg = np.where(meta.trial.to_numpy() <= cfg["dev"]["segment_trials"], 1, 2)
    subj = meta.subject.to_numpy()
    real, null = [], []
    for a, b in ((1, 2), (2, 1)):
        A, B = seg == a, seg == b
        real.append(attack_vectors(X[A], subj[A], [(X[B], subj[B])], users, cfg, 0))
        shuffled = rng.permutation(subj[A])
        null.append(attack_vectors(X[A], shuffled, [(X[B], subj[B])], users, cfg, 0))
    names = ("logistic", "rbf_svm", "mlp")
    r = np.stack([np.mean([x[name] for x in real], axis=0) for name in names])
    z = np.stack([np.mean([x[name] for x in null], axis=0) for name in names])
    threshold = cfg["m1"]["threshold_factor"] / len(users)
    p_real = float(min(1.0, len(r) * np.min([t_greater(v, threshold) for v in r])))
    p_null = float(min(1.0, len(z) * np.min([t_greater(v, threshold) for v in z])))
    return dict(criterion="strongest of 3 attackers: mean full-feature recall > 10/N",
                scenario="within-day, day 1 (first vs second 100 trials, both directions)",
                threshold=threshold, real_p=p_real,
                real=dict(zip(names, map(float, r.mean(1)))), null_p=p_null,
                null=dict(zip(names, map(float, z.mean(1)))), null_fails=bool(p_null > 0.05),
                note="within-day values are far above the threshold; M1 is a low-risk "
                     "replication estimate and is reported as such")


def dev_features(signals, cfg, artifact_dir):
    """P0 representation (unstandardized 190-d tangent vectors at the day-1 Riemannian mean)
    of every retained day-1 trial, for the M1 within-day check."""
    with threadpool_limits(limits=1):
        train = finite_trials(valid_covariances(signals))
        model = fit_defended(train, "P0", training_config(cfg, 8), cfg["seeds"][0], 1)
        X, _ = model(train)
    joblib.dump(dict(X=X, meta=train.meta), Path(artifact_dir) / "dev_p0_features.joblib")


# ------------------------------------------------------------------ final (days 2-3)

def final_streams(model_logits, cfg, selection, key, seed, tau, art):
    """Day-1 cross-fitted stream (enrolment) and the frozen model's day-2/3 streams."""
    oof = load_oof(art, key, seed, selection)
    temperature = selection["temperatures"][key][str(seed)]
    day1 = day_stream(oof.meta, oof.X, temperature, tau, cfg)
    later = {day: day_stream(meta, logits, temperature, tau, cfg)
             for day, (meta, logits) in model_logits.items()}
    return day1, later


def check_selection(cfg, selection, root=ROOT):
    """The final analysis runs only on the configuration, tau*, claim statuses and input files
    (exp_m's import closure, both configs, the fetch script) of the sealed day-1 selection."""
    if selection["config_hash"] != digest(analysis_config(cfg)):
        raise LeakageError("Configuration changed since day-1 development")
    recorded = selection.get("input_sha256", {})
    changed = sorted(p for p, h in input_hashes(root).items() if recorded.get(p) != h)
    if changed:
        raise LeakageError(f"Inputs differ from those of the day-1 selection: {changed}")
    gates = selection["gates"]
    if cfg.get("tau_star") is None or tau_key(cfg["tau_star"]) != tau_key(gates["tau"]["tau_star"]):
        raise LeakageError("tau_star is not the tau rule's day-1 result")
    for claim in ("M2", "M3"):
        if cfg[claim.lower()]["status"] != gates[claim]["status"]:
            raise LeakageError(f"{claim} status differs from its day-1 gate")


def estimate_tables(und, dfd, util, cfg, n_users):
    """Pre-registered estimates (no pass/fail): within-group excess and N-way recall
    undefended/defended and their difference; task statistics and utility loss; M3's
    secondary estimates (rank and top-1 for every attacker along the k curve)."""
    label = f"g{cfg['estimates']['g']}"
    ident, utility, curves = [], [], []
    for (t, scen), parts in und.items():
        res = mean_attacks(parts)
        for (attacker, k), (S, rec) in res.items():
            rank, top1 = np.diag(S), rec["nway"]
            curves.append(dict(tau=float(t), scenario=scen, attacker=attacker, k=k,
                               top1_vs_1N=t_greater(top1[np.isfinite(top1)], 1 / n_users).item(),
                               top1_vs_2N=t_greater(top1[np.isfinite(top1)], 2 / n_users).item(),
                               **{f"rank_{x}": v for x, v in interval(rank).items()},
                               **{f"top1_{x}": v for x, v in interval(top1).items()}))
    for (t, defence, scen), parts in dfd.items():
        base, res = mean_attacks(und[(t, scen)]), mean_attacks(parts)
        for (attacker, k), (_, rec) in res.items():
            rec0 = base[(attacker, k)][1]
            values = {f"{label}_excess_undefended": rec0[f"D_{label}"],
                      f"{label}_excess_defended": rec[f"D_{label}"],
                      f"{label}_excess_reduction": rec0[f"D_{label}"] - rec[f"D_{label}"],
                      "nway_undefended": rec0["nway"], "nway_defended": rec["nway"]}
            for stat, v in values.items():
                ident.append(dict(tau=float(t), defence=defence, scenario=scen,
                                  attacker=attacker, k=k, statistic=stat, **interval(v)))
    for (t, defence, scen), parts in util.items():
        values = {k: nanmean([p[k] for p in parts]) for k in parts[0]}
        for stat in ("utility", "net", "wrong", "hold"):
            values[f"{stat}_loss"] = values[f"{stat}_undefended"] - values[f"{stat}_defended"]
        if defence != "recal":  # privacy cost: against the recalibration-only optimum
            recal = util[(t, "recal", scen)]
            for stat in ("utility", "net"):
                values[f"{stat}_privacy_cost"] = nanmean(
                    [p[f"{stat}_defended"] for p in recal]) - values[f"{stat}_defended"]
        for stat, v in values.items():
            utility.append(dict(tau=float(t), defence=defence, scenario=scen, statistic=stat,
                                **interval(v)))
    return pd.DataFrame(ident), pd.DataFrame(utility), pd.DataFrame(curves)


def run_final(training, evaluation, cfg, selection, result_dir, artifact_dir):
    """Sealed analysis of days 2-3. Seal verification happens in main before data access."""
    require_training(training)
    if set(evaluation.session) != {"E"} or set(evaluation.meta.day) != {2, 3}:
        raise ValueError("Final evaluation must be recording days 2 and 3")
    check_selection(cfg, selection)
    start = time.perf_counter()
    out, art = Path(result_dir), Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    (art / "final").mkdir(parents=True, exist_ok=True)
    users = np.asarray(cfg["subjects"], dtype=np.int64)
    n = len(users)
    m3c, est, alpha = cfg["m3"], cfg["estimates"], cfg["dirichlet_alpha"]
    attackers, label = cfg["bayes_attackers"], f"g{est['g']}"
    ks = sorted(set(m3c["k_curve"]) | {m3c["k_star"], m3c["k_base"]} | set(est["ks"]))
    taus = [cfg["tau_star"], cfg["tau_rule"]["secondary_tau"]]
    torch_threads(cfg)
    with threadpool_limits(limits=1):
        T = valid_covariances(training)
        E = valid_covariances(evaluation)
    Tf = finite_trials(T)
    days = {d: E.subset(E.meta.day.to_numpy() == d) for d in (2, 3)}
    exclusions = pd.concat([training.meta, evaluation.meta]).groupby(["subject", "day"]).finite \
        .agg(total="size", retained="sum").reindex(pd.MultiIndex.from_product(
            [users, (1, 2, 3)], names=["subject", "day"]), fill_value=0).reset_index()
    write_csv(exclusions, out / "final_exclusions.csv")
    missing_user_days = exclusions.loc[exclusions.total == 0, ["subject", "day"]] \
        .to_dict(orient="records")
    timing, task_rows = [], []
    und, dfd, util, m2_parts = {}, {}, {}, []
    p0_reps = None
    for s in settings(cfg):
        key = s["key"]
        for seed in cfg["seeds"]:
            t0 = time.perf_counter()
            with threadpool_limits(limits=1):
                model = fit_defended(Tf, s["condition"], training_config(cfg, s["d"]), seed,
                                     selection["budgets"][budget_key(s["rc"], s["d"])],
                                     s["target"])
                outputs = {d: model(days[d]) for d in (2, 3)}
                day1_rep = model(Tf)[0]
            joblib.dump(model, art / "final" / f"model_{key}_{seed}.joblib")
            timing.append(dict(stage="fit", key=key, seed=seed,
                               seconds=time.perf_counter() - t0))
            for d in (2, 3):
                good = days[d].meta.finite.to_numpy(bool)
                task_rows.extend(_task_rows(days[d].subset(good), outputs[d][1][good], key=key,
                                            seed=seed, day=d))
            if key == "P0" and seed == cfg["seeds"][0]:
                p0_reps = (day1_rep, {d: outputs[d][0] for d in (2, 3)})
            if key == m3c["condition"]:
                for tau in taus:
                    t = tau_key(tau)
                    day1, later = final_streams(
                        {d: (days[d].meta, outputs[d][1]) for d in (2, 3)}, cfg, selection,
                        key, seed, tau, art)
                    scen = scenarios(day1, later)
                    group = calibration_groups(day1, users, est["g"], cfg["target_prior"])
                    rng = rng_for(cfg["root_seed"], "final", key, seed, t)
                    for name, pairs in scen.items():
                        und.setdefault((t, name), []).append(scenario_attack(
                            pairs, users, ks, attackers, alpha, {label: group}, rng))
                    for dname, _, M, enrol_draw in defences(
                            day1, users, group, cfg, selection["rr_matched"][t]["q"], tau):
                        for pp in cfg["postprocessing_seeds"]:
                            prng = rng_for(cfg["root_seed"], "pp", key, seed, t, dname, pp)
                            mapped = {d: apply_channel(later[d], M, users, prng) for d in (2, 3)}
                            for name, pairs in scenarios(enrol_draw(prng), mapped).items():
                                dfd.setdefault((t, dname, name), []).append(scenario_attack(
                                    pairs, users, est["ks"], attackers, alpha, {label: group},
                                    prng))
                        for name, pairs in scen.items():
                            util.setdefault((t, dname, name), []).append(
                                utility_vectors(pairs, M, users, tau))
            if key == cfg["m2"]["condition"]:
                m2_parts.append(m2_vectors(model, Tf, days, day1_rep, outputs, users, cfg, seed))
    write_csv(pd.DataFrame(task_rows), out / "final_task.csv")
    ident, utility, curves = estimate_tables(und, dfd, util, cfg, n)
    write_csv(ident, out / "final_estimates_identification.csv")
    write_csv(utility, out / "final_estimates_utility.csv")
    write_csv(curves, out / "final_estimates_k_curves.csv")
    t0 = time.perf_counter()
    r, rnames = m1_vectors(p0_reps, Tf, days, users, cfg)
    timing.append(dict(stage="m1_battery", seconds=time.perf_counter() - t0))
    m1 = {sc: dict(m1_test(np.stack([v for nm, v in zip(rnames, r) if nm.startswith(sc + "/")]),
                           n, cfg), attackers={nm: float(np.nanmean(v))
                                               for nm, v in zip(rnames, r) if nm.startswith(sc)})
          for sc in ("cal_to_dep", "dep_to_dep")}
    battery = np.stack([nanmean([p["battery"][a] for p in m2_parts])
                        for a in ("logistic", "rbf_svm", "mlp")])
    m2 = m2_test(battery, nanmean([p["head"] for p in m2_parts]), n, cfg)
    t3 = tau_key(cfg["tau_star"])
    res = mean_attacks(und[(t3, "cal_to_dep")])
    a3 = m3c["attacker"]
    m3 = m3_test(res[(a3, m3c["k_star"])][0], res[(a3, m3c["k_base"])][0], cfg,
                 rng_for(cfg["root_seed"], "m3_permutation"))
    claims = dict(M1=m1["cal_to_dep"], M2=m2, M3=m3)
    family = ["M1"] + [c for c in ("M2", "M3") if cfg[c.lower()]["status"] == "confirmatory"]
    pvalues = {c: claims[c]["p"] for c in family}
    adjusted = holm(pvalues)
    decision = dict(
        alpha=cfg["alpha"], holm_family=family, p=pvalues, p_holm=adjusted,
        not_evaluable=[c for c in claims if not claims[c]["evaluable"]],
        missing_user_days=missing_user_days,
        supported={c: bool(v <= cfg["alpha"]) for c, v in adjusted.items()},
        M1=dict(m1["cal_to_dep"], scenario="cal_to_dep",
                strongest=float(np.max(list(m1["cal_to_dep"]["attackers"].values())))),
        M1_dep_to_dep_estimate=m1["dep_to_dep"],
        M2=dict(m2, status=cfg["m2"]["status"], scenario="cal_to_dep"),
        M3=dict(m3, status=cfg["m3"]["status"], scenario="cal_to_dep", tau=cfg["tau_star"],
                attacker=a3, k_star=m3c["k_star"], k_base=m3c["k_base"]),
        seconds=time.perf_counter() - start)
    write_json(out / "decision.json", decision)
    write_csv(pd.DataFrame(timing), out / "final_timing.csv")
    return decision


def m1_vectors(p0_reps, Tf, days, users, cfg):
    """Cross-day full-feature identification: candidates = L battery x {cal_to_dep,
    dep_to_dep}; recall per user averaged over the scenario's test days."""
    day1_rep, later = p0_reps
    subj = {d: days[d].subject for d in (2, 3)}
    good = {d: np.isfinite(later[d]).all(axis=1) for d in (2, 3)}
    seed = cfg["seeds"][0]
    cal = attack_vectors(day1_rep, Tf.subject, [(later[d][good[d]], subj[d][good[d]])
                                                for d in (2, 3)], users, cfg, seed)
    dep = [attack_vectors(later[a][good[a]], subj[a][good[a]],
                          [(later[b][good[b]], subj[b][good[b]])], users, cfg, seed)
           for a, b in ((2, 3), (3, 2))]
    names, rows = [], []
    for attacker in ("logistic", "rbf_svm", "mlp"):
        names += [f"cal_to_dep/{attacker}", f"dep_to_dep/{attacker}"]
        rows += [cal[attacker], np.mean([x[attacker] for x in dep], axis=0)]
    return np.stack(rows), names


def m2_vectors(model, Tf, days, day1_rep, outputs, users, cfg, seed):
    """Independent battery on the P3 representation vs the co-trained identity head,
    cross-day (day 1 -> days 2 and 3)."""
    tests = []
    head = []
    for d in (2, 3):
        good = np.isfinite(outputs[d][0]).all(axis=1)
        tests.append((outputs[d][0][good], days[d].subject[good]))
        scores = model.adversary_logits(days[d].subset(good))
        pred = model.identity_classes_[scores.argmax(axis=1)]
        head.append(pd.Series(pred == days[d].subject[good]).groupby(days[d].subject[good])
                    .mean().reindex(users).to_numpy(float))
    battery = attack_vectors(day1_rep, Tf.subject, tests, users, cfg, seed)
    return dict(battery={a: battery[a] for a in ("logistic", "rbf_svm", "mlp")},
                head=np.mean(head, axis=0))


# ------------------------------------------------------------------ seal verification

def import_closure(module="exp_m", root=ROOT):
    """beat modules in the static transitive import set of `module` (function-level
    imports included) plus beat/__init__.py."""
    seen, todo = set(), [module]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        tree = ast.parse((Path(root) / "beat" / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level == 1 and node.module:
                    todo.append(node.module.split(".")[0])
                elif node.level == 1:
                    todo.extend(a.name for a in node.names)
                elif node.module and node.module.startswith("beat."):
                    todo.append(node.module.split(".")[1])
            elif isinstance(node, ast.Import):
                todo.extend(a.name.split(".")[1] for a in node.names
                            if a.name.startswith("beat."))
    return sorted(f"beat/{name}.py" for name in seen | {"__init__"})


def input_files(root=ROOT):
    """Files whose bytes the day-1 selection depends on: exp_m's import closure, both
    configs and the fetch script."""
    return sorted(set(import_closure(root=root)) | set(SELECTION_INPUTS))


def input_hashes(root=ROOT):
    return {path: file_sha256(Path(root) / path) for path in input_files(root)}


def verify_seal(commit, cfg, root=ROOT):
    """Seal-4 checks before any day-2/3 access: annotated tag seal-4 naming `commit`, an
    ancestor of HEAD; the supplied config equals the committed one with frozen thresholds;
    every seal-4 file and every beat module in exp_m's import closure byte-equal to the
    seal commit; the sealed selection's recorded input hashes equal the sealed bytes of those
    inputs; package versions equal to those recorded in the sealed selection; and the seal
    manifest (seals/manifest.yaml, beat.verify.check_seals) names seal-4 and passes."""
    if not commit:
        raise LeakageError("The seal-4 commit is required")
    root = Path(root)

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL)
    try:
        if git("cat-file", "-t", "seal-4").decode().strip() != "tag":
            raise LeakageError("seal-4 must be an annotated tag")
        oid = git("rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
        if git("rev-parse", "--verify", "seal-4^{commit}").decode().strip() != oid:
            raise LeakageError("Tag seal-4 does not name the supplied commit")
        git("merge-base", "--is-ancestor", oid, "HEAD")
        sealed_cfg = yaml.safe_load(git("show", f"{oid}:configs/exp_m.yaml"))
        if sealed_cfg != cfg:
            raise LeakageError("Supplied config differs from the sealed config")
        if cfg.get("thresholds_status") != "frozen":
            raise LeakageError("Thresholds are not frozen in the sealed config")
        for path in dict.fromkeys(SEAL_FILES + tuple(import_closure(root=root))):
            if git("show", f"{oid}:{path}") != (root / path).read_bytes():
                raise LeakageError(f"Sealed input differs: {path}")
        sealed_inputs = {path: hashlib.sha256(git("show", f"{oid}:{path}")).hexdigest()
                         for path in input_files(root)}
    except subprocess.CalledProcessError as exc:
        raise LeakageError("Unable to verify the seal-4 commit") from exc
    selection = json.loads((root / "results/exp_m/selection.json").read_text(encoding="utf-8"))
    recorded = selection.get("input_sha256", {})
    stale = sorted(p for p, h in sealed_inputs.items() if recorded.get(p) != h)
    if stale:
        raise LeakageError(f"The sealed selection was made with other inputs than the sealed "
                           f"ones: {stale}")
    if selection["versions"] != versions():
        raise LeakageError(f"Package versions differ from the sealed ones: {versions()}")
    if selection["config_hash"] != digest(analysis_config(cfg)):
        raise LeakageError("Sealed selection does not match the sealed config")
    manifest = yaml.safe_load((root / "seals/manifest.yaml").read_text(encoding="utf-8"))
    if not any(s.get("tag") == "seal-4" and s.get("commit") == oid for s in manifest["seals"]):
        raise LeakageError("seals/manifest.yaml has no seal-4 entry naming the seal commit")
    errors = check_seals(root)
    if errors:
        raise LeakageError("Seal manifest check failed: " + "; ".join(errors))
    return oid


def check_data(cfg, root=ROOT):
    """Before loading: every day-1 processed file, and every day-2/3 processed file that
    exists, has the SHA-256 recorded in the checksum files when it was written, and both raw
    members of each present user-day have a recorded hash. Returns the day-2/3 user-days
    without a processed file (missing cells)."""
    root, missing = Path(root), []
    for day in (1, 2, 3):
        recorded = read_checksums(root / CHECKSUMS[day])
        for subject in cfg["subjects"]:
            name = f"{cfg['data_dir']}/day{day}_sub{subject:03d}.npz"
            if day > 1 and not (root / name).exists():
                missing.append(dict(subject=subject, day=day))
                continue
            if recorded.get(name) != sha256_file(root / name):
                raise LeakageError(f"Processed file differs from its recorded hash: {name}")
            for kind in KINDS:
                if member_path(subject, day, kind) not in recorded:
                    raise LeakageError(f"No recorded raw hash for {member_path(subject, day, kind)}")
    return missing


# ------------------------------------------------------------------ entry point

def write_selection(cfg, out, dev, streams, root=ROOT):
    selection = dict(dev, config_hash=digest(analysis_config(cfg)),
                     input_sha256=input_hashes(root),
                     thresholds_status=cfg["thresholds_status"], gates=streams["gates"],
                     rr_matched=streams["rr_matched"],
                     m3_power=streams["power"].to_dict(orient="records"),
                     tau_grid=streams["tau_grid"].to_dict(orient="records"),
                     command_streams=streams["streams"].to_dict(orient="records"),
                     within_day_identification=streams["summary"].to_dict(orient="records"))
    write_json(Path(out) / "selection.json", selection)
    return selection


def log_attempt(**fields):
    """Every final attempt (start, failure, completion) in results/exp_m/final_attempts.jsonl."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "final_attempts.jsonl", "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                **fields)) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["dev", "streams", "final"])
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--seal-commit")
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args(argv)
    cfg = load_cfg(args.config)
    if not 1 <= cfg["workers"] <= 6:
        raise ValueError("Shared CPU limit: at most 6 workers")
    if args.stage == "final":
        if args.result_dir is None or args.artifact_dir is None:
            raise LeakageError("final requires explicit --result-dir and --artifact-dir")
        if args.result_dir.resolve() != RESULTS or args.artifact_dir.resolve() != ARTIFACTS:
            raise LeakageError("The real final writes only to results/exp_m and artifacts/exp_m")
        if (RESULTS / "decision.json").exists():
            raise LeakageError("The final has completed; a completed final is not rerun")
        seal = verify_seal(args.seal_commit, cfg)
        post_seal_allowlist(seal)
        log_attempt(event="start", commit=seal)
        try:
            selection = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
            missing = check_data(cfg)
            training = load_trials(ROOT / cfg["data_dir"], cfg["subjects"], 1)
            evaluation = [load_trials(ROOT / cfg["data_dir"], cfg["subjects"], d,
                                      allow_later_days=True, skip_missing=True) for d in (2, 3)]
            evaluation = Trials(np.concatenate([e.X for e in evaluation]),
                                pd.concat([e.meta for e in evaluation], ignore_index=True))
            run_final(training, evaluation, cfg, selection, RESULTS, ARTIFACTS)
        except BaseException as exc:
            log_attempt(event="failed", commit=seal, error=f"{type(exc).__name__}: {exc}")
            raise
        write_json(RESULTS / "seal.json", dict(commit=seal, missing_processed_files=missing))
        log_attempt(event="completed", commit=seal)
        return
    out, art = args.result_dir or RESULTS, args.artifact_dir or ARTIFACTS
    cfg = dict(cfg, _result_dir=str(out), _artifact_dir=str(art))
    with day1_only() as audit:
        signals = load_trials(ROOT / cfg["data_dir"], cfg["subjects"], 1)
        if args.stage == "dev":
            dev = run_dev(signals, cfg, out, art)
            dev_features(signals, cfg, art)
        else:
            dev = json.loads((Path(out) / "dev_decoders.json").read_text(encoding="utf-8"))
            # Stream-level settings may still change before the seal; decoders may not.
            if dev["development_config_hash"] != digest(development_config(cfg)):
                raise ValueError("Decoder configuration changed since the development stage")
            streams = run_streams(signals, cfg, out, art)
            write_selection(cfg, out, dev, streams)
    write_json(Path(out) / f"day_access_{args.stage}.json",
               dict(opened_day1=len(set(audit["opened"])), blocked_later_days=audit["blocked"],
                    later_days_opened=False,
                    guard="CPython open audit hook; day-2/3 names raise LeakageError"))


if __name__ == "__main__":
    from beat.exp_m import main as imported_main
    imported_main()
