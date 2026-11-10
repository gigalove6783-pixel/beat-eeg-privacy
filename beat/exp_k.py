"""Experiment K (EXPLORATORY): user identity in decoded motor-imagery command streams.

    python -m beat.exp_k pilot   # declared grid, timing pilot, runtime projection, drop list
    python -m beat.exp_k reproject  # projection + drop list again from timing_pilot.csv
    python -m beat.exp_k run     # declared grid minus the drop list, then the power analysis
    python -m beat.exp_k cells   # Bayes cells and report again, reusing run's stored battery
    python -m beat.exp_k power   # power analysis only (needs the run's identity curves)

Streams use Experiment C's command rule (beat.exp_c.decide). Calibration-day streams are
cross-fitted: BCI IV 2a T logits are Experiment B's leave-one-run-out outputs
(artifacts/exp_b/dev/oof/<setting>), Lee2019 session-1 logits are L's leave-one-block-out
outputs (artifacts/exp_l/oof_*). In-sample calibration streams (frozen final model applied
to its own training day) are a contrast only. Deployment-day logits are the frozen final
models' outputs (artifacts/exp_b/final, artifacts/exp_l/outputs_*). Everything is
exploratory and conditional on these fitted systems.

Commands are coded as outputs 0 = left, 1 = hold, 2 = right (command + 1); targets are the
cue labels (0 left, 1 right). Weighted utility per trial = P(correct move) - w P(wrong move)
with w = tau / (1 - tau), the wrong-move cost implied by the hold rule (wrong_weight); net
progress = P(correct) - P(wrong) (the w = 1, tau = 0.5 case) is kept as a secondary metric.
EQ = g-indistinguishable equalized decoding: per-user stochastic post-processing matrices M_u
(command -> output) from a linear program (eq_lp) that maximizes mean weighted utility subject
to identical output distributions within each group of users. The guarantee, per variant and
with its assumptions, is stated in eq_lp's docstring (also results/exp_k/eq_guarantee.md).
The calibration-day stream a cross-day attacker enrols on is post-processed by cross-fitted
matrices (crossfit_matrices: no row is equalized by matrices estimated from its own fold);
the in-sample post-processed stream is reported as a labelled lower bound
(scenario cross_day_eq_insample).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed, parallel_config
from scipy.optimize import linprog
from scipy.special import gammaln
from threadpoolctl import threadpool_limits

from .attacks import fit_attackers
from .common import rng_for, t_greater, t_interval, t_less, write_csv, write_json, write_text
from .config import ROOT, load_config
from .data import Trials
from .exp_c import command_descriptors, decide, fit_temperature

RESULTS = ROOT / "results" / "exp_k"
ARTIFACTS = ROOT / "artifacts" / "exp_k"
B_ART = ROOT / "artifacts" / "exp_b"
L_ART = ROOT / "artifacts" / "exp_l"
LABEL = "EXPLORATORY - Experiment K (not confirmatory; conditional on the fitted systems)"
BAYES = ("cat_blind", "cat_aware", "markov_blind", "markov_aware")
BATTERY = ("logistic", "rbf_svm", "mlp")
SCENARIO_LABEL = {"within_day": "within-day (deployment segment A -> B)",
                  "cross_day": "cross-day (cross-fitted calibration day -> deployment day)",
                  "cross_day_insample": "cross-day contrast (in-sample calibration stream)",
                  "cross_day_eq_insample": "cross-day lower bound (EQ matrices fitted on the "
                                           "enrolment stream itself)"}
KEY = ["subject", "session", "run", "trial"]
CHUNK = 512


def load_cfg(path=None):
    return yaml.safe_load(Path(path or ROOT / "configs" / "exp_k.yaml").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ streams

@dataclass(frozen=True)
class Stream:
    """Command stream sorted by subject, segment, recorded order."""
    subject: np.ndarray
    segment: np.ndarray
    target: np.ndarray
    output: np.ndarray

    def __len__(self):
        return len(self.subject)

    def subset(self, mask):
        idx = np.flatnonzero(mask) if np.asarray(mask).dtype == bool else np.asarray(mask)
        return Stream(self.subject[idx], self.segment[idx], self.target[idx], self.output[idx])

    def with_output(self, output):
        return replace(self, output=np.asarray(output, dtype=np.int8))


def first_per_user(stream, n):
    """First n rows (recorded order) of every subject."""
    start = np.r_[0, np.flatnonzero(np.diff(stream.subject)) + 1]
    first = np.repeat(start, np.diff(np.r_[start, len(stream)]))
    return stream.subset(np.arange(len(stream)) - first < n)


def _frame(source, logits):
    frame = pd.DataFrame({c: np.asarray(source[c]) for c in ("subject", "session", "run", "trial",
                                                             "label")})
    frame["session"] = frame.session.astype(str)
    logits = np.asarray(logits, dtype=float)
    frame["logit_left"], frame["logit_right"] = logits[:, 0], logits[:, 1]
    return frame.sort_values(KEY).reset_index(drop=True)


def _temperature(frame, bounds, tolerance):
    meta = frame[["subject", "session", "run", "trial", "label"]]
    return fit_temperature(Trials(frame[["logit_left", "logit_right"]].to_numpy(), meta),
                           bounds, tolerance)


def load_logits(dataset, key, seed):
    """(calibration OOF, calibration in-sample, deployment) logit frames and temperature info."""
    if dataset == "bnci2014_001":
        temp_cfg = load_config("exp_c")["temperature"]
        files = sorted((B_ART / "dev" / "oof" / key).glob(f"seed{seed}_fold*.npz"))
        if len(files) != 6:
            raise FileNotFoundError(f"2a {key} seed {seed}: expected 6 OOF folds")
        parts = []
        for path in files:
            with np.load(path, allow_pickle=False) as z:
                parts.append(_frame(z, z["logits"]))
        oof = pd.concat(parts, ignore_index=True).sort_values(KEY).reset_index(drop=True)
        with np.load(B_ART / "final" / key / f"seed{seed}_outputs.npz", allow_pickle=False) as z:
            final = _frame(z, z["logits"])
        insample = oof[KEY].merge(final[final.session == "T"], on=KEY, how="left",
                                  validate="one_to_one")
        dep = final[final.session == "E"].reset_index(drop=True)
        temperature = _temperature(oof, temp_cfg["bounds"], temp_cfg["tolerance"])
        stored = json.loads((ROOT / "artifacts/exp_c/dev/calibration.json").read_text(
            encoding="utf-8"))["temperatures"].get(key, {}).get(str(seed))
        info = dict(temperature=temperature, stored_temperature=stored,
                    temperature_source="fit_temperature on cross-fitted T logits (exp_c rule)")
    else:
        lcfg = yaml.safe_load((ROOT / "configs/exp_l.yaml").read_text(encoding="utf-8"))
        oof_trials = joblib.load(L_ART / f"oof_{key}_{seed}.joblib")
        out = joblib.load(L_ART / f"outputs_{key}_{seed}.joblib")
        oof = _frame(oof_trials.meta, oof_trials.X)
        finite_t, finite_e = out["T"].finite.to_numpy(bool), out["E"].finite.to_numpy(bool)
        insample = _frame(out["T"][finite_t], out["tl"][finite_t])
        dep = _frame(out["E"][finite_e], out["el"][finite_e])
        stored = json.loads((ROOT / "results/exp_l/selection.json").read_text(
            encoding="utf-8"))["temperatures"][key][str(seed)]
        refit = _temperature(oof, **lcfg["temperature"])
        info = dict(temperature=stored, stored_temperature=stored, refit_temperature=refit,
                    temperature_source="results/exp_l/selection.json (session-1 CV)")
    if not np.isfinite(dep[["logit_left", "logit_right"]].to_numpy()).all():
        raise ValueError(f"{dataset} {key} {seed}: non-finite deployment logits")
    return oof, insample, dep, info


def to_stream(frame, temperature, tau, segment_runs):
    commands = decide(frame[["logit_left", "logit_right"]].to_numpy(), temperature, tau)
    return Stream(frame.subject.to_numpy(np.int64),
                  np.where(frame.run.to_numpy() <= segment_runs, 1, 2).astype(np.int8),
                  frame.label.to_numpy(np.int8), (commands.astype(np.int16) + 1).astype(np.int8))


def build_streams(cfg, datasets=None, conditions=None, seeds=None, write=True):
    """Every command stream of the declared grid plus provenance/consistency rows."""
    streams, rows, records = {}, [], []
    replay = None
    for dataset, dcfg in cfg["datasets"].items():
        if datasets and dataset not in datasets:
            continue
        for condition, key in dcfg["conditions"].items():
            if conditions and condition not in conditions:
                continue
            for seed in seeds or cfg["model_seeds"]:
                oof, insample, dep, info = load_logits(dataset, key, seed)
                for tau in cfg["command"]["taus"]:
                    for source, frame in (("cal_oof", oof), ("cal_insample", insample),
                                          ("dep", dep)):
                        stream = to_stream(frame, info["temperature"], tau, dcfg["segment_runs"])
                        streams[(dataset, condition, seed, tau, source)] = stream
                        cmd = stream.output.astype(int) - 1
                        correct = (cmd == np.where(stream.target == 1, 1, -1))
                        wrong = (cmd == np.where(stream.target == 1, -1, 1))
                        rows.append(dict(
                            dataset=dataset, condition=condition, setting_key=key, seed=seed,
                            tau=tau, source=source, trials=len(stream),
                            users=len(np.unique(stream.subject)),
                            hold_rate=float((cmd == 0).mean()), correct_rate=float(correct.mean()),
                            wrong_rate=float(wrong.mean()),
                            net_progress=float(correct.mean() - wrong.mean()), **info))
                        if write:
                            records.append(frame[["subject", "session", "run", "trial"]].assign(
                                dataset=dataset, condition=condition, seed=seed, tau=tau,
                                source=source, segment=stream.segment, target=stream.target,
                                command=cmd.astype(np.int8)))
                    if dataset == "bnci2014_001" and condition in ("P0", "P2-RC_d8"):
                        if replay is None:
                            replay = pd.read_csv(
                                ROOT / "artifacts/exp_c/final/replay_trials.csv.gz",
                                usecols=["subject", "session", "run", "trial", "tau", "mode",
                                         "command", "condition", "seed"])
                            replay = replay[replay["mode"] == "command"]
                        part = replay[(replay.condition == key) & (replay.seed == seed)
                                      & np.isclose(replay.tau, tau)]
                        mine = dep[KEY].assign(mine=streams[(dataset, condition, seed, tau,
                                                             "dep")].output.astype(int) - 1)
                        merged = mine.merge(part.astype({"session": str}), on=KEY, how="inner")
                        rows[-1]["replay_compared"] = len(merged)
                        rows[-1]["replay_mismatches"] = int((merged.mine != merged.command).sum())
    if write:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        write_csv(pd.concat(records, ignore_index=True), ARTIFACTS / "command_records.csv.gz")
    return streams, pd.DataFrame(rows)


# ------------------------------------------------------------------ windows and counts

def _bounds(stream):
    key = stream.subject.astype(np.int64) * 16 + stream.segment
    cut = np.flatnonzero(np.diff(key)) + 1
    return np.r_[0, cut], np.r_[cut, len(key)]


def window_starts(stream, k, stride=None):
    """Starts of k consecutive rows within one (subject, segment); test windows do not overlap."""
    stride = stride or k
    a, b = _bounds(stream)
    parts = [np.arange(s, e - k + 1, stride) for s, e in zip(a, b)]
    return np.concatenate(parts).astype(np.int64) if parts else np.empty(0, np.int64)


def _onehot(index, size):
    out = np.zeros((len(index), size))
    out[np.arange(len(index)), index] = 1
    return out


def _cum(values):
    return np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])


class Counts:
    """Cumulative count tables for O(1) window counts.

    cat (1x3): outputs; aware (2x3): outputs by target; trans (3x3): previous -> current
    output; trans_aware (6x3): (previous output, current target) -> current output.
    Transitions never cross a segment boundary.
    """

    def __init__(self, stream):
        o, y = stream.output.astype(np.int64), stream.target.astype(np.int64)
        start = np.zeros(len(stream), dtype=bool)
        start[_bounds(stream)[0]] = True
        prev = np.r_[0, o[:-1]]
        tb, ta = _onehot(prev * 3 + o, 9), _onehot((prev * 2 + y) * 3 + o, 18)
        tb[start], ta[start] = 0, 0
        self.stream = stream
        self.tables = dict(cat=(_cum(_onehot(o, 3)), (1, 3)), aware=(_cum(_onehot(y * 3 + o, 6)), (2, 3)),
                           trans=(_cum(tb), (3, 3)), trans_aware=(_cum(ta), (6, 3)))

    def windows(self, starts, k):
        end = starts + k
        out = {}
        for name, (cum, shape) in self.tables.items():
            lo = starts + 1 if name.startswith("trans") else starts
            out[name] = (cum[end] - cum[lo]).reshape(-1, *shape)
        o, y = self.stream.output[starts].astype(np.int64), self.stream.target[starts].astype(np.int64)
        out["first_cat"] = _onehot(o, 3).reshape(-1, 1, 3)
        out["first_aware"] = _onehot(y * 3 + o, 6).reshape(-1, 2, 3)
        return out

    def enrol(self, users):
        idx = np.searchsorted(users, self.stream.subject)
        out = {}
        for name, (cum, shape) in self.tables.items():
            total = np.zeros((len(users), cum.shape[1]))
            np.add.at(total, idx, np.diff(cum, axis=0))
            out[name] = total.reshape(len(users), *shape)
        return out


# ------------------------------------------------------------------ Bayes attackers

def dirmult(m, n, alpha):
    """log Dirichlet-multinomial predictive of each window table m (W,R,C) under every user's
    posterior Dir(n + alpha) (U,R,C), up to user-independent constants -> (W, U)."""
    flat = m.reshape(len(m), -1)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    mu = unique.reshape(len(unique), *m.shape[1:])
    a = n + alpha
    total = a.sum(-1)
    base = gammaln(a)
    out = np.empty((len(unique), len(n)))
    for s in range(0, len(unique), CHUNK):
        part = mu[s:s + CHUNK]
        value = (gammaln(a[None] + part[:, None]) - base[None]).sum(axis=(2, 3))
        value += (gammaln(total)[None] - gammaln(total[None] + part.sum(-1)[:, None])).sum(-1)
        out[s:s + CHUNK] = value
    return out[np.asarray(inverse).reshape(-1)]


def bayes_scores(attacker, win, enrol, alpha):
    """Log marginal likelihood of each test window under each enrolled user (uniform prior).

    cat_*: exchangeable Dirichlet-categorical over outputs (blind) or outputs given targets
    (aware); markov_*: first output categorical, then first-order transitions (aware: given
    the current target as well).
    """
    if attacker == "cat_blind":
        return dirmult(win["cat"], enrol["cat"], alpha)
    if attacker == "cat_aware":
        return dirmult(win["aware"], enrol["aware"], alpha)
    if attacker == "markov_blind":
        return dirmult(win["first_cat"], enrol["cat"], alpha) + dirmult(
            win["trans"], enrol["trans"], alpha)
    if attacker == "markov_aware":
        return dirmult(win["first_aware"], enrol["aware"], alpha) + dirmult(
            win["trans_aware"], enrol["trans_aware"], alpha)
    raise ValueError(attacker)


def argmax_random(scores, rng):
    """Row-wise argmax with uniformly random tie-breaking (-inf entries never win)."""
    top = scores.max(axis=1, keepdims=True)
    tie = scores >= top - 1e-9 * np.maximum(1.0, np.abs(top))
    priority = np.where(tie, rng.random(scores.shape), -1.0)
    return priority.argmax(axis=1)


def per_user(correct, truth, users):
    idx = np.searchsorted(users, truth)
    num = np.bincount(idx, weights=np.asarray(correct, float), minlength=len(users))
    den = np.bincount(idx, minlength=len(users))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def recalls(scores, users, truth, groupings, rng, predicted=None):
    """Per-user recall: N-way (argmax over all users) and within-group (argmax over the
    true user's group; the group is given to the attacker)."""
    pred = predicted if predicted is not None else users[argmax_random(scores, rng)]
    out = {"nway": per_user(pred == truth, truth, users)}
    t_idx = np.searchsorted(users, truth)
    for label, group in groupings.items():
        masked = np.where(group[None, :] == group[t_idx][:, None], scores, -np.inf)
        out[label] = per_user(users[argmax_random(masked, rng)] == truth, truth, users)
    return out


def chance(users, groupings):
    out = {"nway": np.full(len(users), 1.0 / len(users))}
    for label, group in groupings.items():
        sizes = np.bincount(group)
        out[label] = 1.0 / sizes[group]
    return out


# ------------------------------------------------------------------ attack battery

def battery_features(stream, k, stride, view):
    """C's 13 command descriptors (blind) plus 6 per-target output frequencies (aware)."""
    starts = window_starts(stream, k, stride)
    groups = starts[:, None] + np.arange(k)[None, :]
    X, _ = command_descriptors(stream.output.astype(int) - 1, groups)
    if view == "aware":
        aware = Counts(stream).windows(starts, k)["aware"]
        X = np.hstack([X, (aware / np.maximum(aware.sum(-1, keepdims=True), 1)).reshape(len(X), 6)])
    return X, stream.subject[starts]


def battery_job(spec, enrol, test, users, groupings, dataset):
    """Existing battery (logistic, RBF SVM, MLP) on one enrolment/test pair, one k and view.

    2a uses configs/exp_b.yaml attackers (beat.attacks.fit_attackers, inner-CV SVM grid);
    L uses configs/exp_l.yaml (beat.exp_l._fit_attack, fixed SVM) exactly as sealed.
    N-way predictions are the estimators' own predict; within-group predictions restrict
    predict_proba (logistic, MLP) or decision_function (SVM) to the true user's group.
    """
    start = time.perf_counter()
    rng = rng_for(spec["root_seed"], "battery_tie", *[str(spec[c]) for c in (
        "dataset", "condition", "seed", "tau", "defence", "pp_seed", "scenario", "direction",
        "k", "view")])
    out, diagnostics = [], []
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A, ya = battery_features(enrol, spec["k"], 1, spec["view"])
        B, yb = battery_features(test, spec["k"], spec["k"], spec["view"])
        try:
            if not len(B):
                raise ValueError("no test windows")
            if dataset == "lee2019":
                from .exp_l import _fit_attack
                lcfg = yaml.safe_load((ROOT / "configs/exp_l.yaml").read_text(encoding="utf-8"))
                models = _fit_attack(A, ya, lcfg, spec["seed"])
            else:
                models = fit_attackers(A, ya, load_config("exp_b")["attackers"], spec["seed"])
        except ValueError as exc:
            return dict(spec=spec, rows=[], status="not_evaluable", reason=str(exc),
                        train_windows=len(A), test_windows=len(B),
                        seconds=time.perf_counter() - start, diagnostics=[])
        for name in BATTERY:
            model = models[name]
            classes = np.asarray(model.classes_)
            if not np.array_equal(classes, users):
                raise ValueError("battery classes differ from the enrolled users")
            scores = model.decision_function(B) if name == "rbf_svm" else model.predict_proba(B)
            rec = recalls(scores, users, yb, groupings, rng, predicted=model.predict(B))
            for metric, values in rec.items():
                out.append(dict(attacker=f"{name}_{spec['view']}", metric=metric, recall=values))
            if name == "mlp":
                diagnostics.append(dict(**getattr(model, "convergence_", {})))
    return dict(spec=spec, rows=out, status="ok", reason="", train_windows=len(A),
                test_windows=len(B), seconds=time.perf_counter() - start, diagnostics=diagnostics)


# ------------------------------------------------------------------ defences

def confusion_counts(stream, users):
    counts = np.zeros((len(users), 2, 3))
    np.add.at(counts, (np.searchsorted(users, stream.subject), stream.target.astype(int),
                       stream.output.astype(int)), 1)
    return counts


def net_progress_from(joint):
    """joint (U,2,3) = P(target, output) per user -> (correct, wrong, hold, net)."""
    correct = joint[:, 0, 0] + joint[:, 1, 2]
    wrong = joint[:, 0, 2] + joint[:, 1, 0]
    return correct, wrong, joint[:, :, 1].sum(1), correct - wrong


def wrong_weight(tau):
    """Wrong-move cost w implied by the hold rule: a move is issued when the calibrated
    maximum probability p reaches tau, and the move's expected utility p - w (1 - p) is zero
    exactly at p = tau for w = tau / (1 - tau) (w = 1 at tau = 0.5: net progress)."""
    return float(tau) / (1.0 - float(tau))


def make_groups(users, utility, size):
    """Users sorted by calibration net progress (ties by id) in consecutive chunks of `size`;
    leftover users join the last group, so group sizes are size..2*size-1."""
    n = len(users)
    order = np.lexsort((users, utility))
    count = max(1, n // size)
    group = np.empty(n, dtype=np.int64)
    group[order] = np.minimum(np.arange(n) // size, count - 1)
    return group


def g_labels(cfg, n_users):
    out = {}
    for g in cfg["defences"]["eq_g"]:
        size = n_users if g == "all" else int(g)
        if size > n_users:
            continue
        out.setdefault("gall" if size >= n_users else f"g{size}", min(size, n_users))
    return out


def defence_list(cfg, labels):
    rr = [f"rr_q{q:g}" for q in cfg["defences"]["rr_q"]]
    eq = [f"eq{variant[0]}_{label}" for variant in cfg["defences"]["eq_variants"]
          for label in labels]
    return ["none"] + rr + eq


def rr_matrix(q):
    return (1 - q) * np.eye(3) + q / 3


def rr_epsilon(q):
    """Per-command local differential privacy level of k-ary RR over 3 outputs."""
    return float(np.log((1 - q + q / 3) / (q / 3)))


def eq_lp(confusion, prior, variant, wrong_weight=1.0):
    """Utility-optimal equalizing post-processing for one group.

    confusion (n, 2, 3) = P(command | target, user); prior (2,) = target distribution;
    wrong_weight w = cost of a wrong move relative to a correct one (w = tau / (1 - tau),
    see wrong_weight(); w = 1 gives net progress). Variables: M_u (3 x 3, rows = command,
    columns = output, row-stochastic) per user and the shared distribution (marginal:
    P(output), 3; confusion: P(output | target), 2 x 3). Maximizes the mean over users of the
    weighted utility sum_y prior_y sum_c C_u(y,c) [M_u(c, toward y) - w M_u(c, away from y)]
    subject to: marginal  sum_c p_u(c) M_u(c, o) = q(o) for all u, o (p_u = prior @ C_u);
                confusion sum_c C_u(y, c) M_u(c, o) = Q(y, o) for all u, y, o.
    "Always hold" is feasible, so the LP is always feasible. Returns M (n,3,3), shared,
    optimum and an optimality certificate (strong-duality gap, dual infeasibility).

    Guarantee (exact for the confusion matrices C_u given to the LP), per variant:
    - EQ-confusion: every user of a group has the same P(output | target) = Q(y, .). If each
      trial is post-processed independently, the target sequence does not depend on the
      user, and commands are conditionally independent across trials given the targets and
      the user with a fixed C_u, then all users of a group have the same distribution of
      output sequences given the targets. Every attacker - target-aware or target-blind,
      any number of commands k - then identifies a user within the group at exactly chance
      1/|group|.
    - EQ-marginal: every user of a group has the same P(output) = q under the assumed target
      prior. It covers target-blind attackers only, and only if targets are i.i.d. with
      that prior and independent of the user (sequences additionally need the same
      conditional independence). A target-aware attacker still sees P(output | target),
      which may differ between users of a group.
    - Group membership is revealed: an attacker can still learn the group, which carries at
      most log2(number of groups) <= log2(N / g) bits; EQ bounds identification within a
      group only.
    - The equality holds for the estimates the LP receives (smoothed calibration counts).
      Finite calibration data (estimation error) and any calibration -> deployment change
      (day-to-day drift, decoder mismatch, sequential dependence, phase shifts) leave
      residual leakage; residual_rows measures it (results/exp_k/residual_leakage.csv).
    - Optimality: among per-trial post-processing channels of the decoded command that
      achieve the equality for these estimates, M attains the maximal mean weighted utility
      (certified by the returned duality gap and dual feasibility).
    """
    n = len(confusion)
    prior = np.asarray(prior, dtype=float)
    shared = 3 if variant == "marginal" else 6
    nv = 9 * n + shared
    c = np.zeros(nv)
    for u in range(n):
        for y, (good, bad) in enumerate(((0, 2), (2, 0))):
            for cmd in range(3):
                weight = prior[y] * confusion[u, y, cmd] / n
                c[9 * u + 3 * cmd + good] -= weight
                c[9 * u + 3 * cmd + bad] += wrong_weight * weight
    rows, b = [], []
    for u in range(n):
        for cmd in range(3):
            row = np.zeros(nv)
            row[9 * u + 3 * cmd:9 * u + 3 * cmd + 3] = 1
            rows.append(row)
            b.append(1.0)
    for u in range(n):
        if variant == "marginal":
            p = prior @ confusion[u]
            for o in range(3):
                row = np.zeros(nv)
                row[9 * u + 3 * np.arange(3) + o] = p
                row[9 * n + o] = -1
                rows.append(row)
                b.append(0.0)
        else:
            for y in range(2):
                for o in range(3):
                    row = np.zeros(nv)
                    row[9 * u + 3 * np.arange(3) + o] = confusion[u, y]
                    row[9 * n + 3 * y + o] = -1
                    rows.append(row)
                    b.append(0.0)
    A, b = np.asarray(rows), np.asarray(b)
    res = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(f"EQ LP failed: {res.message}")
    x = np.clip(res.x, 0, None)
    M = x[:9 * n].reshape(n, 3, 3)
    M /= M.sum(-1, keepdims=True)
    dual = np.asarray(res.eqlin.marginals)
    reduced = c - A.T @ dual
    certificate = dict(objective=float(-res.fun), dual_objective=float(-(b @ dual)),
                       duality_gap=float(abs(c @ res.x - b @ dual)),
                       dual_infeasibility=float(max(0.0, -reduced.min())),
                       complementary_slackness=float(np.max(np.abs(res.x * reduced))))
    return M, x[9 * n:], float(-res.fun), certificate


def achieved(confusion, M, prior, variant):
    """Output distribution of each user after post-processing (n,3) or (n,2,3)."""
    joint = np.einsum("uyc,uco->uyo", confusion, M)
    return np.einsum("y,uyo->uo", np.asarray(prior), joint) if variant == "marginal" else joint


def fit_eq(counts, groups, variant, prior, smoothing, wrong_weight=1.0):
    """Per-group EQ LP on smoothed calibration confusion estimates."""
    smoothed = counts + smoothing
    confusion = smoothed / smoothed.sum(-1, keepdims=True)
    M = np.zeros((len(counts), 3, 3))
    rows = []
    for gid in np.unique(groups):
        members = np.flatnonzero(groups == gid)
        Mg, shared, optimum, cert = eq_lp(confusion[members], prior, variant, wrong_weight)
        M[members] = Mg
        dist = achieved(confusion[members], Mg, prior, variant)
        correct, wrong = net_progress_from(np.einsum("y,uyc->uyc", np.asarray(prior),
                                                     confusion[members]))[:2]
        rows.append(dict(group=int(gid), size=len(members), wrong_weight=wrong_weight,
                         lp_objective=optimum,
                         undefended_objective=float((correct - wrong_weight * wrong).mean()),
                         equalization_residual=float(np.max(np.abs(dist - dist[0]))), **cert))
    return M, rows


def eq_spec(defence):
    """EQ defence name -> (LP variant, grouping label)."""
    return ("marginal" if defence.startswith("eqm") else "confusion"), defence.split("_", 1)[1]


def channel(defence, cal_counts, groupings, cfg, wrong_weight=1.0):
    n = len(cal_counts)
    if defence == "none":
        return np.repeat(np.eye(3)[None], n, axis=0), []
    if defence.startswith("rr_q"):
        return np.repeat(rr_matrix(float(defence[4:]))[None], n, axis=0), []
    variant, label = eq_spec(defence)
    return fit_eq(cal_counts, groupings[label], variant, cfg["defences"]["target_prior"],
                  cfg["defences"]["eq_smoothing"], wrong_weight)


def _draw(stream, probs, rng):
    draw = rng.random(len(stream))
    return stream.with_output((draw[:, None] > np.cumsum(probs, axis=1)[:, :2]).sum(axis=1))


def apply_channel(stream, M, users, rng):
    return _draw(stream, M[np.searchsorted(users, stream.subject), stream.output.astype(int)], rng)


def crossfit_folds(stream, folds):
    """Fold id per row: `folds` contiguous blocks of each user's recorded order."""
    start = np.r_[0, np.flatnonzero(np.diff(stream.subject)) + 1]
    size = np.diff(np.r_[start, len(stream)])
    position = np.arange(len(stream)) - np.repeat(start, size)
    return position * folds // np.repeat(size, size)


def crossfit_matrices(stream, users, groups, variant, prior, smoothing, wrong_weight, folds):
    """EQ matrices for each fold fitted on the stream's other folds -> (fold id per row,
    matrices (folds, U, 3, 3)); apply_crossfit post-processes each row with its fold's
    matrices, so no enrolment row is equalized by matrices estimated from itself."""
    fold = crossfit_folds(stream, folds)
    matrices = [fit_eq(confusion_counts(stream.subset(fold != f), users), groups, variant,
                       prior, smoothing, wrong_weight)[0] for f in range(folds)]
    return fold, np.stack(matrices)


def apply_crossfit(stream, fold, matrices, users, rng):
    idx = np.searchsorted(users, stream.subject)
    return _draw(stream, matrices[fold, idx, stream.output.astype(int)], rng)


# ------------------------------------------------------------------ information accounting

def mutual_information(joint):
    joint = joint / joint.sum()
    pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (pa @ pb)[nz])).sum())


def task_information(counts, M, wrong_weight=1.0):
    """Per-user expected (target, output) joint on a day and its task statistics.

    counts (U,2,3) are the undefended (target, command) counts of that day; the output joint is
    the counts pushed through M. utility = correct - wrong_weight * wrong. Bits = plug-in
    I(Y;O) minus the Miller-Madow bias (|Y|-1)(|O|-1) / (2 n ln 2)."""
    joint = np.einsum("uyc,uco->uyo", counts / counts.sum((1, 2), keepdims=True), M)
    correct, wrong, hold, net = net_progress_from(joint)
    n = counts.sum((1, 2))
    bits = np.array([mutual_information(j) for j in joint])
    active = (joint.sum(1) > 1e-12).sum(1)
    bits = bits - (active - 1) / (2 * n * np.log(2))
    return dict(correct=correct, wrong=wrong, hold=hold, net=net,
                utility=correct - wrong_weight * wrong, bits=bits)


def _conditional_mi(u, o, strata, n_users, n_strata):
    counts = np.bincount((strata * n_users + u) * 3 + o,
                         minlength=n_strata * n_users * 3).reshape(n_strata, n_users, 3)
    total = counts.sum()
    value = 0.0
    for s in range(n_strata):
        block = counts[s]
        if block.sum() > 0:
            value += block.sum() / total * mutual_information(block.astype(float))
    return value


def identity_information(stream, users, view, group=None, rng=None, permutations=20):
    """Permutation-corrected plug-in identity information per trial, in bits.

    blind: I(U;O); aware: I(U;O|Y); markov: I(U;O_t|O_{t-1}) within segments. With a group
    partition, the information is conditional on the (revealed) group. The bias is removed by
    subtracting the mean plug-in value after permuting user labels within strata."""
    u = np.searchsorted(users, stream.subject)
    o = stream.output.astype(np.int64)
    keep = np.ones(len(stream), dtype=bool)
    if view == "blind":
        z, nz = np.zeros(len(stream), np.int64), 1
    elif view == "aware":
        z, nz = stream.target.astype(np.int64), 2
    else:
        z, nz = np.r_[0, o[:-1]], 3
        keep[_bounds(stream)[0]] = False
    g = np.zeros(len(users), np.int64) if group is None else np.asarray(group)
    strata = g[u] * nz + z
    n_strata = int(g.max() + 1) * nz
    u, o, strata = u[keep], o[keep], strata[keep]
    plug = _conditional_mi(u, o, strata, len(users), n_strata)
    rng = rng or np.random.default_rng(0)
    order = np.argsort(strata, kind="stable")
    nulls = []
    for _ in range(permutations):
        shuffled = order[np.lexsort((rng.random(len(order)), strata[order]))]
        perm = np.empty_like(u)
        perm[order] = u[shuffled]
        nulls.append(_conditional_mi(perm, o, strata, len(users), n_strata))
    null = float(np.mean(nulls))
    return dict(bits=plug - null, plugin=plug, null=null)


def fano_k(bits, n_users, accuracy):
    """Fano lower bound on commands needed for identification accuracy A among n users,
    assuming conditionally i.i.d. commands with per-trial identity information `bits`:
    k >= (log2 N - h(1-A) - (1-A) log2(N-1)) / bits."""
    pe = 1 - accuracy
    h = 0.0 if pe <= 0 or pe >= 1 else -(pe * np.log2(pe) + (1 - pe) * np.log2(1 - pe))
    need = np.log2(n_users) - h - pe * np.log2(n_users - 1)
    if need <= 0:
        return 0.0
    return float(need / bits) if bits > 0 else float("inf")


# ------------------------------------------------------------------ Bayes evaluation

def scenario_pairs(sources, enrolment_sizes):
    dep = sources["dep"]
    seg = {s: dep.subset(dep.segment == s) for s in (1, 2)}
    for size in ["all"] + [str(s) for s in enrolment_sizes]:
        for a, b in ((1, 2), (2, 1)):
            enrol = seg[a] if size == "all" else first_per_user(seg[a], int(size))
            yield "within_day", size, f"{a}to{b}", enrol, seg[b]
    yield "cross_day", "all", "cal_to_dep", sources["cal_oof"], dep
    if "cal_insample" in sources:
        yield "cross_day_insample", "all", "cal_to_dep", sources["cal_insample"], dep
    if "cal_eq_insample" in sources:
        yield "cross_day_eq_insample", "all", "cal_to_dep", sources["cal_eq_insample"], dep


def bayes_evaluate(sources, users, groupings, ks, cfg, rng, k_sweep):
    """{(scenario, enrolment, attacker, k, metric): per-user recall averaged over directions}."""
    alpha = cfg["attackers"]["dirichlet_alpha"]
    acc = {}
    cache = {}
    for scenario, size, _, enrol, test in scenario_pairs(
            sources, cfg["attackers"]["enrolment_sizes"]):
        enrolled = Counts(enrol).enrol(users)
        key = id(test)
        if key not in cache:
            cache[key] = Counts(test)
        counts = cache[key]
        for k in (ks if size == "all" else k_sweep):
            starts = window_starts(test, k)
            if not len(starts):
                continue
            win = counts.windows(starts, k)
            truth = test.subject[starts]
            for attacker in cfg["attackers"]["bayes"]:
                scores = bayes_scores(attacker, win, enrolled, alpha)
                for metric, values in recalls(scores, users, truth, groupings, rng).items():
                    acc.setdefault((scenario, size, attacker, k, metric), []).append(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {key: np.nanmean(values, axis=0) for key, values in acc.items()}


def _synthetic(stream, confusion, rng):
    """Same targets/structure, commands redrawn i.i.d. from P(command | target, user)."""
    users = np.unique(stream.subject)
    probs = confusion[np.searchsorted(users, stream.subject), stream.target.astype(int)]
    draw = rng.random(len(stream))
    return stream.with_output((draw[:, None] > np.cumsum(probs, axis=1)[:, :2]).sum(axis=1))


def residual_rows(cal, dep, users, groupings, defence, cfg, rng, ks, wrong_weight=1.0):
    """Residual-leakage decomposition for one EQ cell (matrices from the weighted-utility LP).

    Components (one row each; residual_summary averages replicates and takes differences):
    total      : matrices fitted on the cross-fitted calibration day, measured on the real
                 deployment day;
    estimation : no-shift simulation - truth = the deployment day's empirical P(c | y, u);
                 calibration and deployment streams are redrawn i.i.d. from it with the
                 real target sequences, matrices fitted on the synthetic calibration day
                 (finite calibration data is the only error source);
    floor      : matrices fitted on the truth itself, measured on the same synthetic
                 deployment stream - the measurement floor of the finite-sample leakage
                 estimators (no estimation error);
    split      : real within-day split - fitted on deployment segment A, measured on B.
    Derived columns (residual_summary):
    total_minus_estimation = total - estimation: everything the i.i.d. same-day simulation
                 leaves out - day-to-day drift, mismatch between the cross-fitted calibration
                 decoder outputs and the final model's deployment outputs, sequential
                 dependence within the deployment day and, for Lee2019, the offline -> online
                 phase shift between the deployment segments used by the within-day attacker;
                 it is not drift alone;
    estimation_minus_floor = estimation - floor: leakage attributable to finite calibration
                 data beyond the estimator floor.
    Measures: permutation-corrected identity bits per trial given the group (blind, aware),
    the strongest within-day Bayes within-group excess at the reference k values, and the
    deployment weighted utility of the fitted matrices (utility_weighted).
    """
    variant, label = eq_spec(defence)
    group = groupings[label]
    prior, smoothing = cfg["defences"]["target_prior"], cfg["defences"]["eq_smoothing"]
    sub = {label: group}
    perms = cfg["statistics"]["permutations"]

    def utility(stream, M):
        return float(task_information(confusion_counts(stream, users), M,
                                      wrong_weight)["utility"].mean())

    def measure(cal_s, dep_s, M):
        dd = apply_channel(dep_s, M, users, rng)
        dc = apply_channel(cal_s, M, users, rng)
        row = {f"bits_{v}": identity_information(dd, users, v, group, rng, perms)["bits"]
               for v in ("blind", "aware")}
        row["utility_weighted"] = utility(dep_s, M)
        rec = bayes_evaluate(dict(cal_oof=dc, dep=dd), users, sub, ks, dict(
            cfg, attackers=dict(cfg["attackers"], enrolment_sizes=[])), rng, [])
        ch = 1.0 / np.bincount(group)[group]
        for k in ks:
            row[f"excess_k{k}"] = max(float(np.nanmean(rec[("within_day", "all", a, k, label)] - ch))
                                      for a in cfg["attackers"]["bayes"]
                                      if ("within_day", "all", a, k, label) in rec)
        return row

    truth = confusion_counts(dep, users)
    truth_p = truth / np.maximum(truth.sum(-1, keepdims=True), 1)
    M_total, _ = fit_eq(confusion_counts(cal, users), group, variant, prior, smoothing,
                        wrong_weight)
    rows = [dict(component="total", replicate=0, **measure(cal, dep, M_total))]
    for r in range(cfg["statistics"]["residual_replicates"]):
        scal, sdep = _synthetic(cal, truth_p, rng), _synthetic(dep, truth_p, rng)
        M_est, _ = fit_eq(confusion_counts(scal, users), group, variant, prior, smoothing,
                          wrong_weight)
        rows.append(dict(component="estimation", replicate=r, **measure(scal, sdep, M_est)))
        M_floor, _ = fit_eq(truth, group, variant, prior, smoothing, wrong_weight)
        rows.append(dict(component="floor", replicate=r, **measure(scal, sdep, M_floor)))
    for a, b in ((1, 2), (2, 1)):
        seg_a, seg_b = dep.subset(dep.segment == a), dep.subset(dep.segment == b)
        M_split, _ = fit_eq(confusion_counts(seg_a, users), group, variant, prior, smoothing,
                            wrong_weight)
        applied = apply_channel(seg_b, M_split, users, rng)
        rows.append(dict(component="split", replicate=a, utility_weighted=utility(seg_b, M_split),
                         **{f"bits_{v}": identity_information(applied, users, v, group, rng,
                                                              perms)["bits"]
                            for v in ("blind", "aware")}))
    return rows


def cell_job(cell, sources, cfg, residual=True):
    """Bayes attacks, utility, information and residual leakage for one
    (dataset, condition, model seed, tau) cell, over every defence and post-processing seed."""
    dataset, condition, seed, tau = cell
    start = time.perf_counter()
    dcfg = cfg["datasets"][dataset]
    ks = dcfg["k"]
    k_sweep = sorted({1, cfg["statistics"]["k_reference"], max(ks)})
    cal, dep = sources["cal_oof"], sources["dep"]
    users = np.unique(dep.subject)
    cal_counts, dep_counts = confusion_counts(cal, users), confusion_counts(dep, users)
    prior = np.asarray(cfg["defences"]["target_prior"])
    util_cal = net_progress_from(np.einsum("y,uyc->uyc", prior, cal_counts / np.maximum(
        cal_counts.sum(-1, keepdims=True), 1)))[3]
    labels = g_labels(cfg, len(users))
    groupings = {label: make_groups(users, util_cal, size) for label, size in labels.items()}
    chances = chance(users, groupings)
    defences = (["none"] if condition in cfg["undefended_only"] else defence_list(cfg, labels))
    recall_rows, utility_rows, info_rows, fit_rows, residual_out, timing = [], [], [], [], [], {}
    perms = cfg["statistics"]["permutations"]
    w = wrong_weight(tau)
    with threadpool_limits(limits=1):
        for defence in defences:
            t0 = time.perf_counter()
            M, fits = channel(defence, cal_counts, groupings, cfg, w)
            fit_rows.extend(dict(defence=defence, **row) for row in fits)
            if defence.startswith("eq"):
                variant, label = eq_spec(defence)
                fold, fold_M = crossfit_matrices(cal, users, groupings[label], variant, prior,
                                                 cfg["defences"]["eq_smoothing"], w,
                                                 cfg["defences"]["eq_crossfit_folds"])
            pp_seeds = [0] if defence == "none" else cfg["defences"]["postprocessing_seeds"]
            acc, info = {}, {}
            for pp in pp_seeds:
                rng = rng_for(cfg["seed"], dataset, condition, seed, str(tau), defence, pp)
                if defence == "none":
                    srcs = dict(cal_oof=cal, dep=dep, cal_insample=sources["cal_insample"])
                elif defence.startswith("eq"):
                    srcs = dict(cal_oof=apply_crossfit(cal, fold, fold_M, users, rng),
                                dep=apply_channel(dep, M, users, rng))
                    srcs["cal_eq_insample"] = apply_channel(cal, M, users, rng)
                else:
                    srcs = dict(cal_oof=apply_channel(cal, M, users, rng))
                    srcs["dep"] = apply_channel(dep, M, users, rng)
                ddep = srcs["dep"]
                tie = rng_for(cfg["seed"], "tie", dataset, condition, seed, str(tau), defence, pp)
                for key, values in bayes_evaluate(srcs, users, groupings, ks, cfg, tie,
                                                  k_sweep).items():
                    acc.setdefault(key, []).append(values)
                irng = rng_for(cfg["seed"], "info", dataset, condition, seed, str(tau), defence, pp)
                for view in ("blind", "aware", "markov"):
                    for label in ["nway"] + list(groupings):
                        value = identity_information(ddep, users, view,
                                                     None if label == "nway" else groupings[label],
                                                     irng, perms)
                        info.setdefault((view, label), []).append(value)
            for (scenario, size, attacker, k, metric), values in acc.items():
                recall_rows.append(dict(scenario=scenario, enrolment=size, family="bayes",
                                        attacker=attacker, k=k, metric=metric, defence=defence,
                                        recall=np.nanmean(values, axis=0), chance=chances[metric]))
            for (view, label), values in info.items():
                info_rows.append(dict(defence=defence, view=view, grouping=label,
                                      bits=float(np.mean([v["bits"] for v in values])),
                                      plugin=float(np.mean([v["plugin"] for v in values])),
                                      null=float(np.mean([v["null"] for v in values]))))
            for day, counts in (("cal", cal_counts), ("dep", dep_counts)):
                stats = task_information(counts, M, w)
                utility_rows.append(dict(defence=day and defence, day=day,
                                         **{name: values for name, values in stats.items()}))
            timing[defence] = time.perf_counter() - t0
        if residual:
            for defence in defences:
                if defence.startswith("eq"):
                    rrng = rng_for(cfg["seed"], "residual", dataset, condition, seed, str(tau),
                                   defence)
                    rows = residual_rows(cal, dep, users, groupings, defence, cfg, rrng,
                                         [cfg["statistics"]["k_reference"], max(ks)], w)
                    residual_out.extend(dict(defence=defence, **row) for row in rows)
    return dict(cell=cell, users=users, groupings=groupings, recall=recall_rows,
                utility=utility_rows, information=info_rows, eq_fits=fit_rows,
                residual=residual_out, timing=timing, seconds=time.perf_counter() - start)


# ------------------------------------------------------------------ grid, jobs, projection

def cells(cfg):
    return [(dataset, condition, seed, tau) for dataset, dcfg in cfg["datasets"].items()
            for condition in dcfg["conditions"] for seed in cfg["model_seeds"]
            for tau in cfg["command"]["taus"]]


def battery_specs(cfg, drops=()):
    """Every battery job of the declared grid, tagged with the cut that would remove it."""
    specs = []
    op = cfg["operating_points"]
    for dataset, condition, seed, tau in cells(cfg):
        dcfg = cfg["datasets"][dataset]
        pairs = [("within_day", "1to2"), ("within_day", "2to1"), ("cross_day", "cal_to_dep"),
                 ("cross_day_insample", "cal_to_dep")]
        defences = [("none", 0)]
        if condition in op["conditions"]:
            defences += [(d, pp) for d in op["defences"] for pp in op["postprocessing_seeds"]]
        for defence, pp in defences:
            for scenario, direction in pairs:
                if defence != "none" and scenario == "cross_day_insample":
                    continue
                for k in dcfg["battery_k"]:
                    for view in cfg["attackers"]["views"]:
                        cuts = []
                        if defence != "none":
                            cuts.append("battery_operating_points")
                        if scenario == "cross_day_insample":
                            cuts.append("battery_insample_contrast")
                        if condition in cfg["undefended_only"]:
                            cuts.append("battery_P3_lambda1")
                        if view == "aware" and k >= 20:
                            cuts.append("battery_aware_view_k_ge_20")
                        if seed != cfg["model_seeds"][0]:
                            cuts.append("battery_seeds_1_2")
                        if any(c in drops for c in cuts):
                            continue
                        specs.append(dict(root_seed=cfg["seed"], dataset=dataset,
                                          condition=condition, seed=seed, tau=tau,
                                          defence=defence, pp_seed=pp, scenario=scenario,
                                          direction=direction, k=k, view=view, cuts=cuts))
    return specs


def cell_context(cfg, streams, cell):
    dataset, condition, seed, tau = cell
    sources = {s: streams[(dataset, condition, seed, tau, s)]
               for s in ("cal_oof", "cal_insample", "dep")}
    users = np.unique(sources["dep"].subject)
    cal_counts = confusion_counts(sources["cal_oof"], users)
    prior = np.asarray(cfg["defences"]["target_prior"])
    util_cal = net_progress_from(np.einsum("y,uyc->uyc", prior, cal_counts / np.maximum(
        cal_counts.sum(-1, keepdims=True), 1)))[3]
    groupings = {label: make_groups(users, util_cal, size)
                 for label, size in g_labels(cfg, len(users)).items()}
    return sources, users, cal_counts, groupings


def battery_inputs(cfg, streams, spec, context_cache):
    cell = (spec["dataset"], spec["condition"], spec["seed"], spec["tau"])
    if cell not in context_cache:
        context_cache[cell] = cell_context(cfg, streams, cell)
    sources, users, cal_counts, groupings = context_cache[cell]
    if spec["defence"] == "none":
        cal, dep = sources["cal_oof"], sources["dep"]
    else:
        M, _ = channel(spec["defence"], cal_counts, groupings, cfg)
        # Same generator and draw order as cell_job -> identical defended streams.
        rng = rng_for(cfg["seed"], spec["dataset"], spec["condition"], spec["seed"],
                      str(spec["tau"]), spec["defence"], spec["pp_seed"])
        cal = apply_channel(sources["cal_oof"], M, users, rng)
        dep = apply_channel(sources["dep"], M, users, rng)
    if spec["scenario"] == "within_day":
        a, b = (1, 2) if spec["direction"] == "1to2" else (2, 1)
        enrol, test = dep.subset(dep.segment == a), dep.subset(dep.segment == b)
    else:
        enrol = sources["cal_insample"] if spec["scenario"] == "cross_day_insample" else cal
        test = dep
    return enrol, test, users, groupings


def available_workers(cfg):
    from .exp_l import memory_snapshot
    free = memory_snapshot()["available_mib"]
    spare = free - cfg["memory"]["reserve_mib"]
    return max(1, min(cfg["workers"], int(spare // cfg["memory"]["per_worker_mib"]))), free


def run_parallel(function, argument_lists, cfg, log=None):
    """Batches through a loky pool; the worker count follows the free-memory gate."""
    results, free_log = [], []
    position = 0
    while position < len(argument_lists):
        workers, free = available_workers(cfg)
        waited = 0
        # Below the reserve no batch starts: wait for memory to recover, bounded.
        while free < cfg["memory"]["reserve_mib"] and waited < cfg["memory"]["wait_seconds"]:
            time.sleep(30)
            waited += 30
            workers, free = available_workers(cfg)
        free_log.append(free)
        if free < cfg["memory"]["reserve_mib"]:
            raise MemoryError(f"Free memory {free:.0f} MiB below the reserve")
        batch = argument_lists[position:position + 4 * workers]
        with parallel_config(backend="loky", inner_max_num_threads=1):
            results.extend(Parallel(n_jobs=workers, batch_size=1)(
                delayed(function)(*args) for args in batch))
        position += len(batch)
        if log:
            log(f"{position}/{len(argument_lists)} jobs, workers {workers}, free {free:.0f} MiB")
    return results, dict(min_free_mib=float(min(free_log)) if free_log else None)


# ------------------------------------------------------------------ summaries

def collect_recalls(cell_results, battery_results):
    """Long frame of per-user recall vectors (one row per seed-level estimate)."""
    rows = []
    for res in cell_results:
        dataset, condition, seed, tau = res["cell"]
        for row in res["recall"]:
            rows.append(dict(dataset=dataset, condition=condition, seed=seed, tau=tau, **row))
    groups = {}
    for res in battery_results:
        if res["status"] != "ok":
            continue
        spec = res["spec"]
        scenario = spec["scenario"]
        for row in res["rows"]:
            key = (spec["dataset"], spec["condition"], spec["seed"], spec["tau"],
                   spec["defence"], scenario, row["attacker"], spec["k"], row["metric"])
            groups.setdefault(key, []).append(row["recall"])
    chances = {}
    for res in cell_results:
        dataset, condition, seed, tau = res["cell"]
        chances[(dataset, condition, seed, tau)] = chance(res["users"], res["groupings"])
    for key, values in groups.items():
        dataset, condition, seed, tau, defence, scenario, attacker, k, metric = key
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            recall = np.nanmean(values, axis=0)
        rows.append(dict(dataset=dataset, condition=condition, seed=seed, tau=tau,
                         scenario=scenario, enrolment="all", family="battery", attacker=attacker,
                         k=k, metric=metric, defence=defence, recall=recall,
                         chance=chances[(dataset, condition, seed, tau)][metric]))
    return pd.DataFrame(rows)


def summarize_recalls(frame, cfg):
    """Seed-mean per user, then cohort BA (mean recall), excess over chance and t-interval."""
    keys = ["dataset", "condition", "tau", "defence", "scenario", "enrolment", "family",
            "attacker", "k", "metric"]
    out, vectors = [], {}
    for key, part in frame.groupby(keys, sort=True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            recall = np.nanmean(np.stack(part.recall.to_numpy()), axis=0)
            excess = np.nanmean(np.stack((part.recall - part.chance).to_numpy()), axis=0)
        low, high = t_interval(excess, cfg["statistics"]["confidence"])
        chance_mean = float(np.nanmean(recall - excess))
        row = dict(zip(keys, key), view="aware" if "aware" in key[7] else "blind",
                   ba=float(np.nanmean(recall)), chance=chance_mean,
                   excess=float(np.nanmean(excess)), excess_low=low, excess_high=high,
                   ba_low=low + chance_mean, ba_high=high + chance_mean,
                   users=int(np.isfinite(recall).sum()), users_total=len(recall),
                   seeds=int(part.seed.nunique()))
        out.append(row)
        vectors[key] = excess
    return pd.DataFrame(out), vectors


def strongest(summary):
    """Max over attackers (per family set) of the cohort excess; scenario max over within-day
    and cross-day (cross-fitted), as in C's strongest flag but labelled."""
    rows = []
    base = summary[summary.enrolment == "all"]
    keys = ["dataset", "condition", "tau", "defence", "k", "metric"]
    for family_set, members in (("bayes", ["bayes"]), ("battery", ["battery"]),
                                ("all", ["bayes", "battery"])):
        part = base[base.family.isin(members)]
        for scenario in ("within_day", "cross_day", "cross_day_insample", "cross_day_eq_insample",
                         "max_within_cross"):
            scen = part[part.scenario.isin(["within_day", "cross_day"])] if (
                scenario == "max_within_cross") else part[part.scenario == scenario]
            if scen.empty:
                continue
            idx = scen.groupby(keys).excess.idxmax()
            best = scen.loc[idx.values].copy()
            best["family_set"] = family_set
            best["best_scenario"] = best.scenario
            best["scenario"] = scenario
            best["candidates"] = scen.groupby(keys).size().reindex(
                pd.MultiIndex.from_frame(best[keys])).to_numpy()
            rows.append(best)
    return pd.concat(rows, ignore_index=True)


def information_tables(cell_results, cfg):
    task, ident = [], []
    for res in cell_results:
        dataset, condition, seed, tau = res["cell"]
        seconds = cfg["datasets"][dataset]["trial_seconds"]
        for row in res["utility"]:
            task.append(dict(dataset=dataset, condition=condition, seed=seed, tau=tau,
                             defence=row["defence"], day=row["day"],
                             correct=float(row["correct"].mean()),
                             wrong=float(row["wrong"].mean()), hold=float(row["hold"].mean()),
                             net_progress=float(row["net"].mean()),
                             wrong_weight=wrong_weight(tau),
                             weighted_utility=float(row["utility"].mean()),
                             bits_per_trial=float(row["bits"].mean()),
                             itr_bits_per_min=float(row["bits"].mean() * 60 / seconds)))
        for row in res["information"]:
            ident.append(dict(dataset=dataset, condition=condition, seed=seed, tau=tau,
                              users=len(res["users"]), **row))
    task = pd.DataFrame(task).groupby(["dataset", "condition", "tau", "defence", "day"],
                                      as_index=False).mean(numeric_only=True).drop(columns="seed")
    ident = pd.DataFrame(ident).groupby(["dataset", "condition", "tau", "defence", "view",
                                         "grouping"], as_index=False).mean(
        numeric_only=True).drop(columns="seed")
    return task, ident


def commands_to_identification(curves, ident):
    rows = []
    base = curves[(curves.family_set == "bayes") & (curves.metric == "nway")
                  & curves.scenario.isin(["within_day", "cross_day"])]
    for (dataset, condition, tau, defence, scenario), part in base.groupby(
            ["dataset", "condition", "tau", "defence", "scenario"]):
        part = part.sort_values("k")
        n = int(round(1 / part.chance.iloc[0]))
        row = dict(dataset=dataset, condition=condition, tau=tau, defence=defence,
                   scenario=scenario, users=n, k_max=int(part.k.max()),
                   ba_at_k_max=float(part.ba.iloc[-1]))
        for name, level in (("2x_chance", 2 / n), ("0.10", 0.10), ("0.50", 0.50)):
            hit = part[part.ba >= level]
            row[f"observed_k_{name}"] = int(hit.k.iloc[0]) if len(hit) else None
            for view in ("blind", "aware"):
                bits = ident[(ident.dataset == dataset) & (ident.condition == condition)
                             & (ident.tau == tau) & (ident.defence == defence)
                             & (ident.view == view) & (ident.grouping == "nway")].bits
                if len(bits):
                    row[f"fano_k_{name}_{view}"] = fano_k(float(bits.iloc[0]), n, level)
        rows.append(row)
    return pd.DataFrame(rows)


def frontier_table(curves, task, cfg):
    """Deployment-day utility and identity per defence. Primary utility: weighted utility
    P(correct) - w P(wrong) (w = tau / (1 - tau)) and its loss against the undefended cell;
    correct, wrong-move and hold rates and net progress (secondary) are reported alongside.
    Cross-day EQ identity uses the cross-fitted enrolment stream (cross_day, primary);
    cross_day_eq_insample columns are the in-sample lower bound."""
    rows = []
    kref = cfg["statistics"]["k_reference"]
    best = curves[(curves.family_set == "bayes")]
    dep = task[task.day == "dep"]
    for _, u in dep.iterrows():
        sel = best[(best.dataset == u.dataset) & (best.condition == u.condition)
                   & (best.tau == u.tau) & (best.defence == u.defence)]
        if sel.empty:
            continue
        kmax = int(sel.k.max())
        row = dict(u)
        row.pop("day")
        if u.defence.startswith("rr_q"):
            row["epsilon_per_command"] = rr_epsilon(float(u.defence[4:]))
        own = u.defence.split("_", 1)[1] if u.defence.startswith("eq") else None
        for scenario in ("within_day", "cross_day", "cross_day_eq_insample", "max_within_cross"):
            for k in (kref, kmax):
                cell = sel[(sel.scenario == scenario) & (sel.k == k)]
                nway = cell[cell.metric == "nway"]
                row[f"nway_ba_{scenario}_k{k if k == kref else 'max'}"] = (
                    float(nway.ba.iloc[0]) if len(nway) else np.nan)
                for metric in sorted(set(cell.metric) - {"nway"}):
                    m = cell[cell.metric == metric]
                    row[f"{metric}_excess_{scenario}_k{k if k == kref else 'max'}"] = float(
                        m.excess.iloc[0])
                if own is not None:
                    m = cell[cell.metric == own]
                    row[f"own_group_excess_{scenario}_k{k if k == kref else 'max'}"] = (
                        float(m.excess.iloc[0]) if len(m) else np.nan)
        rows.append(row)
    frame = pd.DataFrame(rows).rename(columns={"correct": "correct_rate", "wrong": "wrong_move_rate",
                                               "hold": "hold_rate"})
    keys = ["dataset", "condition", "tau"]
    undefended = frame[frame.defence == "none"].set_index(keys).weighted_utility
    frame["weighted_utility_loss"] = undefended.reindex(
        pd.MultiIndex.from_frame(frame[keys])).to_numpy() - frame.weighted_utility
    first = keys + ["defence", "wrong_weight", "weighted_utility", "weighted_utility_loss",
                    "correct_rate", "wrong_move_rate", "hold_rate", "net_progress"]
    return frame[first + [c for c in frame.columns if c not in first]]


# ------------------------------------------------------------------ figures

def plot_curves(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = curves[(curves.defence == "none") & (curves.metric == "nway")]
    conditions = ["P0", "P2-RC_d8"]
    datasets = list(dict.fromkeys(base.dataset))
    fig, axes = plt.subplots(len(datasets) * 2, 3, figsize=(13, 3.2 * 2 * len(datasets)),
                             squeeze=False)
    for i, dataset in enumerate(datasets):
        for j, condition in enumerate(conditions):
            for col, scenario in enumerate(("within_day", "cross_day", "cross_day_insample")):
                ax = axes[2 * i + j, col]
                part = base[(base.dataset == dataset) & (base.condition == condition)
                            & (base.scenario == scenario)]
                for (fam, tau), frame in part.groupby(["family_set", "tau"]):
                    if fam == "all":
                        continue
                    frame = frame.sort_values("k")
                    ax.plot(frame.k, frame.ba, marker="o" if fam == "bayes" else "s",
                            linestyle="-" if tau == 0.8 else "--",
                            label=f"{fam} tau={tau:g}")
                if len(part):
                    ax.axhline(part.chance.iloc[0], color="grey", linestyle=":")
                    ax.axhline(2 * part.chance.iloc[0], color="tab:red", linestyle=":")
                ax.set_xscale("log")
                ax.set_title(f"{dataset} {condition}: {scenario}", fontsize=8)
                ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=6)
    fig.supxlabel("commands observed per decision (k)")
    fig.supylabel("strongest identity BA (N-way, undefended)")
    fig.suptitle(LABEL + "\nmax over attackers per family; dotted: chance and 2x chance",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_frontier(frontier, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kref = cfg["statistics"]["k_reference"]
    datasets = list(dict.fromkeys(frontier.dataset))
    taus = cfg["command"]["taus"]
    fig, axes = plt.subplots(len(datasets) * len(taus), 2, figsize=(11, 3.3 * len(datasets) * len(taus)),
                             squeeze=False)
    for i, (dataset, tau) in enumerate([(d, t) for d in datasets for t in taus]):
        part = frontier[(frontier.dataset == dataset) & (frontier.tau == tau)
                        & (frontier.condition.isin(["P0", "P2-RC_d8"]))]
        for condition, frame in part.groupby("condition"):
            for family, marker in (("none", "*"), ("rr", "o"), ("eqm", "^"), ("eqc", "v")):
                sel = frame[frame.defence.str.startswith(family)]
                axes[i, 0].scatter(sel.weighted_utility, sel[f"nway_ba_max_within_cross_k{kref}"],
                                   marker=marker, label=f"{condition} {family}", s=22)
                col = (f"own_group_excess_within_day_k{kref}" if family.startswith("eq")
                       else f"g3_excess_within_day_k{kref}")
                if col in sel:
                    axes[i, 1].scatter(sel.weighted_utility, sel[col], marker=marker, s=22,
                                       label=f"{condition} {family}")
        axes[i, 0].set_title(f"{dataset} tau={tau:g}: N-way BA at k={kref} (max within/cross)",
                             fontsize=8)
        axes[i, 1].set_title(f"{dataset} tau={tau:g}: within-group excess at k={kref} "
                             "(own group for EQ, g3 otherwise; within-day)", fontsize=8)
        for ax in axes[i]:
            ax.set_xlabel(f"deployment weighted utility P(correct) - {wrong_weight(tau):g} P(wrong)",
                          fontsize=7)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(LABEL + " - RR / EQ frontier (Bayes attackers)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ LP evidence

def lp_evidence(seed=0):
    """Synthetic-user evidence that the EQ LP equalizes and attains the utility optimum."""
    prior = np.array([0.5, 0.5])
    rows = []

    def record(case, conf, variant, expected=None, w=1.0):
        M, shared, optimum, cert = eq_lp(conf, prior, variant, w)
        dist = achieved(conf, M, prior, variant)
        rows.append(dict(case=case, variant=variant, users=len(conf), wrong_weight=w,
                         optimum=optimum, expected_optimum=expected,
                         equalization_residual=float(np.max(np.abs(dist - dist[0]))),
                         row_sum_error=float(np.max(np.abs(M.sum(-1) - 1))),
                         min_entry=float(M.min()), **cert))
        return M

    perfect = np.array([[1, 0, 0], [0, 0, 1.]])
    hold = np.array([[0, 1, 0], [0, 1, 0.]])
    good = np.array([[0.9, 0, 0.1], [0.1, 0, 0.9]])
    weak = np.array([[0.7, 0, 0.3], [0.3, 0, 0.7]])
    for variant in ("marginal", "confusion"):
        record("identical users keep identity", np.stack([good, good, good]), variant, 0.8)
        record("perfect + always-hold", np.stack([perfect, hold]), variant,
               0.5 if variant == "marginal" else 0.0)
    record("accuracy 0.9 + 0.7 (garbled to the weaker channel)", np.stack([good, weak]),
           "confusion", 0.4)
    # Weighted utility at tau = 0.8 (w = 4): a move is kept only if it gains more than it
    # costs, so 0.9-accurate users keep their decoder (0.9 - 4 * 0.1) and 0.7-accurate users
    # are better off always holding (0.7 - 4 * 0.3 < 0).
    w = wrong_weight(0.8)
    for variant in ("marginal", "confusion"):
        record("identical users, weighted utility", np.stack([good] * 3), variant,
               0.9 - w * 0.1, w)
        record("identical weak users, weighted utility (moves cost more than they gain)",
               np.stack([weak] * 3), variant, 0.0, w)
    rng = np.random.default_rng(seed)
    for n in (2, 3, 5, 9):
        for variant in ("marginal", "confusion"):
            conf = rng.dirichlet(np.ones(3), size=(n, 2))
            record(f"random Dirichlet users n={n}", conf, variant)
            record(f"random Dirichlet users n={n}, weighted utility", conf, variant, None, w)
    # Sampled outputs: a within-group Bayes attacker on long streams is at chance.
    conf = rng.dirichlet(np.ones(3) * 0.7, size=(3, 2))
    out = []
    for variant in ("marginal", "confusion"):
        M, _, _, _ = eq_lp(conf, prior, variant)
        trials = 4000
        subject = np.repeat(np.arange(3), 2 * trials)
        target = rng.integers(2, size=len(subject))
        probs = conf[subject, target]
        cmd = (rng.random(len(subject))[:, None] > np.cumsum(probs, 1)[:, :2]).sum(1)
        users = np.arange(3)
        segment = np.tile(np.r_[np.ones(trials, np.int8), np.full(trials, 2, np.int8)], 3)
        undefended = Stream(subject, segment, target.astype(np.int8), cmd.astype(np.int8))
        defended = apply_channel(undefended, M, users, rng)
        group = {"gall": np.zeros(3, np.int64)}
        for name, s in (("undefended", undefended), ("defended", defended)):
            rec = bayes_evaluate(dict(cal_oof=s, dep=s), users, group, [50], dict(
                attackers=dict(bayes=list(BAYES), dirichlet_alpha=0.5, enrolment_sizes=[])),
                rng, [])
            for attacker in BAYES:
                value = float(np.nanmean(rec[("within_day", "all", attacker, 50, "gall")]))
                out.append(dict(variant=variant, stream=name, attacker=attacker, k=50,
                                within_group_ba=value, chance=1 / 3))
    return pd.DataFrame(rows), pd.DataFrame(out)


# ------------------------------------------------------------------ power analysis for M

CALIBRATION_SCOPE = (
    "simulate_days draws outputs i.i.d. given (user, target) within a day, with no within-day "
    "drift and no sequential dependence. calibrate_effect fits (shrink, drift_sd) by least "
    "squares to the source's undefended Bayes N-way BA of the categorical attackers (cat_blind, "
    "cat_aware) within-day and cross-day. For K's Lee2019 source the fitted model reproduces the "
    "cross-day targets but underfits within-day identification at small k "
    "(power_calibration_fit.csv). The power function is therefore calibrated for cross-day "
    "statistics with categorical attackers - M's candidates cal_to_dep and dep_to_dep are both "
    "cross-day; the Markov attackers in the simulated candidate set see no sequential "
    "structure, and within-day statistics at small k would be understated.")


def theta_pool(stream_cal, stream_dep, smoothing=0.5):
    """Per-user centred log-probabilities of P(output | target) from both days pooled.

    Categories never observed in the source are structural zeros (-inf)."""
    users = np.unique(stream_dep.subject)
    counts = confusion_counts(stream_cal, users) + confusion_counts(stream_dep, users)
    active = counts.sum((0, 1)) > 0
    p = (counts + smoothing) / (counts + smoothing).sum(-1, keepdims=True)
    log = np.where(active, np.log(p), -np.inf)
    centred = log - np.where(active, log, 0).sum(-1, keepdims=True) / active.sum()
    return np.where(active, centred, -np.inf)


def _softmax(x):
    x = x - np.max(np.where(np.isfinite(x), x, -1e300), axis=-1, keepdims=True)
    e = np.where(np.isfinite(x), np.exp(x), 0.0)
    return e / e.sum(-1, keepdims=True)


def simulate_days(effect, n_users, days, trials, rng, segments=1):
    """Synthetic study: per-user latent P(output | target) = softmax(mu + s (theta_u - mu) +
    delta_day), delta ~ N(0, drift_sd^2) i.i.d. per day/target/category; balanced targets in
    random order; outputs i.i.d. given target (no sequential dependence)."""
    pool = np.asarray(effect["theta"], dtype=float)
    active = np.isfinite(pool[0, 0])
    mu = np.where(active, np.where(np.isfinite(pool), pool, 0).mean(0), -np.inf)
    pick = rng.choice(len(pool), n_users, replace=n_users > len(pool))
    theta = np.where(active, mu + effect["shrink"] * (np.where(np.isfinite(pool[pick]),
                                                               pool[pick], 0) - mu), -np.inf)
    out, probs = [], []
    half = trials // 2
    for _ in range(days):
        noise = effect["drift_sd"] * rng.standard_normal(theta.shape)
        p = _softmax(np.where(active, theta + noise, -np.inf))
        target = np.stack([rng.permutation(np.r_[np.zeros(half), np.ones(trials - half)])
                           for _ in range(n_users)]).astype(np.int64)
        subject = np.repeat(np.arange(n_users), trials)
        tflat = target.ravel()
        cum = np.cumsum(p[subject, tflat], axis=1)
        output = (rng.random(len(subject))[:, None] > cum[:, :2]).sum(1)
        segment = np.tile(1 + (np.arange(trials) * segments) // trials, n_users)
        out.append(Stream(subject, segment.astype(np.int8), tflat.astype(np.int8),
                          output.astype(np.int8)))
        probs.append(p)
    return out, probs


def _bayes_vectors(enrol, tests, users, ks, attackers, alpha, groupings, rng):
    """Per-user recall (nway and groupings) averaged over test streams, per attacker and k."""
    enrolled = Counts(enrol).enrol(users)
    acc = {}
    for test in tests:
        counts = Counts(test)
        for k in ks:
            starts = window_starts(test, k)
            win = counts.windows(starts, k)
            truth = test.subject[starts]
            for attacker in attackers:
                scores = bayes_scores(attacker, win, enrolled, alpha)
                for metric, values in recalls(scores, users, truth, groupings, rng).items():
                    acc.setdefault((attacker, k, metric), []).append(values)
    return {key: np.mean(v, axis=0) for key, v in acc.items()}


def calibration_statistics(effect, n_users, ks, attackers, alpha, rng):
    """L-structured synthetic study (2 days x 200 trials, two 100-trial segments per day):
    mean N-way BA within-day (segment A -> B, both directions) and cross-day (day 1 -> 2)."""
    (cal, dep), _ = simulate_days(effect, n_users, 2, 200, rng, segments=2)
    users = np.arange(n_users)
    seg = {s: dep.subset(dep.segment == s) for s in (1, 2)}
    out = {}
    within = [_bayes_vectors(seg[a], [seg[b]], users, ks, attackers, alpha, {}, rng)
              for a, b in ((1, 2), (2, 1))]
    cross = _bayes_vectors(cal, [dep], users, ks, attackers, alpha, {}, rng)
    for attacker in attackers:
        for k in ks:
            out[("within_day", attacker, k)] = float(np.mean(
                [w[(attacker, k, "nway")].mean() for w in within]))
            out[("cross_day", attacker, k)] = float(cross[(attacker, k, "nway")].mean())
    return out


def calibrate_effect(theta, observed, cfg, seed):
    """Indirect inference: grid over (shrink, drift_sd) minimising squared error between the
    simulated and observed undefended Bayes BA (within-day and cross-day, calibration k).
    Scope and known misfit: CALIBRATION_SCOPE."""
    p = cfg["power"]
    rows = []
    for shrink in p["shrink_grid"]:
        for drift in p["drift_grid"]:
            effect = dict(theta=theta, shrink=shrink, drift_sd=drift)
            sims = [calibration_statistics(effect, len(theta), p["calibration_k"],
                                           p["calibration_attackers"],
                                           cfg["attackers"]["dirichlet_alpha"],
                                           rng_for(seed, "calib", str(shrink), str(drift), r))
                    for r in range(p["calibration_replicates"])]
            mean = {key: float(np.mean([s[key] for s in sims])) for key in sims[0]}
            sse = float(sum((mean[key] - observed[key]) ** 2 for key in observed))
            rows.append(dict(shrink=shrink, drift_sd=drift, sse=sse,
                             **{"sim_" + "_".join(map(str, key)): v for key, v in mean.items()}))
    table = pd.DataFrame(rows)
    best = table.loc[table.sse.idxmin()]
    return dict(shrink=float(best.shrink), drift_sd=float(best.drift_sd), sse=float(best.sse)), table


def calibration_fit(table, observed, best, name):
    """Observed versus simulated calibration targets at the selected (shrink, drift_sd)."""
    row = table[(table.shrink == best["shrink"]) & (table.drift_sd == best["drift_sd"])].iloc[0]
    out = []
    for (scenario, attacker, k), value in observed.items():
        simulated = float(row[f"sim_{scenario}_{attacker}_{k}"])
        out.append(dict(source=name, scenario=scenario, attacker=attacker, k=k,
                        observed_ba=value, simulated_ba=simulated, residual=simulated - value,
                        shrink=best["shrink"], drift_sd=best["drift_sd"]))
    return pd.DataFrame(out)


def study_statistics(effect, design, rule_grid, rng):
    """One simulated M study: per-candidate per-user vectors for M3 and M4 at every k / g.

    EQ matrices are fitted on day 1 with the weighted utility (rule_grid['wrong_weight']);
    day 1 itself, the cal_to_dep enrolment stream, is post-processed by cross-fitted
    matrices (crossfit_matrices), days 2.. by the matrices fitted on all of day 1."""
    n, days, trials = design["n_users"], design["days"], design["trials_per_session"]
    streams, _ = simulate_days(effect, n, days, trials, rng)
    users = np.arange(n)
    attackers, alpha = rule_grid["attackers"], rule_grid["alpha_dirichlet"]
    ks = sorted(set(rule_grid["k_grid"]) | {rule_grid["k_base"]})
    w = rule_grid["wrong_weight"]
    prior = np.array([0.5, 0.5])

    def scenario_sets(day_streams):
        sets = {}
        if "cal_to_dep" in rule_grid["scenarios"]:
            sets["cal_to_dep"] = [(day_streams[0], day_streams[1:])]
        if "dep_to_dep" in rule_grid["scenarios"]:
            sets["dep_to_dep"] = [(day_streams[1], [day_streams[2]]),
                                  (day_streams[2], [day_streams[1]])]
        return sets

    def candidate_vectors(day_streams, groupings, k_list):
        out = {}
        for scenario, pairs in scenario_sets(day_streams).items():
            parts = [_bayes_vectors(e, t, users, k_list, attackers, alpha, groupings, rng)
                     for e, t in pairs]
            for key in parts[0]:
                out[(scenario,) + key] = np.mean([p[key] for p in parts], axis=0)
        return out

    undefended = candidate_vectors(streams, {}, ks)
    m3 = {k: np.stack([undefended[(s, a, k, "nway")] for s in scenario_sets(streams)
                       for a in attackers]) for k in ks}
    m4 = {}
    cal_counts = confusion_counts(streams[0], users)
    util = net_progress_from(np.einsum("y,uyc->uyc", prior, cal_counts / np.maximum(
        cal_counts.sum(-1, keepdims=True), 1)))[3]
    identity = np.repeat(np.eye(3)[None], n, axis=0)
    dep_counts = sum(confusion_counts(s, users) for s in streams[1:])
    base_utility = task_information(dep_counts, identity, w)["utility"]
    for g in rule_grid["g_grid"]:
        group = make_groups(users, util, g)
        groupings = {"grp": group}
        ch = 1.0 / np.bincount(group)[group]
        undef_g = candidate_vectors(streams, groupings, rule_grid["m4_k"])
        for variant in rule_grid["eq_variants"]:
            M, _ = fit_eq(cal_counts, group, variant, prior, rule_grid["eq_smoothing"], w)
            fold, fold_M = crossfit_matrices(streams[0], users, group, variant, prior,
                                             rule_grid["eq_smoothing"], w,
                                             rule_grid["crossfit_folds"])
            defended = ([apply_crossfit(streams[0], fold, fold_M, users, rng)]
                        + [apply_channel(s, M, users, rng) for s in streams[1:]])
            def_g = candidate_vectors(defended, groupings, rule_grid["m4_k"])
            realized = sum(confusion_counts(s, users) for s in defended[1:])
            loss = base_utility - task_information(realized, identity, w)["utility"]
            for k in rule_grid["m4_k"]:
                cands = [(s, a) for s in scenario_sets(streams) for a in attackers]
                m4[(variant, g, k)] = dict(
                    D=np.stack([def_g[(s, a, k, "grp")] - ch for s, a in cands]),
                    D0=np.stack([undef_g[(s, a, k, "grp")] - ch for s, a in cands]),
                    loss=loss)
    return dict(m3=m3, m4=m4)


def m3_pvalue(v_star, v_base, n_users, factor):
    """L's C3 structure: min(1, J min_j max{T(v_j, factor/N), max_l T(v_j - w_l, 0)})."""
    threshold = t_greater(v_star, factor / n_users)
    diff = t_greater(v_star[:, None, :] - v_base[None, :, :], 0.0).max(axis=1)
    return float(min(1.0, len(v_star) * np.maximum(threshold, diff).min()))


def m4_pvalue(D, D0, loss, m, delta, loss_max):
    """Bound (every candidate below m: intersection-union, max p), reduction (C2-style:
    some undefended candidate exceeds every defended one by delta, Bonferroni over J) and
    utility (mean utility loss below loss_max; K passes the weighted-utility loss);
    p = max of the three."""
    p_bound = float(t_less(D, m).max())
    p_red = float(min(1.0, len(D0) * t_greater(D0[:, None, :] - D[None, :, :], delta).max(
        axis=1).min()))
    p_util = float(t_less(loss, loss_max))
    return max(p_bound, p_red, p_util), dict(bound=p_bound, reduction=p_red, utility=p_util)


def power_function(effect, rule=None, design=None, n_sim=200, seed=0, k_grid=None,
                   g_grid=None, m_grid=None, delta_grid=None, eq_variants=None):
    """Parameterised simulation power of M3 and M4 (reusable by M0).

    effect: dict(theta=(P,2,3) centred log-probability pool, shrink, drift_sd) - see
            calibrate_effect / results/exp_k/power_effects.json.
    rule:   decision rule (configs/exp_k.yaml decision_rule); grids default to the rule's
            single values. design: n_users, trials_per_session, days.
    M4's utility criterion is the weighted-utility loss with w = tau / (1 - tau) at the
    rule's tau; the cal_to_dep enrolment day is post-processed by cross-fitted EQ matrices.
    Calibration scope: CALIBRATION_SCOPE (cross-day statistics, categorical attackers).
    Returns (m3, m4) DataFrames: power at the conservative Holm level alpha/holm_family
    ("power") and at alpha ("power_upper"; Holm's last step), plus mean statistics.
    """
    cfg = load_cfg()
    rule = dict(rule or cfg["decision_rule"])
    design = dict(design or {k: cfg["power"][k] for k in ("n_users", "trials_per_session", "days")})
    k_grid = list(k_grid or [rule["k_star"]])
    grid = dict(attackers=rule["attackers"], scenarios=rule["scenarios"],
                alpha_dirichlet=cfg["attackers"]["dirichlet_alpha"], k_grid=k_grid,
                k_base=rule["k_base"], m4_k=k_grid if g_grid is not None else [rule["k_star"]],
                g_grid=list(g_grid or [rule["g_star"]]),
                eq_variants=list(eq_variants or [rule["eq_variant"]]),
                eq_smoothing=cfg["defences"]["eq_smoothing"],
                wrong_weight=wrong_weight(rule["tau"]),
                crossfit_folds=cfg["defences"]["eq_crossfit_folds"])
    m_grid, delta_grid = list(m_grid or [rule["m"]]), list(delta_grid or [rule["delta"]])
    level = rule["alpha"] / rule["holm_family"]
    n = design["n_users"]
    sims, _ = run_parallel(study_statistics, [(effect, design, grid, rng_for(seed, "study", i))
                                              for i in range(n_sim)], cfg)
    m3_rows = []
    for k in k_grid:
        ps = np.array([m3_pvalue(s["m3"][k], s["m3"][rule["k_base"]], n,
                                 rule["threshold_factor"]) for s in sims])
        m3_rows.append(dict(k=k, power=float((ps <= level).mean()),
                            power_upper=float((ps <= rule["alpha"]).mean()),
                            mc_se=float(np.sqrt((ps <= level).mean() * (1 - (ps <= level).mean())
                                                / n_sim)),
                            mean_strongest_ba=float(np.mean([s["m3"][k].mean(1).max() for s in sims])),
                            mean_ba_k_base=float(np.mean([s["m3"][rule["k_base"]].mean(1).max()
                                                          for s in sims])),
                            n_sim=n_sim, level=level))
    m4_rows = []
    for key in sims[0]["m4"]:
        variant, g, k = key
        D = [s["m4"][key]["D"] for s in sims]
        D0 = [s["m4"][key]["D0"] for s in sims]
        loss = [s["m4"][key]["loss"] for s in sims]
        for m in m_grid:
            for delta in delta_grid:
                results = [m4_pvalue(D[i], D0[i], loss[i], m, delta, rule["utility_loss_max"])
                           for i in range(n_sim)]
                ps = np.array([r[0] for r in results])
                comp = pd.DataFrame([r[1] for r in results])
                m4_rows.append(dict(
                    eq_variant=variant, g=g, k=k, m=m, delta=delta,
                    power=float((ps <= level).mean()), power_upper=float((ps <= rule["alpha"]).mean()),
                    pass_bound=float((comp.bound <= level).mean()),
                    pass_reduction=float((comp.reduction <= level).mean()),
                    pass_utility=float((comp.utility <= level).mean()),
                    mean_D_defended_max=float(np.mean([d.mean(1).max() for d in D])),
                    mean_D_undefended_max=float(np.mean([d.mean(1).max() for d in D0])),
                    mean_loss=float(np.mean([np.mean(x) for x in loss])), n_sim=n_sim, level=level))
    return pd.DataFrame(m3_rows), pd.DataFrame(m4_rows)


# ------------------------------------------------------------------ orchestration

def _log(message):
    print(f"[exp_k {time.strftime('%H:%M:%S')}] {message}", flush=True)


def declared_grid(cfg):
    labels = {d: g_labels(cfg, dcfg["users"]) for d, dcfg in cfg["datasets"].items()}
    return dict(
        label=LABEL,
        datasets={d: dict(conditions=list(dcfg["conditions"]), k=dcfg["k"],
                          battery_k=dcfg["battery_k"], groupings=labels[d])
                  for d, dcfg in cfg["datasets"].items()},
        taus=cfg["command"]["taus"], model_seeds=cfg["model_seeds"],
        postprocessing_seeds=cfg["defences"]["postprocessing_seeds"],
        undefended_only=cfg["undefended_only"],
        defences={d: defence_list(cfg, labels[d]) for d in cfg["datasets"]},
        scenarios=cfg["scenarios"], views=cfg["attackers"]["views"],
        attackers=dict(undefended="battery (logistic, rbf_svm, mlp) x views + Bayes "
                                  "(cat_blind, cat_aware, markov_blind, markov_aware)",
                       defended="Bayes attackers only",
                       operating_points=cfg["operating_points"]),
        metrics=["N-way identification BA (mean per-user recall)",
                 "within-group identification (group given) and excess over 1/|own group|"],
        bayes_cells=len(cells(cfg)), battery_jobs=len(battery_specs(cfg)),
        note="P3 lambda=1 (2a: B's stored lambda-1 OOF and final outputs, no refit) is "
             "undefended only.")


def pilot(cfg, streams=None):
    """Timing pilot on real cells, runtime projection and drop list."""
    start = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "grid.json", declared_grid(cfg))
    t0 = time.perf_counter()
    streams, info = build_streams(cfg, seeds=[cfg["model_seeds"][0]], write=False)
    stream_seconds = (time.perf_counter() - t0) * len(cfg["model_seeds"])
    timings = [dict(unit="streams_all_seeds_projected", seconds=stream_seconds)]
    cell_seconds = {}
    for dataset in cfg["datasets"]:
        cell = (dataset, "P2-RC_d8", cfg["model_seeds"][0], 0.8)
        sources = {s: streams[cell + (s,)] for s in ("cal_oof", "cal_insample", "dep")}
        res = cell_job(cell, sources, cfg)
        cell_seconds[dataset] = res["seconds"]
        timings.append(dict(unit="bayes_cell", dataset=dataset, seconds=res["seconds"],
                            **{f"defence_{k}": v for k, v in res["timing"].items()}))
        _log(f"pilot cell {dataset}: {res['seconds']:.1f}s")
    # Battery profiles: undefended P2-RC tau 0.8 seed 0, one per (dataset, scenario, k, view).
    profiles, cache = [], {}
    for dataset, dcfg in cfg["datasets"].items():
        for scenario, direction in (("within_day", "1to2"), ("cross_day", "cal_to_dep")):
            for k in dcfg["battery_k"]:
                for view in cfg["attackers"]["views"]:
                    spec = dict(root_seed=cfg["seed"], dataset=dataset, condition="P2-RC_d8",
                                seed=cfg["model_seeds"][0], tau=0.8, defence="none", pp_seed=0,
                                scenario=scenario, direction=direction, k=k, view=view, cuts=[])
                    enrol, test, users, groupings = battery_inputs(cfg, streams, spec, cache)
                    profiles.append((spec, enrol, test, users, groupings, dataset))
    measured, mem = run_parallel(battery_job, profiles, cfg, _log)
    profile_seconds = {}
    for res in measured:
        s = res["spec"]
        profile_seconds[(s["dataset"], "within" if s["scenario"] == "within_day" else "cross",
                         s["k"], s["view"])] = res["seconds"]
        timings.append(dict(unit="battery_job", dataset=s["dataset"], scenario=s["scenario"],
                            k=s["k"], view=s["view"], seconds=res["seconds"], status=res["status"],
                            train_windows=res["train_windows"], test_windows=res["test_windows"]))
    # Power units.
    t0 = time.perf_counter()
    l_cell = ("lee2019", "P2-RC_d8", cfg["model_seeds"][0], 0.8)
    theta = theta_pool(streams[l_cell + ("cal_oof",)], streams[l_cell + ("dep",)])
    calibration_statistics(dict(theta=theta, shrink=0.8, drift_sd=0.3), len(theta),
                           cfg["power"]["calibration_k"], cfg["power"]["calibration_attackers"],
                           cfg["attackers"]["dirichlet_alpha"], rng_for(1))
    calib_sim = time.perf_counter() - t0
    rule = cfg["decision_rule"]
    t0 = time.perf_counter()
    grid_m3 = dict(attackers=rule["attackers"], scenarios=rule["scenarios"],
                   alpha_dirichlet=cfg["attackers"]["dirichlet_alpha"],
                   k_grid=cfg["power"]["k_grid"], k_base=1, m4_k=[], g_grid=[], eq_variants=[],
                   eq_smoothing=cfg["defences"]["eq_smoothing"],
                   wrong_weight=wrong_weight(rule["tau"]),
                   crossfit_folds=cfg["defences"]["eq_crossfit_folds"])
    study_statistics(dict(theta=theta, shrink=0.8, drift_sd=0.3), cfg["power"], grid_m3, rng_for(2))
    m3_sim = time.perf_counter() - t0
    t0 = time.perf_counter()
    grid_m4 = dict(grid_m3, k_grid=[1], m4_k=[10, 20, 50], g_grid=cfg["power"]["g_grid"],
                   eq_variants=cfg["defences"]["eq_variants"])
    study_statistics(dict(theta=theta, shrink=0.8, drift_sd=0.3), cfg["power"], grid_m4, rng_for(3))
    m4_sim = time.perf_counter() - t0
    timings += [dict(unit="power_calibration_sim", seconds=calib_sim),
                dict(unit="power_m3_sim", seconds=m3_sim),
                dict(unit="power_m4_sim", seconds=m4_sim)]
    timings.append(dict(unit="pilot_total", seconds=time.perf_counter() - start))
    write_csv(pd.DataFrame(timings), RESULTS / "timing_pilot.csv")
    return reproject(cfg, mem)


def reproject(cfg, memory=None):
    """Runtime projection and drop list from the measured units in timing_pilot.csv, at the
    worker count the free-memory gate allows now."""
    t = pd.read_csv(RESULTS / "timing_pilot.csv")
    cell_seconds = dict(zip(t[t.unit == "bayes_cell"].dataset, t[t.unit == "bayes_cell"].seconds))
    jobs = t[t.unit == "battery_job"]
    profile_seconds = {(r.dataset, "within" if r.scenario == "within_day" else "cross",
                        int(r.k), r.view): r.seconds for r in jobs.itertuples()}
    unit = dict(zip(t.unit, t.seconds))
    power_seconds = dict(calibration=unit["power_calibration_sim"], m3=unit["power_m3_sim"],
                         m4=unit["power_m4_sim"])
    workers, free = available_workers(cfg)
    projection = project(cfg, cell_seconds, profile_seconds, power_seconds,
                         unit["pilot_total"], workers)
    projection.update(free_mib_at_projection=free, pilot_memory=memory)
    write_json(RESULTS / "runtime_projection.json", projection)
    return projection


def _power_units(cfg, reduced):
    p = cfg["power"]
    sources = p["calibration_sources"][:1] if reduced else p["calibration_sources"]
    calib = len(sources) * len(p["shrink_grid"]) * len(p["drift_grid"]) * p["calibration_replicates"]
    m3 = len(sources) * len(p["effect_multipliers"]) * p["n_sim"]
    m4 = len(p["effect_multipliers"]) * (p["n_sim"] if not reduced else p["n_sim"] // 2)
    final = p["n_sim_final"]
    return dict(calibration=calib, m3=m3 + final, m4=m4 + final)


def project(cfg, cell_seconds, profile_seconds, power_seconds, pilot_seconds, workers, drops=()):
    """Projected wall time of the declared grid after each cut in `cut_order`; Bayes cells,
    battery jobs and power simulations all run through run_parallel with `workers`."""
    limit = cfg["runtime"]["limit_seconds"]
    stages = []
    applied = []
    for cut in [None] + cfg["cut_order"]:
        if cut is not None:
            applied.append(cut)
        drops_now = list(drops) + applied
        bayes = sum(cell_seconds[d] * (1 if c not in cfg["undefended_only"] else 0.08)
                    * (0.8 if "bayes_enrolment_sweep" in drops_now else 1)
                    for d, c, _, _ in cells(cfg))
        battery = 0.0
        for spec in battery_specs(cfg, drops_now):
            kind = "within" if spec["scenario"] == "within_day" else "cross"
            battery += profile_seconds.get((spec["dataset"], kind, spec["k"], spec["view"]), 0.0)
        units = _power_units(cfg, "power_grid_reduced" in drops_now)
        power = (units["calibration"] * power_seconds["calibration"]
                 + units["m3"] * power_seconds["m3"] + units["m4"] * power_seconds["m4"])
        parallel_work = bayes + battery
        wall = (parallel_work / workers + power / workers + 120) * cfg["runtime"]["safety_factor"]
        stages.append(dict(cuts=list(applied), bayes_work_s=bayes, battery_work_s=battery,
                           power_work_s=power, projected_run_wall_s=wall,
                           pilot_wall_s=pilot_seconds, total_wall_s=wall + pilot_seconds,
                           battery_jobs=len(battery_specs(cfg, drops_now))))
        if wall + pilot_seconds <= limit:
            break
    final = stages[-1]
    chosen = final["cuts"]
    dropped = []
    full = battery_specs(cfg)
    for cut in chosen:
        dropped.append(dict(cut=cut, battery_jobs_removed=sum(cut in s["cuts"] for s in full),
                            description=CUT_DESCRIPTION[cut]))
    write_csv(pd.DataFrame([dict(row, applied_at="projection") for row in dropped],
                           columns=["cut", "battery_jobs_removed", "description", "applied_at"]),
              RESULTS / "drop_list.csv")
    return dict(label=LABEL, limit_seconds=limit, workers=workers, stages=stages,
                drop_list=chosen, fits=bool(final["total_wall_s"] <= limit),
                basis="Measured pilot units (real P2-RC tau 0.8 seed 0 cells and battery profiles) "
                      "x declared counts; parallel parts divided by the worker count; "
                      f"safety factor {cfg['runtime']['safety_factor']}.")


CUT_DESCRIPTION = {
    "battery_operating_points": "existing battery at the defended operating points",
    "battery_insample_contrast": "existing battery on the in-sample cross-day contrast",
    "battery_P3_lambda1": "existing battery for P3 lambda=1 (Bayes attackers kept)",
    "battery_aware_view_k_ge_20": "target-aware battery view at k >= 20",
    "battery_seeds_1_2": "existing battery for model seeds 1-2 (seed 0 kept)",
    "bayes_enrolment_sweep": "deployment-aware enrolment sweep k_e in {5, 20}",
    "power_grid_reduced": "power calibration on the first source only; M4 grid n_sim halved",
}


def _drops_and_cfg(cfg):
    drops = json.loads((RESULTS / "runtime_projection.json").read_text(
        encoding="utf-8"))["drop_list"]
    if "bayes_enrolment_sweep" in drops:
        cfg = dict(cfg, attackers=dict(cfg["attackers"], enrolment_sizes=[]))
    _log(f"drop list: {drops}")
    return drops, cfg


def _run_cells(cfg, streams, start):
    cell_args = []
    for cell in cells(cfg):
        sources = {s: streams[cell + (s,)] for s in ("cal_oof", "cal_insample", "dep")}
        cell_args.append((cell, sources, cfg))
    cell_results, mem = run_parallel(cell_job, cell_args, cfg, _log)
    joblib.dump(cell_results, ARTIFACTS / "cell_results.joblib", compress=3)
    _log(f"Bayes cells done at {time.perf_counter() - start:.0f}s")
    return cell_results, mem


def rerun_cells(cfg):
    """Bayes cells (every defence) and the report again, reusing the attack battery stored by
    `run`: the battery covers undefended streams only (battery_operating_points was cut) and
    its groupings come from calibration net progress, so EQ changes leave it unchanged."""
    start = time.perf_counter()
    drops, cfg = _drops_and_cfg(cfg)
    streams, _ = build_streams(cfg, write=False)
    cell_results, mem = _run_cells(cfg, streams, start)
    previous = json.loads((RESULTS / "run_summary.json").read_text(encoding="utf-8"))
    summary = report(cfg, cell_results, joblib.load(ARTIFACTS / "battery_results.joblib"))
    summary.update(
        cells_rerun_wall_seconds=time.perf_counter() - start,
        battery_source="artifacts/exp_k/battery_results.joblib (stored by the run stage)",
        first_run_wall_seconds=previous.get("first_run_wall_seconds",
                                            previous.get("run_wall_seconds")),
        memory=dict(cells=mem))
    write_json(RESULTS / "run_summary.json", summary)
    return summary


def run(cfg):
    start = time.perf_counter()
    drops, cfg = _drops_and_cfg(cfg)
    streams, info = build_streams(cfg)
    write_csv(info, RESULTS / "stream_summary.csv")
    _log(f"streams built: {len(streams)}")
    cell_results, mem1 = _run_cells(cfg, streams, start)
    specs = battery_specs(cfg, drops)
    cache, battery_args = {}, []
    for spec in specs:
        enrol, test, users, groupings = battery_inputs(cfg, streams, spec, cache)
        battery_args.append((spec, enrol, test, users, groupings, spec["dataset"]))
    battery_results, mem2 = run_parallel(battery_job, battery_args, cfg, _log)
    joblib.dump(battery_results, ARTIFACTS / "battery_results.joblib", compress=3)
    _log(f"battery done at {time.perf_counter() - start:.0f}s")
    summary = report(cfg, cell_results, battery_results)
    summary["run_wall_seconds"] = time.perf_counter() - start
    summary["memory"] = dict(cells=mem1, battery=mem2)
    write_json(RESULTS / "run_summary.json", summary)
    return summary


def report(cfg, cell_results, battery_results):
    frame = collect_recalls(cell_results, battery_results)
    summary, _ = summarize_recalls(frame, cfg)
    summary.insert(0, "label", "exploratory")
    write_csv(summary, RESULTS / "identity_curves_all.csv")
    best = strongest(summary)
    write_csv(best, RESULTS / "identity_curves.csv")
    plot_curves(best, RESULTS / "identity_curves.png")
    task, ident = information_tables(cell_results, cfg)
    write_csv(task, RESULTS / "information_task.csv")
    write_csv(ident, RESULTS / "information_identity.csv")
    write_csv(commands_to_identification(best, ident), RESULTS / "commands_to_identification.csv")
    frontier = frontier_table(best, task, cfg)
    write_csv(frontier, RESULTS / "frontier.csv")
    plot_frontier(frontier, cfg, RESULTS / "frontier.png")
    fits, residual = [], []
    for res in cell_results:
        dataset, condition, seed, tau = res["cell"]
        tags = dict(dataset=dataset, condition=condition, seed=seed, tau=tau)
        fits.extend(dict(tags, **row) for row in res["eq_fits"])
        residual.extend(dict(tags, **row) for row in res["residual"])
    fits = pd.DataFrame(fits)
    write_csv(fits, RESULTS / "eq_fits.csv")
    residual = pd.DataFrame(residual)
    write_csv(residual, ARTIFACTS / "residual_replicates.csv")
    decomposition = residual_summary(residual)
    write_csv(decomposition, RESULTS / "residual_leakage.csv")
    battery_log = pd.DataFrame([dict({k: v for k, v in r["spec"].items() if k != "cuts"},
                                     status=r["status"], reason=r["reason"],
                                     train_windows=r["train_windows"],
                                     test_windows=r["test_windows"], seconds=r["seconds"],
                                     mlp_nonconverged=sum(not d.get("converged", True)
                                                          for d in r["diagnostics"]))
                                for r in battery_results])
    write_csv(battery_log, RESULTS / "battery_jobs.csv")
    reproduction = reproduction_check(battery_results)
    write_csv(reproduction, RESULTS / "reproduction_check.csv")
    write_eq_guarantee(RESULTS / "eq_guarantee.md")
    lp, sampled = lp_evidence()
    write_csv(lp, RESULTS / "eq_lp_synthetic.csv")
    write_csv(sampled, RESULTS / "eq_lp_synthetic_sampled.csv")
    return dict(
        label=LABEL, rows=len(summary), battery_jobs=len(battery_results),
        battery_not_evaluable=int((battery_log.status != "ok").sum()) if len(battery_log) else 0,
        eq_fit_max_equalization_residual=float(fits.equalization_residual.max()) if len(fits) else None,
        eq_fit_max_duality_gap=float(fits.duality_gap.max()) if len(fits) else None,
        lp_synthetic_max_equalization_residual=float(lp.equalization_residual.max()),
        lp_synthetic_max_abs_optimum_error=float(
            (lp.optimum - lp.expected_optimum).abs().max()),
        reproduction_max_abs_difference=float(reproduction.abs_difference.max())
        if len(reproduction) else None)


def residual_summary(residual):
    """Component means and the derived differences defined in residual_rows."""
    if residual.empty:
        return residual
    keys = ["dataset", "condition", "tau", "defence"]
    measures = [c for c in residual.columns if c.startswith(("bits_", "excess_", "utility_"))]
    mean = residual.groupby(keys + ["component"])[measures].mean().reset_index()
    wide = mean.pivot_table(index=keys, columns="component", values=measures)
    wide.columns = [f"{m}_{c}" for m, c in wide.columns]
    wide = wide.reset_index()
    for m in measures:
        if f"{m}_total" in wide and f"{m}_estimation" in wide:
            wide[f"{m}_total_minus_estimation"] = wide[f"{m}_total"] - wide[f"{m}_estimation"]
        if f"{m}_estimation" in wide and f"{m}_floor" in wide:
            wide[f"{m}_estimation_minus_floor"] = wide[f"{m}_estimation"] - wide[f"{m}_floor"]
    return wide


def write_eq_guarantee(path):
    """The EQ guarantee and the residual-leakage definitions, from their docstrings."""
    import inspect
    write_text(path, "\n".join([
        "# EQ guarantee and residual leakage (EXPLORATORY - Experiment K)", "",
        "Generated from the docstrings of beat/exp_k.py `eq_lp` and `residual_rows`.", "",
        "## eq_lp", "", "```text", inspect.getdoc(eq_lp), "```", "",
        "## residual_rows (results/exp_k/residual_leakage.csv)", "", "```text",
        inspect.getdoc(residual_rows), "```", ""]))


def record_drops(extra_drops):
    """Cuts applied after the projection join drop_list.csv (applied_at = power stage)."""
    path = RESULTS / "drop_list.csv"
    table = pd.read_csv(path)
    if "applied_at" not in table:
        table["applied_at"] = "projection"
    for cut in extra_drops:
        if cut not in set(table.cut):
            table.loc[len(table)] = dict(cut=cut, battery_jobs_removed=0,
                                         description=CUT_DESCRIPTION[cut],
                                         applied_at="power stage (--drop)")
    write_csv(table, path)


def reproduction_check(battery_results):
    """Battery cells that coincide with sealed L / C cells must reproduce their per-seed BA."""
    rows = []
    sealed_l = pd.read_csv(ROOT / "results/exp_l/identity_per_seed_direction.csv")
    sealed_l = sealed_l[(sealed_l.kind == "observation") & (sealed_l["mode"] == "command")]
    sealed_c = pd.read_csv(ROOT / "results/exp_c/observation_directional.csv")
    l_dir = {"1to2": "offline_to_online", "2to1": "online_to_offline", "cal_to_dep": "T_to_E"}
    for res in battery_results:
        spec = res["spec"]
        if res["status"] != "ok" or spec["defence"] != "none" or spec["view"] != "blind":
            continue
        for row in res["rows"]:
            if row["metric"] != "nway":
                continue
            attacker = row["attacker"].rsplit("_", 1)[0]
            mine = float(np.nanmean(row["recall"]))
            if spec["dataset"] == "lee2019" and spec["scenario"] in ("within_day",
                                                                     "cross_day_insample"):
                if spec["condition"] == "P2-RC_d8" and spec["tau"] == 0.5 and spec["k"] in (1, 20):
                    continue  # sealed C3 cells restrict the test support
                sel = sealed_l[(sealed_l.key == spec["condition"]) & np.isclose(sealed_l.tau, spec["tau"])
                               & (sealed_l.k == spec["k"]) & (sealed_l.attacker == attacker)
                               & (sealed_l.seed == spec["seed"])
                               & (sealed_l.direction == l_dir[spec["direction"]])]
            elif (spec["dataset"] == "bnci2014_001" and spec["scenario"] == "within_day"
                  and spec["k"] == 1 and spec["condition"] in ("P0", "P2-RC_d8")):
                cond = "P0" if spec["condition"] == "P0" else "P2-RC"
                sel = sealed_c[(sealed_c.condition == cond) & (sealed_c.seed == spec["seed"])
                               & (sealed_c.stream == f"command@{spec['tau']:g}")
                               & (sealed_c.k == 1) & (sealed_c.scenario == "main")
                               & (sealed_c.attacker == attacker)
                               & (sealed_c.direction == (0 if spec["direction"] == "1to2" else 1))]
            else:
                continue
            if len(sel) != 1:
                continue
            rows.append(dict(dataset=spec["dataset"], condition=spec["condition"],
                             tau=spec["tau"], seed=spec["seed"], scenario=spec["scenario"],
                             direction=spec["direction"], k=spec["k"], attacker=attacker,
                             k_ba=mine, sealed_ba=float(sel.ba.iloc[0]),
                             abs_difference=abs(mine - float(sel.ba.iloc[0]))))
    return pd.DataFrame(rows)


def observed_targets(curves_all, source, cfg):
    """Observed undefended Bayes N-way BA used to calibrate the effect model."""
    out = {}
    for scenario in ("within_day", "cross_day"):
        for attacker in cfg["power"]["calibration_attackers"]:
            for k in cfg["power"]["calibration_k"]:
                sel = curves_all[(curves_all.dataset == source["dataset"])
                                 & (curves_all.condition == source["condition"])
                                 & np.isclose(curves_all.tau, source["tau"])
                                 & (curves_all.defence == "none")
                                 & (curves_all.scenario == scenario)
                                 & (curves_all.enrolment == "all")
                                 & (curves_all.attacker == attacker) & (curves_all.k == k)
                                 & (curves_all.metric == "nway")]
                out[(scenario, attacker, k)] = float(sel.ba.iloc[0])
    return out


def power_analysis(cfg, extra_drops=()):
    """Calibrate the effect model on K's cross-fitted L streams, then M3/M4 power grids.

    extra_drops: cuts applied after the projection (recorded in power_summary.json)."""
    start = time.perf_counter()
    projection = json.loads((RESULTS / "runtime_projection.json").read_text(encoding="utf-8"))
    reduced = "power_grid_reduced" in list(projection["drop_list"]) + list(extra_drops)
    record_drops(extra_drops)
    p = cfg["power"]
    curves_all = pd.read_csv(RESULTS / "identity_curves_all.csv")
    sources = p["calibration_sources"][:1] if reduced else p["calibration_sources"]
    effects, calib_tables, fit_tables, m3_tables = {}, [], [], []
    seed0 = cfg["model_seeds"][0]
    for source in sources:
        name = f"{source['dataset']}|{source['condition']}|tau{source['tau']:g}"
        streams, _ = build_streams(cfg, datasets=[source["dataset"]],
                                   conditions=[source["condition"]], seeds=[seed0], write=False)
        cell = (source["dataset"], source["condition"], seed0, source["tau"])
        theta = theta_pool(streams[cell + ("cal_oof",)], streams[cell + ("dep",)])
        observed = observed_targets(curves_all, source, cfg)
        best, table = calibrate_effect(theta, observed, cfg, rng_for(cfg["seed"], "calib", name)
                                       .integers(2 ** 31))
        fit_tables.append(calibration_fit(table, observed, best, name))
        table.insert(0, "source", name)
        calib_tables.append(table)
        effects[name] = dict(source=source, theta=theta, **best,
                             observed={"|".join(map(str, k)): v for k, v in observed.items()},
                             users_in_pool=len(theta))
        _log(f"calibrated {name}: {best}")
        for mult in p["effect_multipliers"]:
            effect = dict(theta=theta, shrink=best["shrink"] * mult, drift_sd=best["drift_sd"])
            m3, _ = power_function(effect, n_sim=p["n_sim"], seed=cfg["seed"],
                                   k_grid=p["k_grid"])
            m3.insert(0, "source", name)
            m3.insert(1, "effect_multiplier", mult)
            m3_tables.append(m3)
    write_csv(pd.concat(calib_tables), RESULTS / "power_calibration_grid.csv")
    write_csv(pd.concat(fit_tables), RESULTS / "power_calibration_fit.csv")
    # theta holds -inf structural zeros, which JSON cannot carry: it goes to an .npz.
    np.savez(RESULTS / "power_theta.npz", **{name: e["theta"] for name, e in effects.items()})
    write_json(RESULTS / "power_effects.json", dict(
        label=LABEL, model="simulate_days", theta_file="power_theta.npz",
        calibration_scope=CALIBRATION_SCOPE, calibration_fit="power_calibration_fit.csv",
        effects={name: {k: v for k, v in e.items() if k != "theta"}
                 for name, e in effects.items()}))
    m3 = pd.concat(m3_tables, ignore_index=True)
    write_csv(m3, RESULTS / "power_m3.csv")
    primary = next(iter(effects))
    m4_tables = []
    theta = np.asarray(effects[primary]["theta"])
    for mult in p["effect_multipliers"]:
        effect = dict(theta=theta, shrink=effects[primary]["shrink"] * mult,
                      drift_sd=effects[primary]["drift_sd"])
        _, m4 = power_function(effect, n_sim=p["n_sim"] // (2 if reduced else 1),
                               seed=cfg["seed"], k_grid=[10, 20, 50], g_grid=p["g_grid"],
                               m_grid=p["m_grid"], delta_grid=p["delta_grid"],
                               eq_variants=cfg["defences"]["eq_variants"])
        m4.insert(0, "source", primary)
        m4.insert(1, "effect_multiplier", mult)
        m4_tables.append(m4)
    m4 = pd.concat(m4_tables, ignore_index=True)
    write_csv(m4, RESULTS / "power_m4.csv")
    plot_power(m3, m4, RESULTS / "power.png")
    decision_rule_power(cfg, m3, m4)
    return dict(seconds=time.perf_counter() - start, reduced=reduced,
                extra_drops=list(extra_drops), effects={k: {kk: vv for kk, vv in v.items()
                                                               if kk != "theta"}
                                                           for k, v in effects.items()})


def decision_rule_power(cfg, m3, m4):
    """Rows of the power grids at the configured decision rule (no extra simulation): M3 at
    k = K*, M4 at (eq_variant, g*, K*, m, Delta), for the source at the rule's tau."""
    rule = cfg["decision_rule"]
    effects = json.loads((RESULTS / "power_effects.json").read_text(encoding="utf-8"))["effects"]
    primary = next(name for name, eff in effects.items() if eff["source"]["tau"] == rule["tau"])
    m3_rows = m3[(m3.source == primary) & (m3.k == rule["k_star"])].assign(claim="M3")
    m4_rows = m4[(m4.source == primary) & (m4.eq_variant == rule["eq_variant"])
                 & (m4.g == rule["g_star"]) & (m4.k == rule["k_star"])
                 & np.isclose(m4.m, rule["m"]) & np.isclose(m4.delta, rule["delta"])
                 ].assign(claim="M4")
    table = pd.concat([m3_rows, m4_rows], ignore_index=True)
    table.insert(0, "label", "exploratory")
    write_csv(table, RESULTS / "power_decision_rule.csv")
    return table


def plot_power(m3, m4, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (source, mult), part in m3.groupby(["source", "effect_multiplier"]):
        axes[0].plot(part.k, part.power, marker="o", label=f"{source} x{mult:g}")
    axes[0].axhline(0.8, color="grey", linestyle=":")
    axes[0].set(xscale="log", xlabel="K*", ylabel="power (alpha / Holm family)",
                title="M3: k = K* exceeds 2/N and k = 1 (cross-day)")
    axes[0].legend(fontsize=6)
    sel = m4[(m4.eq_variant == "confusion") & (m4.k == 20)]
    for (mult, g, delta), part in sel.groupby(["effect_multiplier", "g", "delta"]):
        axes[1].plot(part.m, part.power, marker=".", label=f"x{mult:g} g={g} delta={delta:g}")
    axes[1].axhline(0.8, color="grey", linestyle=":")
    axes[1].set(xlabel="m", ylabel="power", title="M4: EQ-confusion at k = 20")
    axes[1].legend(fontsize=5, ncol=2)
    fig.suptitle(LABEL, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pilot", "reproject", "run", "cells", "power",
                                          "decision_rule"])
    parser.add_argument("--drop", action="append", default=[], choices=list(CUT_DESCRIPTION),
                        help="extra cut for the power stage (recorded in power_summary.json)")
    args = parser.parse_args(argv)
    if Path.cwd().resolve() != ROOT.resolve():
        raise SystemExit(f"Run from {ROOT}")
    cfg = load_cfg()
    RESULTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if args.stage == "pilot":
        print(json.dumps(pilot(cfg), indent=1, default=str))
    elif args.stage == "reproject":
        print(json.dumps(reproject(cfg), indent=1, default=str))
    elif args.stage == "run":
        run(cfg)
        write_json(RESULTS / "power_summary.json", power_analysis(cfg))
    elif args.stage == "cells":
        rerun_cells(cfg)
    elif args.stage == "power":
        write_json(RESULTS / "power_summary.json", power_analysis(cfg, args.drop))
    else:
        decision_rule_power(cfg, pd.read_csv(RESULTS / "power_m3.csv"),
                            pd.read_csv(RESULTS / "power_m4.csv"))


if __name__ == "__main__":
    # Re-enter through the importable module so worker functions pickle by reference
    # (functions defined in __main__ are pickled by value and lose module globals).
    from beat.exp_k import main as _main
    _main()
