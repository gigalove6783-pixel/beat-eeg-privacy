"""Experiment C (H3): replayed virtual one-joint arm, transmission modes and the
observation-length identity attack on the frozen Experiment-B models.

    python -m beat.exp_c dev    # session T only: temperatures, rejection rule, T replay smoke run
    python -m beat.exp_c final  # session E replay and attacks; only after the E seal
    ... --variant sealed_gate   # same with the sealed Riemannian-distance gate
                                # into results/exp_c/sealed_gate/ (post-seal deviation 7)

Commands: -1 left (-step), +1 right (+step), 0 hold. Labels build and score the
target trajectory only; they never reach `decide` or `Arm`. Processing times
are replay timings of recorded trials on this CPU, not online latency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from joblib import Parallel, delayed, parallel_config  # noqa: E402
from pyriemann.geometry.distance import distance_riemann  # noqa: E402
from pyriemann.geometry.mean import mean_riemann  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.signal import sosfilt, sosfilt_zi  # noqa: E402
from scipy.special import expit, logsumexp  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from .attacks import ATTACKERS, fit_attackers  # noqa: E402
from .config import ROOT, load_config  # noqa: E402
from .data import (META_COLUMNS, Trials, _mi_runs, bandpass_sos,  # noqa: E402
                   load_trials, require_training)
from .features import covariances  # noqa: E402
from .splits import E_HALVES  # noqa: E402

RESULTS = ROOT / "results" / "exp_c"
ARTIFACTS = ROOT / "artifacts" / "exp_c"
OOF_MANIFEST = ROOT / "artifacts" / "exp_b" / "oof" / "manifest.json"
LEFT, HOLD, RIGHT = -1, 0, 1
KEY = ["subject", "session", "run", "trial"]
RUN = ["subject", "session", "run"]
METRICS = ["mae_deg", "reach_rate", "wrong_direction_moves", "out_of_range_commands",
           "hold_rate", "effective_moves", "unnecessary_movement_deg"]
TIMING_NOTE = ("Replay timing of recorded trials on this CPU (filter = causal SOS filtering "
               "of the trial window's raw samples streamed with filter state carried "
               "through the run, plus float32 cast; samples between windows are streamed "
               "outside the timed path; inference = covariance, rejection rule and frozen "
               "model; control = temperature, tau decision and arm step); not online "
               "latency.")


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _signals_hash(trials):
    digest = hashlib.sha256(np.ascontiguousarray(trials.X).tobytes())
    digest.update(trials.meta[META_COLUMNS].to_csv(index=False).encode())
    return digest.hexdigest()


def _sorted(trials):
    """Recorded order: subject, session, run, chronological trial position."""
    meta = trials.meta.reset_index(drop=True)
    return trials.subset(meta.sort_values(KEY, kind="stable").index.to_numpy())


def stream_name(mode, tau=None):
    return "feature" if mode == "feature" else f"command@{tau:g}"


# ---------------------------------------------------------------- calibration

def fit_temperature(outputs: Trials, bounds, tolerance=1e-6) -> float:
    """NLL-optimal temperature of T out-of-fold logits (outputs.X = [left, right])."""
    require_training(outputs)
    logits = np.asarray(outputs.X, dtype=float)
    finite = np.isfinite(logits).all(axis=1)
    z, y = logits[finite], outputs.y[finite]

    def nll(log_t):
        s = z / np.exp(log_t)
        return float(np.mean(logsumexp(s, axis=1) - s[np.arange(len(s)), y]))

    result = minimize_scalar(nll, bounds=tuple(np.log(bounds)), method="bounded",
                             options={"xatol": tolerance})
    return float(np.exp(result.x))


def _nll(logits, labels, temperature):
    s = np.asarray(logits, dtype=float) / temperature
    return float(np.mean(logsumexp(s, axis=1) - s[np.arange(len(s)), labels]))


@dataclass
class RejectionRule:
    """Per-subject Riemannian-distance gate fitted on that subject's T covariances only."""
    z: float
    references: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)

    def fit(self, covs: Trials):
        require_training(covs)
        usable = np.isfinite(covs.X).all(axis=(1, 2)) & ~covs.meta["artifact"].to_numpy(bool)
        for subject in np.unique(covs.subject):
            use = usable & (covs.subject == subject)
            reference = mean_riemann(covs.X[use])
            distance = np.atleast_1d(distance_riemann(covs.X[use], reference))
            median = np.median(distance)
            mad = 1.4826 * np.median(np.abs(distance - median))
            self.references[int(subject)] = reference
            self.thresholds[int(subject)] = float(median + self.z * mad)
        return self

    def distances(self, covs: Trials) -> np.ndarray:
        out = np.full(len(covs), np.nan)
        finite = np.isfinite(covs.X).all(axis=(1, 2))
        for subject in np.unique(covs.subject[finite]):
            idx = finite & (covs.subject == subject)
            out[idx] = distance_riemann(covs.X[idx], self.references[int(subject)])
        return out

    def accept(self, covs: Trials) -> np.ndarray:
        """True = trial may command; non-finite or beyond-threshold trials are held."""
        distance = self.distances(covs)
        threshold = np.array([self.thresholds[int(s)] for s in covs.subject])
        return np.isfinite(distance) & (distance <= threshold)

    def to_dict(self):
        return dict(gate="riemann_distance", z=self.z,
                    thresholds={str(s): t for s, t in self.thresholds.items()},
                    references={str(s): r.tolist() for s, r in self.references.items()})

    @classmethod
    def from_dict(cls, value):
        return cls(value["z"], {int(s): np.asarray(r) for s, r in value["references"].items()},
                   {int(s): float(t) for s, t in value["thresholds"].items()})


@dataclass
class FiniteGate:
    """Gate `none`: only non-finite trials are held (tau holds are applied separately)."""
    thresholds: dict = field(default_factory=dict)

    def fit(self, covs: Trials):
        require_training(covs)
        return self

    def accept(self, covs: Trials) -> np.ndarray:
        return np.isfinite(covs.X).all(axis=(1, 2))

    def to_dict(self):
        return dict(gate="none")


def make_gate(rejection):
    if rejection["gate"] == "riemann_distance":
        return RejectionRule(rejection["z"])
    if rejection["gate"] == "none":
        return FiniteGate()
    raise ValueError(f"Unknown gate {rejection['gate']!r}")


def gate_from_dict(value):
    return FiniteGate() if value["gate"] == "none" else RejectionRule.from_dict(value)


@dataclass
class ReceiverHead:
    """Affine map representation -> [left, right] logits held by the feature-mode receiver."""
    weights: np.ndarray

    def __call__(self, representation):
        representation = np.atleast_2d(representation)
        return representation @ self.weights[:-1] + self.weights[-1]


def identify_head(outputs: Trials, logits, tolerance) -> tuple[ReceiverHead, float]:
    """Recover the frozen model's own affine task head from its T outputs (no labels).

    B's logits are an affine function of the representation (linear head on the
    bottleneck; logistic regression on the standardized tangent vector), so the
    solution is exact up to float rounding; any larger residual is an error.
    """
    require_training(outputs)
    rep, logits = np.asarray(outputs.X, dtype=float), np.asarray(logits, dtype=float)
    ok = np.isfinite(rep).all(axis=1) & np.isfinite(logits).all(axis=1)
    design = np.column_stack([rep[ok], np.ones(ok.sum())])
    weights = np.linalg.lstsq(design, logits[ok], rcond=None)[0]
    head = ReceiverHead(weights)
    error = float(np.max(np.abs(head(rep[ok]) - logits[ok])))
    if error > tolerance:
        raise ValueError(f"Logits are not affine in the representation (error {error:.3g})")
    return head, error


# ---------------------------------------------------------------- commands and arm

def decide(logits, temperature, tau, gate=None) -> np.ndarray:
    """Temperature-scaled max probability below tau -> hold; gate False or NaN -> hold."""
    logits = np.atleast_2d(np.asarray(logits, dtype=float))
    ok = np.isfinite(logits).all(axis=1)
    if gate is not None:
        ok &= np.asarray(gate, dtype=bool)
    margin = np.where(ok, logits[:, 1] - logits[:, 0], 0.0) / temperature
    move = ok & (expit(np.abs(margin)) >= tau)
    return np.where(move, np.where(margin > 0, RIGHT, LEFT), HOLD).astype(np.int8)


class Arm:
    """Label-free one-joint controller; blocked moves leave the angle unchanged."""

    def __init__(self, step_deg, range_deg):
        self.increment, self.limit, self.angle = float(step_deg), float(range_deg), 0.0

    def reset(self):
        self.angle = 0.0

    def step(self, command):
        proposed = self.angle + self.increment * command
        if command != HOLD and abs(proposed) > self.limit:
            return self.angle, True
        self.angle = proposed
        return self.angle, False


def episode_layout(meta, length):
    """Block index within each run, complete-block flag and block-start flag (meta in recorded order)."""
    position = meta.groupby(RUN, sort=False).cumcount().to_numpy()
    size = meta.groupby(RUN, sort=False)["trial"].transform("size").to_numpy()
    block = position // length
    return block, (block + 1) * length <= size, position % length == 0


def _drive(commands, starts, controller):
    arm = Arm(controller["step_deg"], controller["range_deg"])
    angles = np.zeros(len(commands))
    for i, command in enumerate(commands):
        if starts[i]:
            arm.reset()
        angles[i] = arm.step(int(command))[0]
    return angles


def control_pass(scores, temperature, tau, starts, controller, gate=None, head=None):
    """Per-trial decision + arm step in recorded order, timed per trial (ms).

    Command mode: scores are the sender's logits. Feature mode: scores are the
    received representations and `head` derives the logits at the receiver.
    """
    n = len(scores)
    gate = np.ones(n, dtype=bool) if gate is None else np.asarray(gate, dtype=bool)
    arm = Arm(controller["step_deg"], controller["range_deg"])
    commands = np.zeros(n, dtype=np.int8)
    angles, blocked, elapsed = np.zeros(n), np.zeros(n, dtype=bool), np.zeros(n)
    for i in range(n):
        start = time.perf_counter_ns()
        if starts[i]:
            arm.reset()
        logits = scores[i:i + 1] if head is None else head(scores[i:i + 1])
        command = int(decide(logits, temperature, tau, gate[i:i + 1])[0])
        angles[i], blocked[i] = arm.step(command)
        elapsed[i] = time.perf_counter_ns() - start
        commands[i] = command
    return commands, angles, blocked, elapsed / 1e6


def score_episodes(meta, commands, angles, blocked, gated, controller):
    """Episode metrics on complete blocks; labels enter only here (target and scoring)."""
    block, complete, starts = episode_layout(meta, controller["episode_length"])
    direction = np.where(meta["label"].to_numpy() == 1, RIGHT, LEFT)
    target = _drive(direction, starts, controller)
    previous = np.where(starts, 0.0, np.roll(angles, 1))
    moved = angles != previous
    useless = moved & (np.abs(angles - target) >= np.abs(previous - target))
    frame = pd.DataFrame(dict(
        subject=meta["subject"].to_numpy(), session=meta["session"].to_numpy(),
        run=meta["run"].to_numpy(), block=block, error=np.abs(angles - target),
        wrong=moved & (commands != direction), blocked=blocked,
        hold=commands == HOLD, moved=moved,
        unnecessary=np.where(useless, np.abs(angles - previous), 0.0),
        gated=np.asarray(gated, dtype=bool)))[complete]
    episodes = frame.groupby(RUN + ["block"], sort=True).agg(
        mae_deg=("error", "mean"), end_error_deg=("error", "last"),
        wrong_direction_moves=("wrong", "sum"), out_of_range_commands=("blocked", "sum"),
        hold_rate=("hold", "mean"), effective_moves=("moved", "sum"),
        unnecessary_movement_deg=("unnecessary", "sum"), gated_holds=("gated", "sum"),
        trials=("error", "size")).reset_index()
    episodes["reached"] = episodes.end_error_deg <= controller["reach_tolerance_deg"]
    return episodes


def subject_metrics(episodes, tags):
    """Per subject: episode means (counts are per episode), reach rate over episodes."""
    return episodes.groupby(tags + ["subject"], sort=True).agg(
        mae_deg=("mae_deg", "mean"), reach_rate=("reached", "mean"),
        wrong_direction_moves=("wrong_direction_moves", "mean"),
        out_of_range_commands=("out_of_range_commands", "mean"),
        hold_rate=("hold_rate", "mean"), effective_moves=("effective_moves", "mean"),
        unnecessary_movement_deg=("unnecessary_movement_deg", "mean"),
        gated_holds=("gated_holds", "sum"), episodes=("block", "size")).reset_index()


def summarize_control(metrics, tags):
    """Seed mean per subject, then 9-subject mean / median / std."""
    per_subject = metrics.groupby(tags + ["subject"])[METRICS].mean().reset_index()
    summary = per_subject.groupby(tags)[METRICS].agg(["mean", "median", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def summarize_timing(trials, tags):
    rows = []
    for key, group in trials.groupby(tags, sort=True):
        row = dict(zip(tags, key if isinstance(key, tuple) else (key,)), trials=len(group))
        for stage in ("filter", "inference", "control", "total"):
            values = group[f"{stage}_ms"].to_numpy()
            row[f"{stage}_median_ms"] = float(np.median(values))
            row[f"{stage}_p95_ms"] = float(np.quantile(values, 0.95))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- replay

def trial_covariances(signals: Trials, estimator):
    """Batch covariances; trials flagged non-finite get NaN (they become holds)."""
    n, c = len(signals), signals.X.shape[1]
    X = np.full((n, c, c), np.nan)
    valid = signals.meta["finite"].to_numpy(bool) & np.isfinite(signals.X).all(axis=(1, 2))
    if valid.any():
        X[valid] = covariances(signals.subset(valid), estimator).X
    return signals.with_X(X)


def _run_key(subject, session, run):
    return int(subject), str(session), int(run)


def load_raw_runs(meta, data_config):
    """Raw EEG (samples x channels) of every run in `meta` and each trial's window start.

    Only the subject/session files present in `meta` are opened.
    """
    fs = data_config["sfreq"]
    offset = int(round((data_config["cue_after_trial_start_s"]
                        + data_config["window_after_cue_s"][0]) * fs))
    out = {}
    for (subject, session), frame in meta.groupby(["subject", "session"], sort=True):
        runs = _mi_runs(ROOT / data_config["raw_dir"] / f"A{int(subject):02d}{session}.mat")
        for run_no in sorted(frame["run"].unique()):
            run = runs[int(run_no) - 1]
            onsets = np.atleast_1d(run.trial).astype(int) - 1  # MATLAB indices are 1-based
            raw = np.asarray(run.X, dtype=np.float64)[:, :data_config["n_eeg_channels"]]
            out[_run_key(subject, session, run_no)] = (
                raw, {t: int(a) + offset for t, a in enumerate(onsets, start=1)})
    return out


def _clean(chunk):
    # Same handling as beat.data: non-finite raw samples enter the filter as zeros.
    return np.where(np.isfinite(chunk), chunk, 0.0)


def replay_session(signals: Trials, raw_runs, rule, sos, estimator, model=None):
    """Live per-trial pass in recorded order (ms per trial).

    filter: causal SOS filtering of the trial window's raw samples, with filter
    state carried through the whole run (samples between windows are streamed
    outside the timed path), cast to float32 like the cached trials; inference:
    covariance, rejection gate and frozen model. Also returns each streamed
    window's max |difference| to the cached window relative to the cached max |value|.
    """
    n, c, length = signals.X.shape
    meta = signals.meta
    rows = [signals.subset([i]) for i in range(n)]  # acquisition, outside the timed path
    keys = [_run_key(*key) for key in meta[RUN].itertuples(index=False)]
    trial_no, finite = meta["trial"].to_numpy(int), meta["finite"].to_numpy(bool)
    gate = np.zeros(n, dtype=bool)
    filter_ms, inference_ms, window_error = np.zeros(n), np.zeros(n), np.zeros(n)
    representation = logits = None
    current = zi = raw = starts = None
    position = 0
    for i, row in enumerate(rows):
        if keys[i] != current:
            current = keys[i]
            raw, starts = raw_runs[current]
            zi = sosfilt_zi(sos)[:, :, None] * _clean(raw[0])[None, None, :]
            position = 0
        a = starts[trial_no[i]]
        if a < position:
            raise ValueError(f"Overlapping windows in run {current}")
        if a > position:
            _, zi = sosfilt(sos, _clean(raw[position:a]), axis=0, zi=zi)
        start = time.perf_counter_ns()
        filtered, zi = sosfilt(sos, _clean(raw[a:a + length]), axis=0, zi=zi)
        window = filtered.T.astype(np.float32)[None]
        middle = time.perf_counter_ns()
        if finite[i] and np.isfinite(window).all():
            cov = covariances(row.with_X(window), estimator)
        else:
            cov = row.with_X(np.full((1, c, c), np.nan))
        gate[i] = rule.accept(cov)[0]
        if model is not None:
            z, scores = model(cov)
            if representation is None:
                representation = np.full((n, z.shape[1]), np.nan)
                logits = np.full((n, 2), np.nan)
            representation[i], logits[i] = z[0], scores[0]
        end = time.perf_counter_ns()
        filter_ms[i], inference_ms[i] = (middle - start) / 1e6, (end - middle) / 1e6
        position = a + length
        cached = np.asarray(row.X[0], dtype=np.float64)
        window_error[i] = (np.max(np.abs(window[0] - cached))
                           / max(float(np.max(np.abs(cached))), np.finfo(np.float32).tiny))
    return dict(gate=gate, representation=representation, logits=logits,
                filter_ms=filter_ms, inference_ms=inference_ms, window_error=window_error)


def stream_check(window_error, tolerance):
    return dict(stream_trials=int(len(window_error)),
                stream_window_max_rel_error=float(np.max(window_error, initial=0.0)),
                stream_matches_cache=bool(np.max(window_error, initial=0.0) <= tolerance))


def _align(saved, meta, required=True):
    """Row index into a saved B mapping for each meta row (by subject/session/run/trial)."""
    frame = pd.DataFrame({c: saved[c] for c in KEY})
    frame["session"] = frame["session"].astype(str)
    frame["row"] = np.arange(len(frame))
    left = meta[KEY].copy()
    left["session"] = left["session"].astype(str)
    merged = left.merge(frame, on=KEY, how="left", validate="one_to_one")
    if required and merged.row.isna().any():
        raise ValueError("Saved B outputs lack replayed trials")
    return merged.row.to_numpy()


# ---------------------------------------------------------------- observation attack

def observation_groups(meta, k, stride):
    """Index arrays of k consecutive outputs within one run; meta in recorded order."""
    groups, rows = [], []
    for (subject, session, run), frame in meta.reset_index(drop=True).groupby(RUN, sort=True):
        idx = frame.index.to_numpy()
        for start in range(0, len(idx) - k + 1, stride):
            groups.append(idx[start:start + k])
            rows.append((subject, session, run))
    return (np.asarray(groups, dtype=int).reshape(-1, k),
            pd.DataFrame(rows, columns=RUN))


def feature_descriptors(messages, groups, min_valid=1):
    """Mean and standard deviation of the non-null representations in each group."""
    windows = messages[groups]
    valid = np.isfinite(windows).all(axis=2).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        descriptors = np.hstack([np.nanmean(windows, axis=1), np.nanstd(windows, axis=1)])
    return descriptors, valid >= min_valid


def command_descriptors(commands, groups, min_valid=None):
    """Hold rate, left/right bias, mean/max run length, 3x3 transition frequencies."""
    w = np.asarray(commands, dtype=int)[groups]
    k = w.shape[1]
    left, right = (w == LEFT).sum(axis=1), (w == RIGHT).sum(axis=1)
    bias = np.where(left + right > 0, (right - left) / np.maximum(left + right, 1), 0.0)
    runs = 1 + (np.diff(w, axis=1) != 0).sum(axis=1)
    longest = np.ones(len(w))
    current = np.ones(len(w))
    for j in range(1, k):
        current = np.where(w[:, j] == w[:, j - 1], current + 1, 1)
        longest = np.maximum(longest, current)
    transitions = [((w[:, :-1] == a) & (w[:, 1:] == b)).sum(axis=1) / max(k - 1, 1)
                   for a in (LEFT, HOLD, RIGHT) for b in (LEFT, HOLD, RIGHT)]
    descriptors = np.column_stack([(w == HOLD).mean(axis=1), bias, k / runs, longest,
                                   *transitions])
    return descriptors, np.ones(len(w), dtype=bool)


def _attack_job(tags, mode, k, stream_t, stream_e, meta_t, meta_e, attackers, seed,
                train_stride, min_valid):
    describe = feature_descriptors if mode == "feature" else command_descriptors

    def build(stream, meta, stride, runs=None):
        groups, info = observation_groups(meta, k, stride)
        X, valid = describe(stream, groups, min_valid)
        keep = np.ones(len(valid), dtype=bool) if runs is None else info.run.isin(runs).to_numpy()
        excluded = int((keep & ~valid).sum())
        keep = keep & valid
        return X[keep], info[keep].reset_index(drop=True), excluded

    scenarios = [("main", d, build(stream_e, meta_e, train_stride, fit),
                  build(stream_e, meta_e, k, test)) for d, (fit, test) in enumerate(E_HALVES)]
    scenarios.append(("auxiliary", 0, build(stream_t, meta_t, train_stride),
                      build(stream_e, meta_e, k)))
    identities = np.unique(meta_e.subject)
    predictions, directional = [], []
    with threadpool_limits(limits=1):
        for scenario, direction, (train_X, train_info, train_excluded), \
                (test_X, test_info, test_excluded) in scenarios:
            counts = train_info.subject.value_counts().reindex(identities, fill_value=0)
            short = counts[counts < attackers["inner_folds"]]
            if len(short):
                # fit_attackers requires inner_folds examples per identity; the cell is
                # recorded as not evaluable instead of tuned some other way.
                directional.extend(dict(
                    **tags, k=k, scenario=scenario, attacker=attacker, direction=direction,
                    groups=len(test_info), train_groups=len(train_info),
                    excluded_groups=test_excluded, excluded_train_groups=train_excluded,
                    evaluable=False,
                    short_identities=";".join(f"{s}:{n}" for s, n in short.items()),
                    ba=np.nan, macro_f1=np.nan) for attacker in ATTACKERS)
                continue
            fitted = fit_attackers(train_X, train_info.subject.to_numpy(), attackers, seed)
            for attacker, estimator in fitted.items():
                predicted = estimator.predict(test_X)
                frame = test_info[["subject", "run"]].copy()
                frame["prediction"] = predicted
                frame = frame.assign(**tags, k=k, scenario=scenario, attacker=attacker,
                                     direction=direction)
                predictions.append(frame)
                directional.append(dict(
                    **tags, k=k, scenario=scenario, attacker=attacker, direction=direction,
                    groups=len(test_info), train_groups=len(train_info),
                    excluded_groups=test_excluded, excluded_train_groups=train_excluded,
                    evaluable=True, short_identities="",
                    ba=float(balanced_accuracy_score(test_info.subject, predicted)),
                    macro_f1=float(f1_score(test_info.subject, predicted, average="macro",
                                            zero_division=0))))
    return (pd.concat(predictions, ignore_index=True) if predictions else None), directional


def bootstrap_seed_mean(frames, samples, seed, confidence):
    """Percentile interval of the seed-mean BA, resampling runs within subject.

    All seeds predict the same recorded runs, so each replicate draws one set of
    run multiplicities per (direction, subject) test set and applies it to every
    seed before averaging. Directions (main scenario) are equally weighted, as in
    the point estimate.
    """
    rng = np.random.default_rng(seed)
    identities = np.unique(np.concatenate(
        [np.r_[f.subject.to_numpy(), f.prediction.to_numpy()] for f in frames]))
    lookup = {s: i for i, s in enumerate(identities)}
    m = len(identities)
    units = {}
    for frame in frames:
        for key, group in frame.groupby(["direction", "subject"], sort=True):
            units.setdefault(key, set()).update(group.run.unique())
    multiplicity = {}
    for key in sorted(units):
        runs = np.array(sorted(units[key]))
        draw = rng.integers(len(runs), size=(samples, len(runs)))
        counts = np.zeros((samples, len(runs)))
        np.add.at(counts, (np.arange(samples)[:, None], draw), 1)
        multiplicity[key] = runs, counts
    scores = []
    for frame in frames:
        directions = []
        for direction, part in frame.groupby("direction", sort=True):
            confusion = np.zeros((samples, m, m))
            present = []
            for subject, group in part.groupby("subject", sort=True):
                runs, counts = multiplicity[(direction, subject)]
                table = np.zeros((len(runs), m))
                for run, unit in group.groupby("run", sort=True):
                    table[np.searchsorted(runs, run)] = np.bincount(
                        unit.prediction.map(lookup), minlength=m)
                confusion[:, lookup[subject]] = counts @ table
                present.append(lookup[subject])
            rows = np.asarray(present)
            recall = confusion[:, rows, rows] / confusion[:, rows, :].sum(axis=2)
            directions.append(recall.mean(axis=1))
        scores.append(np.mean(directions, axis=0))
    total = np.mean(scores, axis=0)
    low, high = np.quantile(total, [(1 - confidence) / 2, (1 + confidence) / 2])
    return float(low), float(high)


def summarize_observation(directional, predictions, streams, config):
    """Seed-mean BA per scenario/attacker; strongest = max over attackers and scenarios.

    A scenario cell is not evaluable when any of its directions or seeds is; the
    strongest attack is the maximum over the evaluable cells only.
    """
    obs, stats = config["observation"], config["statistics"]
    scenario = directional.groupby(["condition", "seed", "stream", "k", "scenario", "attacker"],
                                   as_index=False).agg(
        ba=("ba", "mean"), macro_f1=("macro_f1", "mean"), groups=("groups", "sum"),
        excluded_groups=("excluded_groups", "sum"), evaluable=("evaluable", "all"))
    scenario.loc[~scenario.evaluable, ["ba", "macro_f1"]] = np.nan
    seed_mean = scenario.groupby(["condition", "stream", "k", "scenario", "attacker"],
                                 as_index=False).agg(
        ba_mean=("ba", "mean"), ba_seed_std=("ba", "std"),
        macro_f1_mean=("macro_f1", "mean"), seeds=("seed", "nunique"),
        groups_mean=("groups", "mean"), excluded_groups_mean=("excluded_groups", "mean"),
        evaluable=("evaluable", "all"))
    seed_mean.loc[~seed_mean.evaluable, ["ba_mean", "ba_seed_std", "macro_f1_mean"]] = np.nan
    # Feature groups whose k messages are all null carry no descriptor and are
    # dropped; feature BA is conditional on the retained groups.
    exclusions = seed_mean[seed_mean.stream == "feature"].drop_duplicates(
        ["condition", "k", "scenario"])[["condition", "k", "scenario", "groups_mean",
                                         "excluded_groups_mean"]].reset_index(drop=True)
    exclusions["excluded_fraction"] = exclusions.excluded_groups_mean / (
        exclusions.groups_mean + exclusions.excluded_groups_mean)
    index = predictions.groupby(["condition", "stream", "k", "scenario", "attacker", "seed"],
                                sort=True).indices
    target = obs["threshold_factor"] * obs["chance"]
    strongest = []
    for (condition, stream, k), group in seed_mean.groupby(["condition", "stream", "k"],
                                                           sort=True):
        not_evaluable = int((~group.evaluable).sum())
        group = group[group.evaluable]
        if group.empty:
            strongest.append(dict(
                condition=condition, stream=stream, **streams[stream], k=int(k),
                scenario=None, attacker=None, ba_mean=np.nan, ba_low=np.nan, ba_high=np.nan,
                ba_seed_std=np.nan, macro_f1_mean=np.nan, seeds=0, groups_mean=np.nan,
                excluded_groups_mean=np.nan, not_evaluable_cells=not_evaluable,
                reaches_threshold=False, lower_bound_reaches_threshold=False))
            continue
        best = group.sort_values(["ba_mean", "scenario", "attacker"],
                                 ascending=[False, True, True]).iloc[0]
        frames = [predictions.iloc[idx] for key, idx in index.items()
                  if key[:5] == (condition, stream, k, best.scenario, best.attacker)]
        low, high = bootstrap_seed_mean(frames, stats["bootstrap_samples"], stats["seed"],
                                        stats["confidence"])
        strongest.append(dict(
            condition=condition, stream=stream, **streams[stream], k=int(k),
            scenario=best.scenario, attacker=best.attacker, ba_mean=float(best.ba_mean),
            ba_low=low, ba_high=high, ba_seed_std=float(best.ba_seed_std),
            macro_f1_mean=float(best.macro_f1_mean), seeds=int(best.seeds),
            groups_mean=float(best.groups_mean),
            excluded_groups_mean=float(best.excluded_groups_mean),
            not_evaluable_cells=not_evaluable,
            reaches_threshold=bool(best.ba_mean >= target - 1e-12),
            lower_bound_reaches_threshold=bool(low >= target - 1e-12)))
    strongest = pd.DataFrame(strongest)
    threshold = []
    for (condition, stream), group in strongest.groupby(["condition", "stream"], sort=True):
        group = group.sort_values("k")
        hit, hit_low = group[group.reaches_threshold], group[group.lower_bound_reaches_threshold]
        threshold.append(dict(
            condition=condition, stream=stream, **streams[stream], threshold_ba=target,
            smallest_k=int(hit.k.iloc[0]) if len(hit) else None,
            smallest_k_lower_bound=int(hit_low.k.iloc[0]) if len(hit_low) else None,
            max_k_tested=int(group.k.max())))
    return scenario, seed_mean, strongest, pd.DataFrame(threshold), exclusions


def plot_observation(strongest, config, path):
    conditions = list(dict.fromkeys(strongest.condition))
    columns = min(3, len(conditions))
    rows = int(np.ceil(len(conditions) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 3.6 * rows),
                             sharex=True, sharey=True, squeeze=False)
    streams = list(dict.fromkeys(strongest.sort_values(["mode", "tau"],
                                                       ascending=[False, True]).stream))
    colors = dict(zip(streams, plt.cm.viridis(np.linspace(0, 0.85, len(streams)))))
    chance = config["observation"]["chance"]
    for ax, condition in zip(axes.flat, conditions):
        for stream in streams:
            sub = strongest[(strongest.condition == condition)
                            & (strongest.stream == stream)].sort_values("k")
            ax.errorbar(sub.k, sub.ba_mean, yerr=[sub.ba_mean - sub.ba_low,
                                                  sub.ba_high - sub.ba_mean],
                        marker="o" if stream == "feature" else ".", capsize=3,
                        color="black" if stream == "feature" else colors[stream],
                        linewidth=2 if stream == "feature" else 1, label=stream)
        ax.axhline(chance, color="grey", linestyle=":", label="chance 1/9")
        ax.axhline(config["observation"]["threshold_factor"] * chance, color="tab:red",
                   linestyle="--", label="2 x chance")
        ax.set_xscale("log")
        ax.set_xticks(config["observation"]["k"], [str(k) for k in config["observation"]["k"]])
        ax.minorticks_off()
        ax.set_title(condition)
        ax.set_ylim(0, 1)
    for ax in list(axes.flat)[len(conditions):]:
        ax.axis("off")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="center right", fontsize=8)
    fig.supxlabel("Consecutive outputs observed per decision (k)")
    fig.supylabel("Strongest-attacker identity BA")
    fig.suptitle("Experiment C: identity BA vs observation length (seed mean, 95% run-"
                 "bootstrap interval; max over attackers and scenarios)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- B artifacts

class BSource:
    """Experiment-B artifacts through its public API; tests substitute synthetic sources."""

    def oof_manifest(self):
        return json.loads(OOF_MANIFEST.read_text(encoding="utf-8"))

    def oof(self, condition, seed):
        from .exp_b import load_oof
        return load_oof(condition, seed)

    def frozen(self, condition, seed):
        from .exp_b import load_frozen
        return load_frozen(condition, seed)

    def outputs(self, condition, seed):
        from .exp_b import load_outputs
        return load_outputs(condition, seed)


def rejection_rates(meta, finite, gate, rule):
    """Per subject and session: trials, non-finite, finite trials rejected by the gate."""
    rows = []
    for (subject, session), frame in meta.reset_index(drop=True).groupby(
            ["subject", "session"], sort=True):
        idx = frame.index.to_numpy()
        rows.append(dict(subject=int(subject), session=session,
                         threshold=rule.thresholds.get(int(subject), np.nan),
                         trials=len(idx), nonfinite=int((~finite[idx]).sum()),
                         rejected_finite=int((finite[idx] & ~gate[idx]).sum()),
                         hold_rate=float((~gate[idx]).mean())))
    return pd.DataFrame(rows)


def _available(config, manifest):
    usable = [c for c in config["conditions"] if manifest["selected"].get(c) is not None]
    return usable, [c for c in config["conditions"] if c not in usable]


# ---------------------------------------------------------------- dev (T only)

def run_dev(signals: Trials, config=None, source=None, result_dir=RESULTS,
            artifact_dir=ARTIFACTS, data_config=None, raw_runs=None):
    """T-only: temperatures, per-subject rejection rule and a T replay smoke run on OOF logits."""
    require_training(signals)
    config = config or load_config("exp_c")
    data_config = data_config or load_config("data")
    source = source or BSource()
    result_dir, artifact_dir = Path(result_dir), Path(artifact_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    signals = _sorted(signals)
    raw_runs = raw_runs or load_raw_runs(signals.meta, data_config)
    estimator, sos = data_config["covariance_estimator"], bandpass_sos(data_config)
    covs = trial_covariances(signals, estimator)
    rule = make_gate(config["rejection"]).fit(covs)
    live = replay_session(signals, raw_runs, rule, sos, estimator)
    gate, filter_ms, inference_ms = live["gate"], live["filter_ms"], live["inference_ms"]
    stream = stream_check(live["window_error"], config["timing"]["stream_tolerance"])
    if not stream["stream_matches_cache"]:
        raise ValueError(f"Streamed T windows differ from the cached trials: {stream}")
    rejection = rejection_rates(signals.meta, np.isfinite(covs.X).all(axis=(1, 2)), gate, rule)
    manifest = source.oof_manifest()
    conditions, missing = _available(config, manifest)
    controller, taus = config["controller"], config["taus"]
    _, _, starts = episode_layout(signals.meta, controller["episode_length"])
    temperatures, temperature_rows, episodes, timing = {}, [], [], []
    for condition in conditions:
        temperatures[condition] = {}
        for seed in manifest["seeds"]:
            oof = source.oof(condition, seed)
            oof_trials = Trials(np.asarray(oof["logits"], dtype=float),
                                pd.DataFrame({c: oof[c] for c in META_COLUMNS}))
            temperature = fit_temperature(oof_trials, config["temperature"]["bounds"],
                                          config["temperature"]["tolerance"])
            temperatures[condition][str(seed)] = temperature
            temperature_rows.append(dict(
                condition=condition, seed=seed, temperature=temperature, oof_trials=len(oof_trials),
                nll_unscaled=_nll(oof_trials.X, oof_trials.y, 1.0),
                nll_scaled=_nll(oof_trials.X, oof_trials.y, temperature)))
            # Smoke-only rule: T trials without an OOF row (B-excluded) are held.
            rows = _align(oof, signals.meta, required=False)
            logits = np.full((len(signals), 2), np.nan)
            has = ~np.isnan(rows)
            logits[has] = oof_trials.X[rows[has].astype(int)]
            for tau in taus:
                commands, angles, blocked, control_ms = control_pass(
                    logits, temperature, tau, starts, controller, gate=gate)
                tags = dict(condition=condition, seed=seed, tau=tau, mode="command")
                episodes.append(score_episodes(signals.meta, commands, angles, blocked,
                                               ~gate | ~has, controller).assign(**tags))
                timing.append(pd.DataFrame(dict(
                    **tags, filter_ms=filter_ms, inference_ms=inference_ms, control_ms=control_ms,
                    total_ms=filter_ms + inference_ms + control_ms)))
    tags = ["condition", "seed", "tau", "mode"]
    episodes = pd.concat(episodes, ignore_index=True)
    metrics = subject_metrics(episodes, tags)
    timing = pd.concat(timing, ignore_index=True)
    pd.DataFrame(temperature_rows).to_csv(result_dir / "dev_temperatures.csv", index=False)
    pd.DataFrame(rejection).to_csv(result_dir / "dev_rejection.csv", index=False)
    episodes.to_csv(result_dir / "dev_replay_episodes.csv", index=False)
    metrics.to_csv(result_dir / "dev_replay_metrics.csv", index=False)
    summarize_control(metrics, ["condition", "tau", "mode"]).to_csv(
        result_dir / "dev_replay_summary.csv", index=False)
    summarize_timing(timing, ["condition", "tau"]).to_csv(result_dir / "dev_timing.csv",
                                                          index=False)
    calibration = dict(
        schema=1, config_hash=_hash(config), oof_manifest_hash=_hash(manifest),
        selected=manifest["selected"], seeds=manifest["seeds"], conditions=conditions,
        no_operating_point=missing, t_signals_hash=_signals_hash(signals),
        temperatures=temperatures, rejection=rule.to_dict())
    _json(artifact_dir / "dev" / "calibration.json", calibration)
    summary = dict(
        phase="dev", gate=config["rejection"]["gate"], sessions=["T"], conditions=conditions,
        no_operating_point=missing,
        seeds=manifest["seeds"], taus=taus, t_trials=len(signals),
        t_hold_rate_gate=float((~gate).mean()), **stream,
        wall_seconds=time.perf_counter() - start,
        notes=["Smoke replay uses B's T out-of-fold logits; trials without an OOF row "
               "(B-excluded) are held.",
               "Dev inference timing excludes the frozen model (final models do not exist yet).",
               TIMING_NOTE])
    _json(result_dir / "dev_summary.json", summary)
    return calibration


# ---------------------------------------------------------------- final (E replay)

def replay_condition(model, saved, e_signals, raw_runs, t_meta, t_gate, rule, temperature,
                     sos, config, estimator):
    """E replay of one frozen model in both transmission modes for every tau."""
    controller, transmission = config["controller"], config["transmission"]
    live = replay_session(e_signals, raw_runs, rule, sos, estimator, model)
    gate, rep, logits = live["gate"], live["representation"], live["logits"]
    filter_ms, inference_ms = live["filter_ms"], live["inference_ms"]
    e_meta = e_signals.meta
    e_rows, t_rows = _align(saved, e_meta).astype(int), _align(saved, t_meta).astype(int)
    finite = e_meta["finite"].to_numpy(bool)
    compare = finite & np.isfinite(logits).all(axis=1)
    replay_diff = float(np.max(np.abs(logits[compare] - saved["logits"][e_rows][compare]),
                               initial=0.0))
    t_rep = np.asarray(saved["representation"][t_rows], dtype=float)
    t_logits = np.asarray(saved["logits"][t_rows], dtype=float)
    head, head_t_error = identify_head(Trials(t_rep, t_meta), t_logits,
                                       transmission["head_tolerance"])
    ok = np.isfinite(rep).all(axis=1)
    head_e_error = float(np.max(np.abs(head(rep[ok]) - logits[ok]), initial=0.0))
    if head_e_error > transmission["head_tolerance"]:
        raise ValueError(f"Receiver head does not reproduce E logits ({head_e_error:.3g})")
    message_e = np.where(gate[:, None], rep, np.nan)
    message_t = np.where(t_gate[:, None], t_rep, np.nan)
    block, complete, starts = episode_layout(e_meta, controller["episode_length"])
    streams = {"feature": (message_t, message_e)}
    trials, episodes, checks = [], [], []
    keys = e_meta[KEY].reset_index(drop=True)
    for tau in config["taus"]:
        passes = {
            "command": control_pass(logits, temperature, tau, starts, controller, gate=gate),
            "feature": control_pass(message_e, temperature, tau, starts, controller, head=head)}
        for mode, (commands, angles, blocked, control_ms) in passes.items():
            episodes.append(score_episodes(e_meta, commands, angles, blocked, ~gate,
                                           controller).assign(tau=tau, mode=mode))
            trials.append(keys.assign(
                tau=tau, mode=mode, command=commands, angle=angles, blocked=blocked,
                gated=~gate, filter_ms=filter_ms, inference_ms=inference_ms,
                control_ms=control_ms, total_ms=filter_ms + inference_ms + control_ms))
        command, feature = passes["command"], passes["feature"]
        same = pd.DataFrame(dict(
            subject=e_meta["subject"].to_numpy(), run=e_meta["run"].to_numpy(), block=block,
            commands=command[0] == feature[0], angles=command[1] == feature[1])).groupby(
            ["subject", "run", "block"]).agg(commands=("commands", "all"),
                                             angles=("angles", "all"))
        checks.append(dict(
            tau=tau, trials=len(e_meta), command_mismatches=int((command[0] != feature[0]).sum()),
            blocks=len(same), identical_command_blocks=int(same.commands.sum()),
            identical_trajectories_given_identical_commands=int(
                (same.commands & same.angles).sum()),
            passed=bool((same.angles | ~same.commands).all())))
        streams[stream_name("command", tau)] = (decide(t_logits, temperature, tau, t_gate),
                                                command[0])
    consistency = dict(replay_vs_saved_max_abs_logit=replay_diff,
                       replay_compared_trials=int(compare.sum()),
                       replay_matches_saved=bool(replay_diff <= transmission["replay_tolerance"]),
                       head_t_max_abs_error=head_t_error, head_e_max_abs_error=head_e_error,
                       **stream_check(live["window_error"],
                                      config["timing"]["stream_tolerance"]),
                       e_gated_holds=int((~gate).sum()), e_trials=len(e_meta))
    return dict(trials=pd.concat(trials, ignore_index=True),
                episodes=pd.concat(episodes, ignore_index=True), checks=checks,
                consistency=consistency, streams=streams)


def run_final(signals: Trials, calibration, config=None, source=None, attackers=None,
              result_dir=RESULTS, artifact_dir=ARTIFACTS, data_config=None, raw_runs=None):
    """E replay, metrics, transmission check, observation-length attack, tables and figure."""
    config = config or load_config("exp_c")
    data_config = data_config or load_config("data")
    attackers = attackers or load_config("exp_b")["attackers"]
    source = source or BSource()
    if set(signals.session) != {"T", "E"}:
        raise ValueError("Final requires both T and E signal Trials")
    if calibration["config_hash"] != _hash(config):
        raise ValueError("exp_c config changed since dev; rerun dev before sealing")
    manifest = source.oof_manifest()
    if calibration["oof_manifest_hash"] != _hash(manifest):
        raise ValueError("B's OOF selection changed since dev; rerun dev before sealing")
    signals = _sorted(signals)
    t_signals = signals.subset(signals.session == "T")
    e_signals = signals.subset(signals.session == "E")
    if calibration["t_signals_hash"] != _signals_hash(t_signals):
        raise ValueError("T trials differ from development")
    result_dir, artifact_dir = Path(result_dir), Path(artifact_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "final").mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    torch.set_num_threads(config["torch_threads"])
    estimator, sos = data_config["covariance_estimator"], bandpass_sos(data_config)
    rule = gate_from_dict(calibration["rejection"])
    t_covs, e_covs = (trial_covariances(s, estimator) for s in (t_signals, e_signals))
    t_gate, e_gate = rule.accept(t_covs), rule.accept(e_covs)
    rejection = pd.concat([
        rejection_rates(s.meta, np.isfinite(c.X).all(axis=(1, 2)), g, rule)
        for s, c, g in ((t_signals, t_covs, t_gate), (e_signals, e_covs, e_gate))],
        ignore_index=True)
    raw_runs = raw_runs or load_raw_runs(e_signals.meta, data_config)
    streams_meta = {"feature": dict(mode="feature", tau=None)}
    streams_meta.update({stream_name("command", tau): dict(mode="command", tau=tau)
                         for tau in config["taus"]})
    trials, episodes, checks, consistency, jobs = [], [], [], [], []
    obs = config["observation"]
    for condition in calibration["conditions"]:
        for seed in calibration["seeds"]:
            tags = dict(condition=condition, seed=seed)
            replay = replay_condition(
                source.frozen(condition, seed), source.outputs(condition, seed), e_signals,
                raw_runs, t_signals.meta, t_gate, rule, calibration["temperatures"][condition][str(seed)],
                sos, config, estimator)
            trials.append(replay["trials"].assign(**tags))
            episodes.append(replay["episodes"].assign(**tags))
            checks.extend(dict(**tags, **row) for row in replay["checks"])
            consistency.append(dict(**tags, **replay["consistency"]))
            for stream, (stream_t, stream_e) in replay["streams"].items():
                for k in obs["k"]:
                    jobs.append(delayed(_attack_job)(
                        dict(**tags, stream=stream), streams_meta[stream]["mode"], k,
                        stream_t, stream_e, t_signals.meta, e_signals.meta, attackers, seed,
                        obs["train_stride"], obs["min_valid_outputs"]))
    replay_seconds = time.perf_counter() - start
    with parallel_config(backend="loky", inner_max_num_threads=1):
        attacked = Parallel(n_jobs=config["workers"])(jobs)
    predictions = pd.concat([p for p, _ in attacked if p is not None], ignore_index=True)
    directional = pd.DataFrame([row for _, rows in attacked for row in rows])
    not_evaluable = directional.loc[~directional.evaluable, [
        "condition", "seed", "stream", "k", "scenario", "direction", "attacker",
        "train_groups", "short_identities"]].rename(
        columns=dict(short_identities="identities_below_inner_folds(subject:train_groups)"))
    trials = pd.concat(trials, ignore_index=True)
    episodes = pd.concat(episodes, ignore_index=True)
    tags = ["condition", "seed", "tau", "mode"]
    metrics = subject_metrics(episodes, tags)
    checks, consistency = pd.DataFrame(checks), pd.DataFrame(consistency)
    scenario, seed_mean, strongest, threshold, exclusions = summarize_observation(
        directional, predictions, streams_meta, config)

    rejection.to_csv(result_dir / "rejection_rates.csv", index=False)
    not_evaluable.to_csv(result_dir / "observation_not_evaluable.csv", index=False)
    episodes.to_csv(result_dir / "control_episodes.csv", index=False)
    metrics.to_csv(result_dir / "control_metrics.csv", index=False)
    summarize_control(metrics, ["condition", "tau", "mode"]).to_csv(
        result_dir / "control_summary.csv", index=False)
    summarize_timing(trials, ["condition", "tau", "mode"]).to_csv(
        result_dir / "timing_summary.csv", index=False)
    summarize_timing(trials, ["condition", "seed", "tau", "mode"]).to_csv(
        result_dir / "timing_by_seed.csv", index=False)
    checks.to_csv(result_dir / "transmission_check.csv", index=False)
    consistency.to_csv(result_dir / "replay_consistency.csv", index=False)
    directional.to_csv(result_dir / "observation_directional.csv", index=False)
    scenario.to_csv(result_dir / "observation_identity.csv", index=False)
    seed_mean.to_csv(result_dir / "observation_seed_mean.csv", index=False)
    strongest.to_csv(result_dir / "observation_strongest.csv", index=False)
    threshold.to_csv(result_dir / "observation_threshold.csv", index=False)
    exclusions.to_csv(result_dir / "observation_feature_exclusions.csv", index=False)
    plot_observation(strongest, config, result_dir / "observation_length.png")
    trials.to_csv(artifact_dir / "final" / "replay_trials.csv.gz", index=False)
    predictions.to_csv(artifact_dir / "final" / "attack_predictions.csv.gz", index=False)
    summary = dict(
        phase="final", gate=config["rejection"]["gate"], conditions=calibration["conditions"],
        no_operating_point=calibration["no_operating_point"], seeds=calibration["seeds"],
        taus=config["taus"], k=obs["k"], e_trials=len(e_signals),
        t_hold_rate_gate=float((~t_gate).mean()), e_hold_rate_gate=float((~e_gate).mean()),
        not_evaluable_attack_cells=len(not_evaluable),
        transmission_check_passed=bool(checks.passed.all()),
        transmission_command_mismatches=int(checks.command_mismatches.sum()),
        replay_matches_saved=bool(consistency.replay_matches_saved.all()),
        replay_max_abs_logit_difference=float(consistency.replay_vs_saved_max_abs_logit.max()),
        stream_matches_cache=bool(consistency.stream_matches_cache.all()),
        stream_window_max_rel_error=float(consistency.stream_window_max_rel_error.max()),
        smallest_k=json.loads(threshold.to_json(orient="records")),
        replay_seconds=replay_seconds, wall_seconds=time.perf_counter() - start,
        notes=[TIMING_NOTE,
               "Strongest = max seed-mean identity BA over attackers and scenarios "
               "(main E halves both directions averaged; auxiliary T->E).",
               "Feature-mode groups whose k messages are all null (held by the rejection "
               "rule) are excluded; feature BA is conditional on the retained groups "
               "(observation_feature_exclusions.csv).",
               "Attack cells with an identity below inner_folds training groups are not "
               "evaluable (observation_not_evaluable.csv) and are excluded from the "
               "strongest-attack maximum.",
               "Recorded replay, not closed-loop control."])
    _json(result_dir / "summary.json", summary)
    _json(artifact_dir / "final" / "manifest.json", dict(
        schema=1, calibration_hash=_hash(calibration), config_hash=_hash(config),
        attackers_hash=_hash(attackers), oof_manifest_hash=_hash(manifest),
        wall_seconds=summary["wall_seconds"]))
    print(json.dumps(dict(transmission_check_passed=summary["transmission_check_passed"],
                          wall_seconds=summary["wall_seconds"])), flush=True)
    return summary


def variant(name, config=None):
    """(config, result_dir, artifact_dir): primary uses the config's gate; sealed_gate
    reruns the analysis with the sealed Riemannian-distance gate (post-seal deviation 7)."""
    config = config or load_config("exp_c")
    if name == "primary":
        return config, RESULTS, ARTIFACTS
    rejection = dict(config["rejection"], gate="riemann_distance")
    return dict(config, rejection=rejection), RESULTS / name, ARTIFACTS / name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["dev", "final"])
    parser.add_argument("--variant", choices=["primary", "sealed_gate"], default="primary")
    args = parser.parse_args(argv)
    data_config = load_config("data")
    config, result_dir, artifact_dir = variant(args.variant)
    if args.phase == "dev":
        # Never open session-E files or caches during development.
        run_dev(load_trials(dict(data_config, sessions=["T"])), config,
                result_dir=result_dir, artifact_dir=artifact_dir, data_config=data_config)
    else:
        calibration = json.loads((artifact_dir / "dev" / "calibration.json").read_text(
            encoding="utf-8"))
        run_final(load_trials(data_config), calibration, config, result_dir=result_dir,
                  artifact_dir=artifact_dir, data_config=data_config)


if __name__ == "__main__":
    main()
