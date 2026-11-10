"""Experiment A: decoders B1-B3 x calibration amount, per subject, hypothesis H1.

`python -m beat.exp_a` fits every decoder on every combination of 1, 2, 3 and 6 T runs of
each subject and scores it on all retained E trials of that subject. `--t-only` instead
scores each combination on the subject's held-out T runs (no 6-run level) and drops
session E before any fit or score; it is the only mode allowed before the E seal.
"""
from __future__ import annotations

import argparse
import json
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import ROOT, load_config
from .data import Trials, load_trials, quality_mask, require_training
from .features import CommonTangentSpace, covariances
from .splits import N_RUNS, calibration_sets
from .stats import sign_flip_pvalue

DECODERS = ("B1", "B2", "B3")


def _logistic(C: float, max_iter: int):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, l1_ratio=0.0, max_iter=max_iter))


def band_power(trials: Trials, cfg: dict, sfreq: float) -> np.ndarray:
    """Log mean Welch power per band and channel: (n, n_bands * channels), band-major."""
    nperseg = int(round(cfg["welch_segment_s"] * sfreq))
    f, psd = welch(
        trials.X.astype(np.float64), fs=sfreq, window=cfg["welch_window"], nperseg=nperseg,
        noverlap=int(round(nperseg * cfg["welch_overlap"])), detrend=cfg["welch_detrend"], axis=-1,
    )
    return np.concatenate([np.log(psd[..., (f >= lo) & (f < hi)].mean(axis=-1)) for lo, hi in cfg["bands_hz"]], axis=1)


class BandPowerDecoder:
    """B1: Welch band power (8-13, 13-30 Hz per channel) + standardization + L2 logistic regression."""

    def __init__(self, cfg: dict, sfreq: float, max_iter: int):
        self.cfg, self.sfreq, self.max_iter = cfg, sfreq, max_iter

    def fit(self, train: Trials):
        require_training(train)
        self.clf_ = _logistic(self.cfg["logistic_C"], self.max_iter).fit(band_power(train, self.cfg, self.sfreq), train.y)
        return self

    def predict(self, trials: Trials) -> np.ndarray:
        return self.clf_.predict(band_power(trials, self.cfg, self.sfreq))


class CSPDecoder:
    """B2: CSP (filters from both ends of the generalized eigenvalue spectrum), log-variance,
    LDA with automatic (Ledoit-Wolf) shrinkage."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _log_variance(self, trials: Trials) -> np.ndarray:
        C = covariances(trials, self.cfg["covariance_estimator"]).X
        return np.log(np.einsum("fc,ncd,fd->nf", self.filters_, C, self.filters_))

    def fit(self, train: Trials):
        require_training(train)
        C = covariances(train, self.cfg["covariance_estimator"]).X
        c0, c1 = (C[train.y == k].mean(axis=0) for k in (0, 1))
        _, vecs = eigh(c1, c0 + c1)  # ascending: first filters favour class 0, last favour class 1
        half = self.cfg["n_filters"] // 2
        self.filters_ = np.concatenate([vecs[:, :half], vecs[:, -half:]], axis=1).T
        self.lda_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(self._log_variance(train), train.y)
        return self

    def predict(self, trials: Trials) -> np.ndarray:
        return self.lda_.predict(self._log_variance(trials))


class TangentSpaceDecoder:
    """B3: OAS covariance, tangent space at the training Riemannian mean, standardization + L2 LR."""

    def __init__(self, cfg: dict, max_iter: int):
        self.cfg, self.max_iter = cfg, max_iter

    def fit(self, train: Trials):
        require_training(train)
        cov = covariances(train, self.cfg["covariance_estimator"])
        self.ts_ = CommonTangentSpace().fit(cov)
        self.clf_ = _logistic(self.cfg["logistic_C"], self.max_iter).fit(self.ts_.transform(cov).X, train.y)
        return self

    def predict(self, trials: Trials) -> np.ndarray:
        cov = covariances(trials, self.cfg["covariance_estimator"])
        return self.clf_.predict(self.ts_.transform(cov).X)


def make_decoder(name: str, cfg: dict, sfreq: float):
    max_iter = cfg["logistic_max_iter"]
    if name == "B1":
        return BandPowerDecoder(cfg["b1_band_power"], sfreq, max_iter)
    if name == "B2":
        return CSPDecoder(cfg["b2_csp"])
    if name == "B3":
        return TangentSpaceDecoder(cfg["b3_tangent_space"], max_iter)
    raise ValueError(f"unknown decoder {name}")


def run_experiment(trials: Trials, cfg: dict, sfreq: float, t_only: bool = False) -> pd.DataFrame:
    """One row per subject x decoder x calibration combination with BA and Cohen's kappa.

    Excluded trials (artifact flag or non-finite) are dropped before anything else. Each
    combination's training trials are identical for B1-B3. Test set: all retained E trials
    of the subject, or with t_only the subject's T runs outside the combination.
    """
    kept = trials.subset(quality_mask(trials))
    rows = []
    for subject in np.unique(kept.subject):
        pool = kept.subset((kept.subject == subject) & (kept.session == "T"))
        e_test = None if t_only else kept.subset((kept.subject == subject) & (kept.session == "E"))
        for n_runs in cfg["calibration_runs"]:
            if t_only and n_runs == N_RUNS:
                continue  # no held-out T run remains
            for runs in calibration_sets(n_runs):
                in_cal = np.isin(pool.run, runs)
                train = pool.subset(in_cal)
                test = pool.subset(~in_cal) if t_only else e_test
                for name in DECODERS:
                    pred = make_decoder(name, cfg, sfreq).fit(train).predict(test)
                    rows.append(dict(
                        subject=int(subject), decoder=name, n_runs=n_runs, runs="-".join(map(str, runs)),
                        n_train=len(train), test_session="T" if t_only else "E", n_test=len(test),
                        ba=balanced_accuracy_score(test.y, pred), kappa=cohen_kappa_score(test.y, pred),
                    ))
    return pd.DataFrame(rows)


def subject_values(scores: pd.DataFrame) -> pd.DataFrame:
    """A subject's value at a calibration level: mean over its combinations."""
    return (scores.groupby(["subject", "decoder", "n_runs"], as_index=False)
            .agg(n_combinations=("ba", "size"), ba=("ba", "mean"), kappa=("kappa", "mean")))


def summarize(subject_scores: pd.DataFrame) -> pd.DataFrame:
    """Per decoder x calibration level: mean, median and inter-subject spread of BA and kappa."""
    stats = ["mean", "median", lambda v: v.std(ddof=1), "min", "max"]
    out = subject_scores.groupby(["decoder", "n_runs"]).agg(n_subjects=("subject", "size"),
                                                            **{f"{m}_{s}": (m, f) for m in ("ba", "kappa")
                                                               for s, f in zip(("mean", "median", "sd", "min", "max"), stats)})
    return out.reset_index()


def h1_test(subject_scores: pd.DataFrame, cfg: dict) -> dict:
    """H1: paired 1-run B3 - B2 BA over subjects, exact one-sided sign-flip permutation test."""
    h = cfg["h1"]
    a, b = h["difference"]
    level = subject_scores[subject_scores.n_runs == h["calibration_runs"]].pivot(index="subject", columns="decoder", values="ba")
    diffs = level[a] - level[b]
    return dict(
        hypothesis=f"{h['calibration_runs']}-run {a} - {b} balanced accuracy > 0",
        alternative=h["alternative"],
        n_subjects=int(diffs.size),
        n_sign_patterns=2 ** int(diffs.size),
        mean_difference=float(diffs.mean()),
        median_difference=float(diffs.median()),
        n_positive=int((diffs > 0).sum()),
        p_value=sign_flip_pvalue(diffs.to_numpy(), alternative=h["alternative"]),
        per_subject={int(s): dict(**{k: float(level.loc[s, k]) for k in (a, b)}, difference=float(diffs[s]))
                     for s in diffs.index},
    )


def plot_calibration(subject_scores: pd.DataFrame, path, test_session: str) -> None:
    """BA vs number of calibration runs: subject mean with +/- SD band and every subject's value."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    levels = sorted(subject_scores.n_runs.unique())
    for i, (name, color) in enumerate(zip(DECODERS, ("tab:blue", "tab:orange", "tab:green"))):
        d = subject_scores[subject_scores.decoder == name]
        g = d.groupby("n_runs")["ba"]
        mean, sd = g.mean().reindex(levels), g.std(ddof=1).reindex(levels)
        ax.plot(levels, mean, "o-", color=color, lw=2, label=f"{name} (subject mean ± SD)")
        ax.fill_between(levels, mean - sd, mean + sd, color=color, alpha=0.15)
        ax.scatter(d.n_runs + (i - 1) * 0.08, d.ba, color=color, s=10, alpha=0.5)
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set(xticks=levels, xlabel="number of calibration T runs", ylabel=f"balanced accuracy (session {test_session})",
           title="Experiment A: calibration curve (dots = subjects)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _provenance() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return dict(git_commit=git("rev-parse", "HEAD"),
                sealed_paths_clean=git("status", "--porcelain", "--", "beat", "configs", "docs") == "")


def write_outputs(scores: pd.DataFrame, cfg: dict, out_dir) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    test_session = scores.test_session.iloc[0]
    per_subject = subject_values(scores)
    scores.to_csv(out_dir / "scores.csv", index=False)
    per_subject.to_csv(out_dir / "subject_scores.csv", index=False)
    summarize(per_subject).to_csv(out_dir / "summary.csv", index=False)
    h1 = dict(test_session=test_session, **h1_test(per_subject, cfg), **_provenance(), config=cfg)
    (out_dir / "h1.json").write_text(json.dumps(h1, indent=2), encoding="utf-8")
    plot_calibration(per_subject, out_dir / "calibration_curve.png", test_session)
    return h1


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--t-only", action="store_true",
                        help="score held-out T runs only; never fits or scores session E (pre-seal smoke check)")
    parser.add_argument("--out", help="output directory (default: output_dir, or output_dir/t_only)")
    args = parser.parse_args(argv)
    cfg = load_config("exp_a")
    trials = load_trials()
    if args.t_only:
        trials = trials.subset(trials.session == "T")
    out_dir = ROOT / (args.out or (cfg["output_dir"] + ("/t_only" if args.t_only else "")))
    scores = run_experiment(trials, cfg, load_config("data")["sfreq"], t_only=args.t_only)
    h1 = write_outputs(scores, cfg, out_dir)
    print(summarize(subject_values(scores)).round(3).to_string(index=False))
    print(f"H1 ({h1['test_session']}): mean diff {h1['mean_difference']:+.4f}, p = {h1['p_value']:.4f}; outputs in {out_dir}")


if __name__ == "__main__":
    main()
