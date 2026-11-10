"""Experiment G: fixed reference-session defense versus post-deployment collection.

Source: https://arxiv.org/html/2610.15626v1, equations 1--8, sections 4/5/7,
Table 1, appendices C/H/L/N. This is a specification-based reproduction, not
the authors' unreleased code. AMBIGUITIES below are part of the result JSON.
No utility reproduction or universal anonymization claim is made.
"""
from __future__ import annotations

import os

# Set before numerical imports, including in spawned joblib workers.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed, parallel_config
from pyriemann.estimation import XdawnCovariances
from pyriemann.geometry.base import invsqrtm, sqrtm
from pyriemann.geometry.mean import mean_riemann
from pyriemann.geometry.tangentspace import tangent_space
from pyriemann.tangentspace import TangentSpace
from scipy.linalg import eigh, solve_triangular
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.distance import cdist, pdist
from scipy.stats import permutation_test
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .attacks import fit_attackers
from .config import ROOT
from .data import LeakageError, Trials, _mi_runs, require_training
from .features import covariances
from .stats import sign_flip_pvalue, unit_bootstrap_ba

AMBIGUITIES = [
    "Population Riemannian means use all fitting trials (trial weighting), not a mean of subject means.",
    "Each stage consumes the preceding released signals. SCW's fixed tangent reference is "
    "the pooled Riemannian mean of Stage-1 fitting covariances; the text also permits "
    "interpreting Cbar as the original Stage-1 target. No choice is reselected on S2.",
    "Defense covariance follows the explicit uncentered XX.T/t + 1e-6 I formula, in "
    "the BNCI MAT microvolt units. No demeaning, OAS or trace normalization in the defense.",
    "Recolouring uses regularized source and target Cholesky factors. Thus empirical "
    "output covariance differs from the target by the transported 1e-6 ridge.",
    "SCW scatter is centered sample covariance (ddof=1) + 1e-4 I; shrinkage 0.1 "
    "is applied AFTER this ridge iff subject n<=d; eigenfloor 1e-8 times maximum. "
    "The map acts on z, NOT z minus a subject mean, as equation 5 states.",
    "BCI window chosen as cue+[0,4) seconds (1000 samples), consistent with Appendix C "
    "and Appendix F's imagery window; main-protocol onset is not explicitly specified.",
    "Fifth-order 8-30 Hz Butterworth SOS zero-phase filtering is per extracted epoch "
    "with scipy default odd padding; fifth order is specified in Section 4, but padding "
    "and application boundaries are unspecified. BCA repeats "
    "the same filter on Stage-2 output and adds its untouched residual.",
    "All four MI classes and official artifact-flagged trials are retained to match "
    "288 trials/session. Nonfinite or incomplete epochs are rejected and counted.",
    "CSP details incomplete: one-versus-rest identities, uncentered trace-normalized "
    "trial covariance, arithmetic class means, four smallest and four largest generalized "
    "eigenvectors per identity, log normalized projected variance; no additional subbands "
    "or feature scaling. RBF SVC and SVD LDA use sklearn defaults.",
    "XDAWN uses one fixed filter per identity, OAS for both evoked filters and augmented "
    "covariance, applyfilters=True, fixed Riemannian tangent space and default RBF SVC. "
    "Unlike the paper's oracle maximum over 1/2/4/8, no evaluation-session selection is "
    "performed; this resource-limited result is NOT its oracle reproduction.",
    "Reverse S2->S1 swaps only attacker train/test; the defense remains fitted on S1.",
    "Aggregation uses nonoverlapping consecutive groups within subject/session/run; "
    "remainders are dropped, never crosses halves. Single-trial-trained classifiers "
    "receive mean feature vectors; nearest-template mean tangent is also reported. "
    "This differs from Appendix H random subsets and does not retrain classifiers for k.",
    "Mechanism uses OAS covariances and a separate S1-fitted tangent reference per condition: "
    "mean pairwise Euclidean subject-centroid distance and mean within-subject Euclidean "
    "distance, equally weighted subjects. The paper does not fully define these summaries.",
    "BEAT logistic/RBF-SVM/MLP reuse configs/exp_b.yaml unchanged, on attacker-training-fitted "
    "OAS tangent features; one deterministic seed 42, not Experiment B's five-seed ensemble.",
    "Enrolment curves use the first n retained trials per participant in recording order "
    "within the attacker-training half, including when halves are reversed. The defense "
    "stays fitted on all S1; the attacker's tangent reference/scaling/classifiers are refitted "
    "using only these n labelled trials. Test groups are unchanged across n.",
    "For n<inner_folds, BEAT RBF-SVM and MLP are not evaluable under the requested shared "
    "fit_attackers eligibility rule (MLP itself has no inner search). Fixed-C logistic "
    "regression is fitted separately with identical BEAT pipeline parameters at tiny n. "
    "No CV reduction, oversampling or hyperparameter fallback. All-enrolment cells reuse "
    "the identical original post-deployment fits and predictions.",
    "Run-bootstrap intervals are conditional on fitted attackers, 500 resamples, "
    "and are descriptive rather than confirmatory; intervals are unavailable if any "
    "subject has fewer than two test recording units. Direction averages are unweighted. "
    "Paired signs are exhaustive for nine BCI identities and Monte Carlo (10000 draws, "
    "plus-one correction) for Lee2019 to avoid enumerating 2^54 signs.",
    "Lee2019 uses pre-existing causal 8-30 Hz, cue+[1,4) reduced epochs, unlike the "
    "paper's zero-phase four-second preprocessing; train/test phases become two recording "
    "units. It cannot reconstruct missing samples or undo filtering. Session 2 requires "
    "a separately sealed plan; no Lee2019 data are needed for synthetic tests.",
    "SPDNet-ID and MI utility decoders are not run. The strongest implemented attacker "
    "is not the paper's five-attacker worst, and identity failure alone is not anonymity.",
]

THREAT_MODEL_INTERPRETATION = (
    "Results are threat-model dependent. The paper's transform targets cross-session "
    "linkage when the attacker has reference-session labels only. The post-deployment "
    "audit gives the evaluator additional labelled deployment-session data; its enrolment "
    "curve measures how much of that additional access is needed. Strong identification "
    "in this scenario does not show that the paper's reference-session results are wrong "
    "or contradict its stated threat model. The 2a results are exploratory and concern "
    "a 9-person closed set, not a universal anonymity or utility assessment."
)

ENROLMENT_ATTACKERS = ("paper_ts_svm", "beat_logistic", "beat_rbf_svm", "beat_mlp")

PAPER_BNCI = {
    "raw_S1_to_S2": {"paper_ts_svm": 75.9, "paper_xdawn_svm": 85.1,
                     "paper_csp_svm": 72.2, "paper_csp_lda": 77.9, "SPDNet_ID": 67.4},
    "undefended_S1_to_S2": {"paper_ts_svm": 90.8, "paper_xdawn_svm": 87.6,
                            "paper_csp_svm": 73.2, "paper_csp_lda": 77.3, "SPDNet_ID": 89.9},
    "defended_S1_to_S2": {"paper_ts_svm": 11.1, "paper_xdawn_svm": 15.5,
                          "paper_csp_svm": 15.6, "paper_csp_lda": 3.5, "SPDNet_ID": 15.1},
    "defended_S2_to_S1": {"paper_ts_svm": 11.1, "paper_xdawn_svm": 12.9,
                          "paper_csp_svm": 5.8, "paper_csp_lda": 0.9, "SPDNet_ID": 12.1},
}


def spatial_cov(X):
    X = np.asarray(X, dtype=np.float64)
    return X @ X.swapaxes(-1, -2) / X.shape[-1] + 1e-6 * np.eye(X.shape[-2])


def recolour(X, target):
    source = spatial_cov(X)
    out = np.empty_like(X, dtype=np.float64)
    for i in range(len(X)):
        out[i] = np.linalg.cholesky(target[i]) @ solve_triangular(
            np.linalg.cholesky(source[i]), X[i], lower=True)
    return out


def rma_maps(C, subjects, strength):
    target = mean_riemann(C)
    L = np.linalg.cholesky(target)
    maps = {}
    for s in np.unique(subjects):
        source = np.linalg.cholesky(mean_riemann(C[subjects == s]))
        W = solve_triangular(source.T, L.T, lower=False).T
        maps[int(s)] = strength * W + (1 - strength) * np.eye(C.shape[-1])
    return maps


def apply_rma(X, subjects, maps):
    C = spatial_cov(X)
    target = np.empty_like(C)
    for s in np.unique(subjects):
        ix = subjects == s
        W = maps[int(s)]
        target[ix] = W @ C[ix] @ W.T
    return recolour(X, target)


def scatter(Z):
    return np.cov(Z, rowvar=False, ddof=1) + 1e-4 * np.eye(Z.shape[1])


def subject_scatter(Z):
    S = scatter(Z)
    d = Z.shape[1]
    if len(Z) <= d:
        S = .9 * S + .1 * np.trace(S) / d * np.eye(d)
    vals, vecs = np.linalg.eigh(S)
    return (vecs * np.maximum(vals, 1e-8 * vals.max())) @ vecs.T


def clipped_inverse_tangent(Z, reference):
    c = reference.shape[0]
    i, j = np.triu_indices(c)
    M = np.zeros((len(Z), c, c))
    M[:, i, j] = Z / np.where(i == j, 1., np.sqrt(2.))
    M[:, j, i] = M[:, i, j]
    vals, vecs = np.linalg.eigh(M)
    E = (vecs * np.exp(np.clip(vals, -40., 40.))[:, None, :]) @ vecs.swapaxes(-1, -2)
    root = sqrtm(reference)
    return root @ E @ root


class FourStageDefense:
    """All fitted quantities are S1-only; transform never updates statistics."""

    def __init__(self, strengths=(.9, .35, .6, 0.), sfreq=250):
        if len(strengths) != 4 or not all(0 <= v <= 1 for v in strengths):
            raise ValueError("Four strengths must lie in [0,1]")
        self.strengths = tuple(strengths)
        self.sfreq = sfreq
        self.sos = butter(5, [8, 30], fs=sfreq, btype="bandpass", output="sos")

    def fit(self, train: Trials):
        require_training(train)
        if not np.isfinite(train.X).all():
            raise ValueError("Nonfinite defense input")
        self.subjects_ = np.unique(train.subject)
        self.fit_counts_ = train.meta.groupby("subject").size().to_dict()
        X = np.asarray(train.X, dtype=np.float64)
        g1, g2, alpha, beta = self.strengths
        if g1:
            self.rma_ = rma_maps(spatial_cov(X), train.subject, g1)
            X = apply_rma(X, train.subject, self.rma_)
        if g2:
            self.reference_ = mean_riemann(spatial_cov(X))
            Z = tangent_space(spatial_cov(X), self.reference_, metric="riemann")
            root = sqrtm(scatter(Z))
            self.scw_ = {
                int(s): g2 * (root @ invsqrtm(subject_scatter(Z[train.subject == s])))
                + (1 - g2) * np.eye(Z.shape[1]) for s in self.subjects_
            }
            X = self._scw(X, train.subject)
        if alpha:
            band = sosfiltfilt(self.sos, X, axis=-1)
            self.bca_ = rma_maps(spatial_cov(band), train.subject, alpha)
            X = X - band + apply_rma(band, train.subject, self.bca_)
        if beta:
            self.cleanup_ = rma_maps(spatial_cov(X), train.subject, beta)
        return self

    def _scw(self, X, subjects):
        Z = tangent_space(spatial_cov(X), self.reference_, metric="riemann")
        for s in np.unique(subjects):
            ix = subjects == s
            Z[ix] = Z[ix] @ self.scw_[int(s)].T
        return recolour(X, clipped_inverse_tangent(Z, self.reference_))

    def transform(self, trials: Trials):
        if not np.isin(trials.subject, self.subjects_).all():
            raise ValueError("Subject absent from fitting session")
        X = np.asarray(trials.X, dtype=np.float64).copy()
        if not np.isfinite(X).all():
            raise ValueError("Nonfinite defense input")
        g1, g2, alpha, beta = self.strengths
        if g1:
            X = apply_rma(X, trials.subject, self.rma_)
        if g2:
            X = self._scw(X, trials.subject)
        if alpha:
            band = sosfiltfilt(self.sos, X, axis=-1)
            X = X - band + apply_rma(band, trials.subject, self.bca_)
        if beta:
            X = apply_rma(X, trials.subject, self.cleanup_)
        if not np.isfinite(X).all():
            raise FloatingPointError("Nonfinite released signals")
        return trials.with_X(X)


def load_bnci(cfg, session):
    """Raw helper reused; no Experiment-B two-class or artifact exclusion."""
    signals, rows = [], []
    rejected = 0
    sos = butter(cfg["filter_order"], cfg["bandpass"], fs=cfg["sfreq"],
                 btype="bandpass", output="sos")
    start, stop = (int((2 + v) * cfg["sfreq"]) for v in cfg["bnci"]["window_after_cue"])
    for subject in cfg["bnci"]["subjects"]:
        path = ROOT / cfg["bnci"]["raw_dir"] / f"A{subject:02d}{session}.mat"
        for r, run in enumerate(_mi_runs(path), 1):
            if int(run.fs) != cfg["sfreq"]:
                raise ValueError(f"Unexpected sample rate: {path}")
            raw = np.asarray(run.X, dtype=float)[:, :22]
            for t, (onset, label, artifact) in enumerate(zip(
                    np.atleast_1d(run.trial).astype(int) - 1,
                    np.atleast_1d(run.y).astype(int), np.atleast_1d(run.artifacts)), 1):
                X = raw[onset + start:onset + stop].T
                if X.shape != (22, stop - start) or not np.isfinite(X).all():
                    rejected += 1
                    continue
                signals.append(sosfiltfilt(sos, X, axis=-1))
                rows.append(dict(subject=subject, session=session, run=r, trial=t,
                                 label=label - 1, artifact=bool(artifact), finite=True))
    return Trials(np.stack(signals), pd.DataFrame(rows)), rejected


def load_lee2019(directory, subjects, session=1, *, allow_session2=False):
    """Check authorization BEFORE opening, globbing, or inspecting session-2 paths."""
    if session not in (1, 2):
        raise ValueError("Session must be 1 or 2")
    if session == 2 and not allow_session2:
        raise LeakageError("Lee2019 session 2 is sealed; a new sealed plan is required")
    signals, frames = [], []
    rejected = 0
    for subject in subjects:
        with np.load(Path(directory) / f"sess{session}_subj{subject:02d}.npz",
                     allow_pickle=False) as z:
            X = z["X"]
            phases = z["phase"].astype(str)
            if not np.isin(phases, ["train", "test"]).all():
                raise ValueError("Unknown Lee2019 phase")
            if X.ndim != 3 or X.shape[1:] != (20, 750):
                raise ValueError("Expected reduced 20-channel, 750-sample epochs")
            meta = pd.DataFrame(dict(subject=subject, session="T" if session == 1 else "E",
                                     run=np.where(phases == "train", 1, 2), trial=z["trial"],
                                     label=z["label"], artifact=False, finite=z["finite"]))
            keep = z["finite"] & np.isfinite(X).all(axis=(1, 2))
            order = meta.loc[keep].sort_values(["run", "trial"], kind="stable").index
            signals.append(X[order].astype(float))
            frames.append(meta.loc[order])
            rejected += int((~keep).sum())
    return Trials(np.concatenate(signals), pd.concat(frames, ignore_index=True)), rejected


def chronological_halves(trials):
    first = np.zeros(len(trials), dtype=bool)
    for _, frame in trials.meta.groupby("subject", sort=True):
        ordered = frame.sort_values(["run", "trial"], kind="stable").index.to_numpy()
        # Use whole recording units when available (2a: runs 1--3 / 4--6).
        runs = np.sort(frame.run.unique())
        if len(runs) >= 2:
            first[frame.index] = frame.run.isin(runs[:len(runs) // 2])
        else:
            first[ordered[:len(ordered) // 2]] = True
    return trials.subset(first), trials.subset(~first)


def aggregate_features(X, meta, k):
    features, records = [], []
    for _, frame in meta.groupby(["subject", "session", "run"], sort=True):
        ix = frame.sort_values("trial", kind="stable").index.to_numpy()
        # A removed nonfinite trial must not turn a gap into consecutive trials.
        positions = meta.loc[ix, "trial"].to_numpy()
        segments = np.split(ix, np.flatnonzero(np.diff(positions) != 1) + 1)
        for segment in segments:
            for start in range(0, len(segment) - k + 1, k):
                group = segment[start:start + k]
                features.append(X[group].mean(axis=0))
                record = meta.loc[group[0]].to_dict()
                record["last_trial"] = int(meta.loc[group[-1], "trial"])
                records.append(record)
    if not features:
        raise ValueError(f"No consecutive groups for k={k}")
    return np.stack(features), pd.DataFrame(records)


def first_labelled_trials(train, n):
    """Only the supplied attack-training half is eligible for enrolment."""
    indices = []
    for _, frame in train.meta.groupby("subject", sort=True):
        ordered = frame.sort_values(["run", "trial"], kind="stable").index.to_numpy()
        if n != "all" and len(ordered) < n:
            raise ValueError(f"Insufficient enrolment trials: requested {n}, have {len(ordered)}")
        indices.extend(ordered if n == "all" else ordered[:n])
    return train.subset(indices)


def evaluate_enrolment(train, test, cfg, attack_config, condition, direction,
                       full_rows, full_predictions, full_convergence):
    """Same single-trial fits, mean-feature test aggregation, and fixed S1 defense."""
    rows, predictions, convergence, memberships = [], [], [], []
    for n in cfg["enrolment_n"]:
        enrolled = first_labelled_trials(train, n)
        member = enrolled.meta[["subject", "session", "run", "trial"]].copy()
        member["condition"], member["direction"], member["enrolment_n"] = condition, direction, n
        memberships.append(member)
        common = dict(condition=condition, scenario="post_deployment", direction=direction,
                      enrolment_n=str(n), n_train=len(enrolled),
                      min_train_per_subject=int(enrolled.meta.groupby("subject").size().min()))
        if n == "all":
            for original in full_rows:
                if original["attacker"] in ENROLMENT_ATTACKERS and original["k"] in cfg["enrolment_k"]:
                    rows.append(dict(original, **{"enrolment_n": "all",
                                     "min_train_per_subject": common["min_train_per_subject"],
                                     "status": "evaluable", "reason": ""}))
            selected = full_predictions[
                full_predictions.attacker.isin(ENROLMENT_ATTACKERS)
                & full_predictions.k.isin(cfg["enrolment_k"])].copy()
            selected["enrolment_n"] = "all"
            predictions.append(selected)
            convergence.append(dict(full_convergence, enrolment_n="all", reused=True))
            continue
        ts = TangentSpace(metric="riemann", tsupdate=False)
        F = ts.fit_transform(covariances(enrolled).X)
        Q = ts.transform(covariances(test).X)
        models = {"paper_ts_svm": SVC().fit(F, enrolled.subject)}
        eligible = common["min_train_per_subject"] >= attack_config["inner_folds"]
        if eligible:
            models.update({f"beat_{name}": model for name, model in
                           fit_attackers(F, enrolled.subject, attack_config, cfg["seed"]).items()})
            convergence.append(dict(condition=condition, direction=direction, enrolment_n=str(n),
                                    reused=False, **models["beat_mlp"].convergence_))
        else:
            # The shared helper rejects the whole bundle; logistic needs no CV.
            models["beat_logistic"] = Pipeline([
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(
                    C=attack_config["logistic_C"], max_iter=attack_config["logistic_max_iter"],
                    solver="lbfgs", random_state=cfg["seed"])),
            ]).fit(F, enrolled.subject)
        for k in cfg["enrolment_k"]:
            grouped, meta = aggregate_features(Q, test.meta, k)
            truth = meta.subject.to_numpy()
            for name in ENROLMENT_ATTACKERS:
                row = dict(common, attacker=name, k=k, n_test=len(meta))
                if name not in models:
                    rows.append(dict(row, accuracy=None, balanced_accuracy=None,
                                     status="not_evaluable",
                                     reason=f"n={n} < inner_folds={attack_config['inner_folds']}; "
                                            "shared BEAT fit eligibility unchanged"))
                    continue
                pred = models[name].predict(grouped)
                rows.append(dict(row, accuracy=float(accuracy_score(truth, pred)),
                                 balanced_accuracy=float(balanced_accuracy_score(truth, pred)),
                                 status="evaluable", reason=""))
                records = meta[["subject", "session", "run", "trial", "last_trial"]].copy()
                for key in ("condition", "scenario", "direction", "enrolment_n", "attacker", "k"):
                    records[key] = row[key]
                records["prediction"] = pred
                predictions.append(records)
    return (rows, pd.concat(predictions, ignore_index=True), convergence,
            pd.concat(memberships, ignore_index=True))


class OVRCSP:
    def __init__(self, pairs=4):
        self.pairs = pairs

    def fit(self, X, y):
        C = spatial_cov(X)
        C /= np.trace(C, axis1=1, axis2=2)[:, None, None]
        filters = []
        if 2 * self.pairs > X.shape[1]:
            raise ValueError("Too many CSP components")
        for s in np.unique(y):
            a, b = C[y == s].mean(axis=0), C[y != s].mean(axis=0)
            _, vectors = eigh(a, a + b)
            filters.append(np.concatenate([vectors[:, :self.pairs],
                                           vectors[:, -self.pairs:]], axis=1).T)
        self.filters_ = np.stack(filters)
        return self

    def transform(self, X):
        C = spatial_cov(X)
        C /= np.trace(C, axis1=1, axis2=2)[:, None, None]
        features = []
        for W in self.filters_:
            variance = np.einsum("ic,ncd,id->ni", W, C, W)
            features.append(np.log(variance / variance.sum(axis=1, keepdims=True)))
        return np.concatenate(features, axis=1)


def evaluate_split(train, test, cfg, attack_config, condition, scenario, direction):
    """Attacker preprocessing is fitted only on its own labelled collection."""
    with threadpool_limits(limits=1):
        ts = TangentSpace(metric="riemann", tsupdate=False)
        F = ts.fit_transform(covariances(train).X)
        Q = ts.transform(covariances(test).X)
        models = {"paper_ts_svm": SVC().fit(F, train.subject)}
        models.update({f"beat_{name}": model for name, model in
                       fit_attackers(F, train.subject, attack_config, cfg["seed"]).items()})
        feature_pairs = {name: (F, Q) for name in models}
        csp = OVRCSP(cfg["csp_pairs"]).fit(train.X, train.subject)
        CF, CQ = csp.transform(train.X), csp.transform(test.X)
        models["paper_csp_svm"] = SVC().fit(CF, train.subject)
        models["paper_csp_lda"] = LinearDiscriminantAnalysis().fit(CF, train.subject)
        feature_pairs.update({name: (CF, CQ) for name in ("paper_csp_svm", "paper_csp_lda")})
        xd = XdawnCovariances(nfilter=cfg["xdawn_filters"], applyfilters=True,
                              estimator="oas", xdawn_estimator="oas")
        xt = TangentSpace(metric="riemann", tsupdate=False)
        XF = xt.fit_transform(xd.fit_transform(train.X, train.subject))
        XQ = xt.transform(xd.transform(test.X))
        models["paper_xdawn_svm"] = SVC().fit(XF, train.subject)
        feature_pairs["paper_xdawn_svm"] = (XF, XQ)
        identities = np.unique(train.subject)
        templates = np.stack([F[train.subject == s].mean(axis=0) for s in identities])
        models["mean_tangent_template"] = None
        feature_pairs["mean_tangent_template"] = (F, Q)
        rows, predictions = [], []
        convergence = models["beat_mlp"].convergence_
        for name, model in models.items():
            for k in cfg["k"]:
                grouped, meta = aggregate_features(feature_pairs[name][1], test.meta, k)
                pred = (identities[cdist(grouped, templates).argmin(axis=1)] if model is None
                        else model.predict(grouped))
                truth = meta.subject.to_numpy()
                row = dict(condition=condition, scenario=scenario, direction=direction,
                           attacker=name, k=k, n_train=len(train), n_test=len(truth),
                           accuracy=float(accuracy_score(truth, pred)),
                           balanced_accuracy=float(balanced_accuracy_score(truth, pred)))
                if k == 1:
                    if meta.groupby("subject").run.nunique().min() < 2:
                        row.update(ba_ci_low=None, ba_ci_high=None,
                                   ba_ci_status="unavailable: fewer than two runs per subject")
                    else:
                        row["ba_ci_low"], row["ba_ci_high"] = unit_bootstrap_ba(
                            truth, pred, truth, meta.run, n_boot=cfg["bootstrap_samples"],
                            seed=cfg["seed"])
                        row["ba_ci_status"] = "conditional run bootstrap"
                rows.append(row)
                records = meta[["subject", "session", "run", "trial", "last_trial"]].copy()
                for key in ("condition", "scenario", "direction", "attacker", "k"):
                    records[key] = row[key]
                records["prediction"] = pred
                predictions.append(records)
        prediction_frame = pd.concat(predictions, ignore_index=True)
        convergence_record = dict(condition=condition, scenario=scenario, direction=direction,
                                  **convergence)
        enrolment = (evaluate_enrolment(train, test, cfg, attack_config, condition, direction,
                                       rows, prediction_frame, convergence_record)
                     if scenario == "post_deployment" else None)
        print(f"completed {condition} {scenario} {direction}", flush=True)
        return rows, prediction_frame, convergence_record, enrolment


def mechanism(train, test):
    ts = TangentSpace(metric="riemann", tsupdate=False).fit(covariances(train).X)
    Z = ts.transform(covariances(test).X)
    centers, spread = [], []
    for s in np.unique(test.subject):
        z = Z[test.subject == s]
        center = z.mean(axis=0)
        centers.append(center)
        spread.append(np.linalg.norm(z - center, axis=1).mean())
    between = float(pdist(np.stack(centers)).mean())
    within = float(np.mean(spread))
    return dict(between_subject_mean_distance=between, within_subject_mean_distance=within,
                within_over_between=within / between, between_over_within=between / within)


def sealed_lee_plan(path):
    if path is None:
        raise LeakageError("Lee2019 execution blocked until the Lead seals a new plan")
    raw = Path(path).read_bytes()
    plan = json.loads(raw)
    required = dict(experiment="G", dataset="lee2019", status="sealed", allow_session2=True)
    if any(plan.get(k) != v for k, v in required.items()):
        raise LeakageError("Plan must explicitly seal Experiment G Lee2019 session-2 access")
    return hashlib.sha256(raw).hexdigest()


def paired_recall_test(difference, dataset, seed=42):
    if dataset == "bnci":
        return dict(signflip_p=sign_flip_pvalue(difference),
                    signflip_method="exact 2^9 subject signs")
    result = permutation_test(
        (np.asarray(difference),), np.mean, permutation_type="samples",
        alternative="greater", n_resamples=10000, batch=100,
        rng=np.random.default_rng(seed))
    return dict(signflip_p=float(result.pvalue),
                signflip_method="Monte Carlo 10000 subject signs, plus-one correction")


def write_paper_comparison(out, results, metadata):
    """Only matched stated attacker configurations/policies receive a difference."""
    comparison = []
    if metadata["dataset"] == "bnci":
        cfg = metadata["configuration"]
        for key, attackers in PAPER_BNCI.items():
            condition, direction = key.split("_", 1)
            for attacker, published in attackers.items():
                paper_policy = "Fixed independently of Session 2; Section 5"
                ours_policy = "Fixed configuration; no evaluation-session selection"
                if attacker == "paper_ts_svm":
                    paper_config = ours_config = (
                        "OAS covariance; Riemannian tangent space; sklearn-default RBF SVC")
                    matched = True
                    note = ("Matched stated attacker configuration and selection policy only; "
                            "documented preprocessing/defense reproduction ambiguities still apply.")
                elif attacker == "paper_xdawn_svm":
                    selected = ("8" if condition == "undefended" and direction == "S1_to_S2"
                                else "1" if condition == "defended" and direction == "S1_to_S2"
                                else "not separately specified for this row")
                    paper_config = ("OAS XdawnCovariances with evoked filters applied; "
                                    f"nfilter selected by oracle ({selected}); tangent-space RBF SVM")
                    ours_config = (f"OAS augmented covariance and OAS evoked estimator; "
                                   f"applyfilters=True; fixed nfilter={cfg['xdawn_filters']}; "
                                   "Riemannian tangent space; default RBF SVC")
                    paper_policy = (
                        "Oracle maximum on held-out session over nfilter={1,2,4,8}; "
                        "Section 5 reports 8 undefended and 1 defended for BCI S1->S2")
                    matched = False
                    note = ("Difference withheld: paper uses held-out-session oracle selection, "
                            "ours a fixed filter count, even if the selected count coincides.")
                elif attacker in ("paper_csp_svm", "paper_csp_lda"):
                    classifier = "RBF SVC" if attacker == "paper_csp_svm" else "LDA"
                    paper_config = (
                        f"One-versus-rest CSP on trace-normalized covariances; {classifier}; "
                        "component count, variance normalization and solver details unspecified")
                    ours_config = (
                        f"One-versus-rest identity CSP; trace-normalized uncentered covariances; "
                        f"{cfg['csp_pairs']} smallest + {cfg['csp_pairs']} largest eigenvectors "
                        "per identity; log normalized projected variance; no scaling/subbands; "
                        + ("default RBF SVC" if attacker == "paper_csp_svm" else "default SVD LDA"))
                    matched = False
                    note = ("Difference withheld: CSP implementation choices are explicit local "
                            "assumptions; equivalence to the underspecified paper configuration "
                            "cannot be established.")
                else:
                    paper_config = (
                        "SPDNet-ID BiMap c->c/2->c/4; Adam 5e-4; 400 epochs; batch 64; "
                        "patience 40 on 10% stratified training split; seed 42")
                    ours_config, ours_policy = "Not implemented/evaluated", "Not applicable"
                    matched = False
                    note = "Difference withheld: SPDNet-ID was not evaluated."
                match = results[(results.condition == condition) & (results.direction == direction)
                                & (results.attacker == attacker) & (results.k == 1)]
                ours = None if match.empty else float(match.accuracy.iloc[0] * 100)
                comparable = matched and ours is not None
                if ours is None:
                    note = "Difference unavailable: no observed result for this row. " + note
                comparison.append(dict(condition=condition, direction=direction, attacker=attacker,
                                       paper_accuracy_percent=published, observed_accuracy_percent=ours,
                                       paper_attacker_configuration=paper_config,
                                       observed_attacker_configuration=ours_config,
                                       paper_selection_policy=paper_policy,
                                       observed_selection_policy=ours_policy,
                                       comparable=comparable,
                                       difference_pp=ours - published if comparable else None,
                                       comparison_note=note))
        pd.DataFrame(comparison).to_csv(out / "paper_table1_comparison.csv", index=False)
    return comparison


def write_outputs(out, results, predictions, mechanisms, metadata, enrolment_outputs):
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "identity_by_direction.csv", index=False)
    predictions.to_csv(out / "predictions.csv.gz", index=False, compression="gzip")
    avg = results.groupby(["condition", "scenario", "attacker", "k"], as_index=False)[
        ["accuracy", "balanced_accuracy"]].mean()
    avg.to_csv(out / "identity_averaged.csv", index=False)
    comparison = write_paper_comparison(out, results, metadata)
    pd.DataFrame([dict(condition=k, **v) for k, v in mechanisms.items()]).to_csv(
        out / "mechanism.csv", index=False)
    recalls = predictions[predictions.k == 1].assign(
        correct=lambda df: df.subject == df.prediction).groupby(
            ["condition", "scenario", "attacker", "subject"]).correct.mean().unstack("condition")
    paired = []
    for (scenario, attacker), group in recalls.groupby(level=["scenario", "attacker"]):
        difference = group["undefended"] - group["defended"]
        paired.append(dict(scenario=scenario, attacker=attacker,
                           mean_recall_reduction=float(difference.mean()),
                           **paired_recall_test(difference.to_numpy(), metadata["dataset"],
                                                metadata["seed"])))
    pd.DataFrame(paired).to_csv(out / "paired_subject_recall.csv", index=False)
    enrolment = pd.DataFrame([r for output in enrolment_outputs for r in output[0]])
    enrolment.to_csv(out / "enrolment_by_direction.csv", index=False)
    pd.concat([o[1] for o in enrolment_outputs], ignore_index=True).to_csv(
        out / "enrolment_predictions.csv.gz", index=False, compression="gzip")
    pd.concat([o[3] for o in enrolment_outputs], ignore_index=True).to_csv(
        out / "enrolment_training_trials.csv", index=False)
    curve_rows = []
    for key, group in enrolment.groupby(["condition", "attacker", "enrolment_n", "k"]):
        valid = len(group) == 2 and (group.status == "evaluable").all()
        curve_rows.append(dict(condition=key[0], attacker=key[1], enrolment_n=key[2], k=int(key[3]),
                               accuracy=float(group.accuracy.mean()) if valid else None,
                               balanced_accuracy=float(group.balanced_accuracy.mean()) if valid else None,
                               status="evaluable" if valid else "not_evaluable",
                               reason="" if valid else "; ".join(sorted(set(group.reason))),
                               directions_evaluated=int((group.status == "evaluable").sum())))
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(out / "enrolment_averaged.csv", index=False)
    metadata.update(ambiguities_and_deviations=AMBIGUITIES, paper_comparison=comparison,
                    mechanism=mechanisms, identity=avg.to_dict(orient="records"),
                    paired_subject_recall=paired, threat_model_interpretation=THREAT_MODEL_INTERPRETATION,
                    enrolment_curve=curve_rows,
                    enrolment_mlp_convergence=[record for o in enrolment_outputs for record in o[2]])
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, allow_nan=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    colors = plt.get_cmap("tab10")
    for col, scenario in enumerate(("reference_session", "post_deployment")):
        for row, condition in enumerate(("undefended", "defended")):
            ax = axes[row, col]
            part = avg[(avg.scenario == scenario) & (avg.condition == condition)]
            for i, (attacker, values) in enumerate(part.groupby("attacker")):
                ax.plot(values.k, values.balanced_accuracy, marker="o", color=colors(i),
                        label=attacker.replace("paper_", "").replace("beat_", "BEAT "))
            ax.axhline(metadata["chance"], color="black", linestyle=":", label="Chance")
            ax.set_title(f"{scenario.replace('_', ' ').title()} | {condition}")
            ax.set_xticks([1, 5, 10, 20])
            ax.set_ylim(0, 1.03)
            ax.grid(alpha=.2)
            if col == 0:
                ax.set_ylabel("Identity balanced accuracy")
            if row == 1:
                ax.set_xlabel("Consecutive test trials (k)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle(f"Experiment G | {metadata['dataset']} | exploratory, fixed S1 defense\n"
                 "Two directions averaged; fixed 1-filter XDAWN, not paper's oracle")
    fig.tight_layout(rect=(0, .10, 1, .94))
    fig.savefig(out / "identity_vs_k.png", dpi=160)
    plt.close(fig)
    sizes = [str(n) for n in metadata["configuration"]["enrolment_n"]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    names = dict(paper_ts_svm="Paper tangent SVM", beat_logistic="BEAT logistic",
                 beat_rbf_svm="BEAT RBF SVM", beat_mlp="BEAT MLP")
    for row, condition in enumerate(("undefended", "defended")):
        for col, k in enumerate(metadata["configuration"]["enrolment_k"]):
            ax = axes[row, col]
            for i, attacker in enumerate(ENROLMENT_ATTACKERS):
                values = curve[(curve.condition == condition) & (curve.k == k)
                               & (curve.attacker == attacker)].set_index("enrolment_n")
                ax.plot(range(len(sizes)), values.reindex(sizes).balanced_accuracy,
                        marker="o", color=colors(i), label=names[attacker])
            ax.axhline(metadata["chance"], color="black", linestyle=":", label="Chance")
            ax.set_title(f"{condition.title()} | test aggregation k = {k}")
            ax.set_xticks(range(len(sizes)), sizes)
            ax.set_ylim(0, 1.03)
            ax.grid(alpha=.2)
            if col == 0:
                ax.set_ylabel("Identity balanced accuracy")
            if row == 1:
                ax.set_xlabel("Labelled deployment trials per participant (n)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9)
    fig.suptitle(f"Experiment G | {metadata['dataset']} | post-deployment enrolment\n"
                 "Additional deployment-session labels; both chronological directions averaged")
    fig.text(.5, .065, "BEAT RBF SVM / MLP: n < inner_folds not evaluable (not zero). "
             "'all' = entire training half.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .1, 1, .92))
    fig.savefig(out / "enrolment_curve.png", dpi=160)
    plt.close(fig)


def run(dataset, cfg, sealed_plan=None):
    plan_hash = sealed_lee_plan(sealed_plan) if dataset == "lee2019" else None
    if not 1 <= cfg["workers"] <= 3:
        raise ValueError("Shared CPU budget permits 1--3 workers")
    started = time.time()
    np.random.seed(cfg["seed"])
    out = ROOT / "results" / "exp_g" / dataset
    artifacts = ROOT / "artifacts" / "exp_g" / dataset
    artifacts.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        if dataset == "bnci":
            train, reject1 = load_bnci(cfg, "T")
            test, reject2 = load_bnci(cfg, "E")
        else:
            spec = cfg["lee2019"]
            train, reject1 = load_lee2019(ROOT / spec["processed_dir"], spec["subjects"], 1)
            test, reject2 = load_lee2019(ROOT / spec["processed_dir"], spec["subjects"], 2,
                                        allow_session2=True)
        print(f"{dataset}: S1={len(train)}, S2={len(test)}; fitting S1-only defense", flush=True)
        defense = FourStageDefense(cfg[dataset]["strengths"], cfg["sfreq"]).fit(train)
        defended_train, defended_test = defense.transform(train), defense.transform(test)
        joblib.dump(defense, artifacts / "defense.joblib")
        conditions = {"undefended": (train, test), "defended": (defended_train, defended_test)}
        for condition, sessions in conditions.items():
            for session, trials in zip(("T", "E"), sessions):
                np.savez(artifacts / f"{condition}_{session}.npz", X=trials.X,
                         **{c: trials.meta[c].to_numpy() for c in trials.meta})
        mechanisms = {condition: mechanism(*sessions) for condition, sessions in conditions.items()}
    with open(ROOT / "configs" / "exp_b.yaml", encoding="utf-8") as f:
        attack_config = yaml.safe_load(f)["attackers"]
    jobs = []
    for condition, (a, b) in conditions.items():
        half1, half2 = chronological_halves(b)
        for scenario, direction, tr, te in [
            ("reference_session", "S1_to_S2", a, b),
            ("reference_session", "S2_to_S1", b, a),
            ("post_deployment", "first_to_second", half1, half2),
            ("post_deployment", "second_to_first", half2, half1),
        ]:
            jobs.append(delayed(evaluate_split)(tr, te, cfg, attack_config,
                                                condition, scenario, direction))
    print(f"Evaluating {len(jobs)} condition/split jobs, workers={cfg['workers']}", flush=True)
    with parallel_config(backend="loky", n_jobs=cfg["workers"], inner_max_num_threads=1,
                         temp_folder=str(artifacts)):
        outputs = Parallel()(jobs)
    results = pd.DataFrame([row for output in outputs for row in output[0]])
    predictions = pd.concat([output[1] for output in outputs], ignore_index=True)
    metadata = dict(
        experiment="G", dataset=dataset, exploratory=True,
        source="https://arxiv.org/html/2610.15626v1", defense_fit_session="T",
        operating_point=cfg[dataset]["strengths"], configuration=cfg, attackers=attack_config,
        seed=cfg["seed"], workers=cfg["workers"], threads_per_worker=1,
        chance=1 / len(np.unique(train.subject)), session2_plan_sha256=plan_hash,
        lee2019_session2_accessed=dataset == "lee2019",
        retained_trials={"S1": len(train), "S2": len(test)},
        rejected_trials={"S1": reject1, "S2": reject2},
        trial_counts=pd.concat([train.meta, test.meta]).groupby(
            ["subject", "session", "run", "label"]).size().rename("n").reset_index().to_dict("records"),
        mlp_convergence=[output[2] for output in outputs],
        elapsed_seconds=time.time() - started,
        versions={name: importlib.metadata.version(name) for name in
                  ("numpy", "scipy", "scikit-learn", "pyriemann", "joblib")},
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256((ROOT / "configs" / "exp_g.yaml").read_bytes()).hexdigest(),
    )
    write_outputs(out, results, predictions, mechanisms, metadata,
                  [output[3] for output in outputs if output[3] is not None])
    print(f"Finished: {out}; elapsed={metadata['elapsed_seconds']:.1f}s", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("bnci", "lee2019"))
    parser.add_argument("--sealed-plan", type=Path,
                        help="Required only for a later explicitly sealed Lee2019 run")
    args = parser.parse_args(argv)
    with open(ROOT / "configs" / "exp_g.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(args.phase, cfg, args.sealed_plan)


if __name__ == "__main__":
    # Spawned workers and serialized defense resolve beat.exp_g, never __main__.
    from beat.exp_g import main as imported_main
    imported_main()
