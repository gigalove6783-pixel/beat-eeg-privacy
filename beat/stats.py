"""Paired tests and resampling intervals used for every reported comparison."""
import itertools

import numpy as np
from scipy import stats as sps
from sklearn.metrics import balanced_accuracy_score


def sign_flip_pvalue(diffs, alternative: str = "greater") -> float:
    """Exact paired sign-flip permutation p-value of the mean difference (2^n patterns)."""
    d = np.asarray(diffs, dtype=float)
    signs = np.array(list(itertools.product([1.0, -1.0], repeat=d.size)))
    null = (signs * d).mean(axis=1)
    obs = d.mean()
    tol = 1e-12
    if alternative == "greater":
        return float(np.mean(null >= obs - tol))
    if alternative == "less":
        return float(np.mean(null <= obs + tol))
    return float(np.mean(np.abs(null) >= abs(obs) - tol))


def paired_t_lower_bound(diffs, confidence: float = 0.95) -> float:
    """One-sided lower confidence bound of the mean paired difference."""
    d = np.asarray(diffs, dtype=float)
    n = d.size
    return float(d.mean() - sps.t.ppf(confidence, n - 1) * d.std(ddof=1) / np.sqrt(n))


def unit_bootstrap_ba(y_true, y_pred, subject, unit, n_boot: int = 2000, seed: int = 0, confidence: float = 0.95):
    """Percentile interval of balanced accuracy, resampling units (e.g. runs) within each subject."""
    y_true, y_pred, subject, unit = map(np.asarray, (y_true, y_pred, subject, unit))
    groups = [
        [np.flatnonzero((subject == s) & (unit == u)) for u in np.unique(unit[subject == s])]
        for s in np.unique(subject)
    ]
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([units[i] for units in groups for i in rng.integers(0, len(units), len(units))])
        values[b] = balanced_accuracy_score(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(values, [(1 - confidence) / 2, (1 + confidence) / 2])
    return float(lo), float(hi)
