"""Small shared helpers for the post-seal-3 modules (exp_k, exp_m); sealed modules stay untouched."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats


# Text outputs use LF on every platform, so files match their git blobs byte for byte
# (Path.write_text and pandas.to_csv default to CRLF on Windows).
def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, allow_nan=False, default=_plain) + "\n")


def write_csv(frame, path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", **kwargs)


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serialisable: {type(value).__name__}")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=_plain).encode()).hexdigest()


def rng_for(*parts) -> np.random.Generator:
    """Deterministic generator from any mix of ints and strings (strings hashed stably)."""
    words = [int(p) if isinstance(p, (int, np.integer)) else
             int.from_bytes(hashlib.sha256(str(p).encode()).digest()[:4], "little")
             for p in parts]
    return np.random.default_rng(np.random.SeedSequence(words))


def t_greater(values, a=0.0, axis=-1):
    """One-sided one-sample t-test p-value for mean(values) > a (vectorised over other axes).

    Zero spread gives p = 0 when the mean exceeds a and 1 otherwise.
    """
    v = np.asarray(values, dtype=float)
    n = v.shape[axis]
    mean, sd = v.mean(axis=axis), v.std(axis=axis, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (mean - a) / (sd / np.sqrt(n))
    p = stats.t.sf(t, n - 1)
    return np.where(sd > 0, p, np.where(mean > a, 0.0, 1.0))


def t_less(values, a=0.0, axis=-1):
    """One-sided one-sample t-test p-value for mean(values) < a."""
    return t_greater(-np.asarray(values, dtype=float), -a, axis=axis)


def t_interval(values, confidence=0.95):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan, np.nan
    half = stats.t.ppf((1 + confidence) / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean() - half), float(v.mean() + half)


def holm(pvalues: dict) -> dict:
    """Holm step-down adjusted p-values."""
    order = sorted(pvalues, key=pvalues.get)
    adjusted, previous = {}, 0.0
    for i, key in enumerate(order):
        previous = max(previous, (len(order) - i) * pvalues[key])
        adjusted[key] = min(1.0, previous)
    return adjusted
