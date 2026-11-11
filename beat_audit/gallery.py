"""Uniform tie-breaking in independence prediction and held-out measurement."""

import numpy as np
from scipy.special import gammaln

from .metrics import normalized


def predict(mated, nonmated, n, wm=None, wn=None):
    """Expected rank-1 probability, exactly integrating a uniformly random tie."""
    order = np.argsort(nonmated)
    scores = np.asarray(nonmated)[order]
    cumulative = np.r_[0, np.cumsum(normalized(wn, len(scores))[order])]
    lo = np.searchsorted(scores, mated, side="left")
    hi = np.searchsorted(scores, mated, side="right")
    a, b = cumulative[lo], cumulative[hi] - cumulative[lo]
    # Integral of (a+b*t)^(N-1), algebraically the specified binomial sum.
    # This avoids overflowing binomial coefficients in larger-N scenarios.
    value = a ** (n - 1)
    positive = b > 0
    top = a[positive] + b[positive]
    ratio = a[positive] / top
    value[positive] = top ** n * (1 - ratio ** n) / (n * b[positive])
    return float(np.dot(normalized(wm, len(mated)), value))


def measure(score_rows, n):
    """[probe,user-candidate] scores; candidate 0 is mated.

    Each row uses ONE common probe and independently sampled reference windows
    from distinct held-out users. Average over subsets and uniform tie draws,
    analytically (Rao-Blackwellisation), rather than privileging the first label.
    """
    size = score_rows.shape[1]
    if not 1 <= n <= size:
        raise ValueError("gallery.N: choose a gallery size between 1 and the held-out fold size")
    mate = score_rows[:, :1]
    lower = np.sum(score_rows[:, 1:] < mate, axis=1)
    tied = np.sum(score_rows[:, 1:] == mate, axis=1)
    choose_log = lambda count, selected: (
        gammaln(count + 1) - gammaln(selected + 1) - gammaln(count - selected + 1))
    denominator = choose_log(size - 1, n - 1)
    probability = np.zeros(len(score_rows))
    # Hypergeometric subset probabilities integrate the exact same galleries
    # without retaining an exponential number of subset-score arrays.
    for j in range(n):
        valid = (tied >= j) & (lower >= n - 1 - j)
        probability[valid] += np.exp(
            choose_log(tied[valid], j) + choose_log(lower[valid], n - 1 - j) - denominator
        ) / (j + 1)
    return float(np.mean(probability))
