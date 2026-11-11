"""User-weighted inference: one matching, its reverse, and distinct-user bootstrap."""

from itertools import permutations, product
import math

import numpy as np


def auc(mated, nonmated):
    """Probability of strict win plus half a tie, without a score Cartesian product."""
    other = np.sort(nonmated)
    return float(np.mean(
        (np.searchsorted(other, mated, side="left") +
         np.searchsorted(other, mated, side="right")) / (2 * len(other))
    ))


def auc_cube(mated, nonmated):
    """[physical reference, assigned mate, assigned partner] AUCs."""
    n = len(mated)
    result = np.empty((n, n, n))
    if all(len(value) == 1 for row in mated + nonmated for value in row):
        a = np.asarray(mated).reshape(n, n)
        b = np.asarray(nonmated).reshape(n, n)
        return (a[:, :, None] > b[:, None, :]).astype(float) + (
            a[:, :, None] == b[:, None, :]) * 0.5
    for ref in range(n):
        lengths = np.array([len(value) for value in mated[ref]])
        starts = np.r_[0, np.cumsum(lengths)[:-1]]
        values = np.concatenate(mated[ref])
        for partner in range(n):
            other = np.sort(nonmated[ref][partner])
            ranks = (np.searchsorted(other, values, side="left") +
                     np.searchsorted(other, values, side="right")) / (2 * len(other))
            result[ref, :, partner] = np.add.reduceat(ranks, starts) / lengths
    return result


def observed_partners(forward, reverse):
    """AUC_uv, averaged over directions from the SAME physical identity matching."""
    n = len(forward)
    ix = np.arange(n)
    result = (forward[ix, ix, :] + reverse[ix, ix, :]) / 2
    np.fill_diagonal(result, np.nan)
    return result


def matching_values(forward, reverse, matches):
    """Vectorised statistic for matchings B_pi(i); reverse reference is B_pi(i)."""
    n = len(forward)
    i, v = np.indices((n, n))
    mask = i != v
    i, v = i[mask], v[mask]
    pi_i, pi_v = matches[:, i], matches[:, v]
    return ((forward[i[None], pi_i, pi_v] +
             reverse[pi_i, i[None], v[None]]) / 2).mean(axis=1)


def matching_test(cubes, *, draws=10_000, seed=0):
    """Independent fold matchings per draw, coupled directions/days/model seeds.

    cubes contains (forward,reverse) already averaged over pairs and model seeds.
    Enumerate the Cartesian product of fold permutations when it is smaller than
    the budget; otherwise use uniform independent draws (identity allowed).
    """
    sizes = [len(a) for a, _ in cubes]
    observed = float(np.average([
        matching_values(a, b, np.arange(len(a))[None])[0] for a, b in cubes
    ], weights=sizes))
    total = math.prod(math.factorial(n) for n in sizes)
    count = min(draws, total)
    rng = np.random.default_rng(seed)
    distribution = np.zeros(count)
    if total <= draws:
        choices = [list(permutations(range(n))) for n in sizes]
        arrangements = list(product(*(range(len(choice)) for choice in choices)))
        for f, ((a, b), n) in enumerate(zip(cubes, sizes)):
            p = np.array([choices[f][arr[f]] for arr in arrangements])
            distribution += n * matching_values(a, b, p)
    else:
        for (a, b), n in zip(cubes, sizes):
            p = np.argsort(rng.random((count, n)), axis=1)
            distribution += n * matching_values(a, b, p)
    distribution /= sum(sizes)
    # The statistic is a sum of finite AUC fractions: allow round-off in ties.
    pvalue = (1 + np.count_nonzero(distribution >= observed - 1e-12)) / (1 + count)
    return {"auc": observed, "p": float(pvalue), "draws": count,
            "exact_matchings": total <= draws}


def bootstrap(matrices, *, replicates=2000, seed=0):
    """Percentile interval with duplicated subjects, never self-pairs.

    Matrices can be partner AUCs or paired differences on identical populations.
    A fold with <2 distinct sampled subjects contributes neither numerator nor
    denominator. Only a replicate with all folds omitted is redrawn.
    """
    rng = np.random.default_rng(seed)
    values = []
    while len(values) < replicates:
        batch = max(32, replicates - len(values))
        numerator, denominator = np.zeros(batch), np.zeros(batch)
        for matrix in matrices:
            n = len(matrix)
            copies = rng.multinomial(n, np.repeat(1 / n, n), size=batch)
            partners = n - copies
            valid = (copies > 0).sum(axis=1) >= 2
            per_user = copies @ np.nan_to_num(matrix).T
            per_user = np.divide(per_user, partners, out=np.zeros_like(per_user, dtype=float),
                                 where=partners > 0)
            numerator[valid] += np.sum(copies[valid] * per_user[valid], axis=1)
            denominator[valid] += n
        keep = denominator > 0
        values.extend((numerator[keep] / denominator[keep]).tolist())
    values = np.asarray(values[:replicates])
    return {
        "interval": np.quantile(values, [0.025, 0.975]).tolist(),
        "replicates": replicates,
        "unreliable": any(len(m) <= 3 for m in matrices),
    }


def holm(pvalues):
    p = np.asarray(pvalues)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        p[order] * np.arange(len(p), 0, -1)
    ))
    return adjusted.tolist()
