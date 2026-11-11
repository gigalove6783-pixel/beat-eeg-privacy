"""Exact small-sample tests of the physical matching and subject bootstrap."""

from itertools import permutations, product

import numpy as np
import pytest

from beat_audit import AuditConfig, Condition, Population, Segment, audit_stream
from beat_audit.inference import auc_cube, bootstrap, matching_test


def test_one_matching_defines_both_directions_exactly():
    # Independent oracle enumerates physical score comparisons, not AUC tensors.
    forward = np.array([[0.9, 0.1, 0.4], [0.2, 0.8, 0.6], [0.5, 0.3, 0.7]])
    reverse = np.array([[0.8, 0.4, 0.5], [0.1, 0.9, 0.2], [0.7, 0.6, 0.3]])
    score = lambda x, y: float(x > y) + 0.5 * float(x == y)
    statistic = lambda pi: np.mean([
        (score(forward[i, pi[i]], forward[i, pi[v]]) +
         score(reverse[pi[i], i], reverse[pi[i], v])) / 2
        for i in range(3) for v in range(3) if i != v
    ])
    tables = lambda a: [[np.array([value]) for value in row] for row in a]
    cubes = [(auc_cube(tables(forward), tables(forward)),
              auc_cube(tables(reverse), tables(reverse)))] * 2
    result = matching_test(cubes, draws=10_000, seed=9)
    observed = statistic((0, 1, 2))
    draws = [np.mean([statistic(a), statistic(b)])
             for a, b in product(permutations(range(3)), repeat=2)]
    expected_p = (1 + sum(value >= observed - 1e-12 for value in draws)) / 37
    assert result["exact_matchings"] and result["draws"] == 36
    assert result["auc"] == observed
    assert result["p"] == expected_p


def test_engine_averages_seed_aucs_and_uses_one_shared_matching(tmp_path):
    groups = [np.arange(start, start + 3) for start in (0, 3, 6)]
    inputs, score_matrices = {}, []
    for seed in (8, 19):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(9, 2, 2, 3)) + rng.normal(size=(9, 1, 1, 3))
        path = tmp_path / f"seed-{seed}.npz"
        np.savez(path, outputs=x.reshape(-1, 3),
                 user=np.repeat([f"u{u}" for u in range(9)], 4),
                 day=np.tile(["1", "1", "2", "2"], 9),
                 segment=np.tile(["1", "2"], 18))
        inputs[str(seed)] = path
        matrices = []
        for group in groups:
            dev = np.setdiff1d(np.arange(9), group)
            development = np.concatenate([x[dev, 0, 0], x[dev, 1, 1]])
            a = (x[group, 0, 0] - development.mean(0)) / development.std(0)
            b = (x[group, 1, 1] - development.mean(0)) / development.std(0)
            a /= np.linalg.norm(a, axis=1)[:, None]
            b /= np.linalg.norm(b, axis=1)[:, None]
            matrices.append(a @ b.T)
        score_matrices.append(matrices)

    def rank(a, b):
        return float(a > b) + 0.5 * float(a == b)

    def statistic(matrix, matching):
        return np.mean([
            (rank(matrix[i, matching[i]], matrix[i, matching[j]]) +
             rank(matrix[i, matching[i]], matrix[j, matching[i]])) / 2
            for i in range(3) for j in range(3) if i != j
        ])

    assignments = list(product(list(permutations(range(3))), repeat=3))
    seed_nulls = np.array([
        [np.mean([statistic(m, p) for m, p in zip(matrices, assignment)])
         for assignment in assignments] for matrices in score_matrices
    ])
    observed = seed_nulls[:, 0].mean()
    shared_null = seed_nulls.mean(0)
    expected_p = (1 + np.count_nonzero(shared_null >= observed - 1e-12)) / 217
    independent_null = (seed_nulls[0, :, None] + seed_nulls[1, None, :]) / 2
    independent_p = (1 + np.count_nonzero(independent_null >= observed - 1e-12)) / 46657
    separate_p = np.mean([
        (1 + np.count_nonzero(values >= values[0] - 1e-12)) / 217 for values in seed_nulls
    ])
    assert expected_p == 4 / 217
    assert expected_p != independent_p and expected_p != separate_p
    conditions = [Condition("cross", "cross-day",
                            (((Segment("1", "1"),), (Segment("2", "2"),)),))]
    config = AuditConfig(k_grid=(1,), comparators=("cosine",), bootstraps=39,
                         gallery=False, closed_set=False, sequential=False, threads=1)
    folds = {f"u{u}": u // 3 for u in range(9)}
    report = audit_stream(inputs, conditions=conditions, config=config,
                          population=Population(list(folds), folds))
    result = report["cells"][0]["comparators"]["cosine"]
    assert result["auc"] == pytest.approx(observed, abs=1e-12)
    assert result["p"] == expected_p
    assert result["draws"] == 216


def test_subject_bootstrap_weights_heterogeneous_user_copies():
    a = np.array([[np.nan, 0.1, 0.4], [0.3, np.nan, 0.6], [0.4, 0.7, np.nan]])
    result = bootstrap([a], replicates=2000, seed=3)
    # Exact enumeration of the 24 nondegenerate ordered resamples gives these
    # endpoints. Giving distinct users equal weight instead gives [0.20, 0.65].
    np.testing.assert_allclose(result["interval"], [1 / 6, 2 / 3], atol=1e-12)
    assert result["unreliable"]
    difference = bootstrap([a - a], replicates=2000, seed=4)
    np.testing.assert_allclose(difference["interval"], [0, 0])
