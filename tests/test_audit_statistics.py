"""Analytic A2 checks and the sealed enrolled-likelihood parity contract."""

from itertools import combinations

import numpy as np

from beat_audit.closed_set import enrolled_log_likelihood
from beat_audit.comparators import Comparator, fit_prior, log_bayes_factor, training_pairs
from beat_audit.gallery import measure, predict
from beat_audit.metrics import cllr, cllr_min
from beat_audit.sequential import trajectory
from beat_audit.validate import analytic_checks, sequential_checks


def test_analytic_score_distributions():
    results = analytic_checks()
    assert all(value["passed"] for value in results.values()), results


def test_enrolled_likelihood_matches_sealed_bayes_scores():
    from beat.exp_k import bayes_scores
    windows = np.array([[[1, 0, 2], [0, 2, 1]], [[0, 0, 0], [1, 1, 1]]])
    enrollment = np.array([[[4, 2, 1], [2, 3, 1]], [[1, 3, 2], [4, 0, 1]]])
    alpha = np.array([[0.2, 0.5, 1.3], [0.8, 0.4, 0.6]])
    observed = enrolled_log_likelihood(windows, enrollment, alpha)
    expected = bayes_scores("cat_aware", {"aware": windows}, {"aware": enrollment}, alpha)
    np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=0)
    expected_blind = bayes_scores(
        "cat_blind", {"cat": windows[:, :1]}, {"cat": enrollment[:, :1]}, 0.5)
    np.testing.assert_allclose(enrolled_log_likelihood(
        windows[:, :1], enrollment[:, :1], 0.5), expected_blind, atol=1e-12, rtol=0)


def test_prior_never_observed_command_cue_and_endpoint():
    windows = np.zeros((30, 2, 4))
    windows[np.arange(30), 0, np.arange(30) % 3] = 100
    alpha, info = fit_prior(windows)
    assert np.all(alpha > 0)
    assert info["grid_endpoint"] and info["alpha0"] == 0.1
    assert np.isfinite(log_bayes_factor(windows[0], windows[1], alpha))
    assert log_bayes_factor(windows[0, 1:], windows[1, 1:], alpha[1:]) == 0


def test_uniform_gallery_ties_in_prediction_and_measurement():
    for n in range(2, 12):
        assert abs(predict([0, 0], [0, 0, 0], n) - 1 / n) < 1e-12
        assert abs(measure(np.zeros((4, 11)), n) - 1 / n) < 1e-12


def test_gallery_subset_expectation_scales_without_enumerating_galleries():
    scores = np.array([[1, 1, 2, 0, 0, 1], [3, 0, 0, 0, 0, 0]], dtype=float)
    for n in range(2, 7):
        expected = []
        for subset in combinations(range(1, 6), n - 1):
            for row in scores[:, (0, *subset)]:
                expected.append(0 if np.max(row) > row[0] else 1 / np.sum(row == row[0]))
        np.testing.assert_allclose(measure(scores, n), np.mean(expected), atol=1e-12)
    large = np.zeros((1, 64))
    large[0, 1] = 1
    # Avoid the one superior candidate (probability 32/63), then win a 32-way tie.
    np.testing.assert_allclose(measure(large, 32), 1 / 63, atol=1e-12)


def test_oracle_calibration_perfect_and_tied_scores():
    assert cllr([0, 0], [0, 0]) == 1
    assert cllr_min([1, 1], [0, 0]) == 0
    assert abs(cllr_min([0, 0], [0, 0]) - 1) < 1e-12


def test_sequential_prefix_identity_and_error_characteristics():
    x, alpha = np.array([[8, 2, 0], [1, 1, 2]]), np.full((2, 3), 0.4)
    commands, cues = np.array([0, 2, 1, 0, 1]), np.array([0, 1, 0, 0, 1])
    path = trajectory(x, commands, cues, alpha)
    counts = np.zeros_like(x)
    for t, (command, cue) in enumerate(zip(commands, cues)):
        counts[cue, command] += 1
        np.testing.assert_allclose(path[t], log_bayes_factor(x, counts, alpha), atol=1e-12)
    results = sequential_checks(n=200)
    assert results["ville"]["passed"], results
    assert results["drift_effect"]["passed"], results
    assert "no error-bound assertion" in results["markov"]["interpretation"]


def test_all_feature_attacks_are_frozen_and_training_pairs_bounded():
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(6), 8)
    x = rng.normal(size=(48, 210)) + labels[:, None]
    sides = [({u: x[labels == u][:4] for u in range(6)},
              {u: x[labels == u][4:] for u in range(6)})]
    left, right, binary = training_pairs(sides, list(range(6)), 4)
    assert len(binary) <= 20_000 and set(binary) == {0, 1}
    assert left.shape[1] == right.shape[1] == 210
    assert all(not np.array_equal(a, b) for a, b in zip(left, right))
    for name in ("cosine", "lda_cosine", "pair_logistic"):
        comparator = Comparator(name, seed=4).fit(x, labels, side_pairs=sides)
        before = comparator.scaler.mean_.copy()
        score = comparator.score(x[:4], x[4:8])
        comparator.score(np.full((4, 210), 1e6), x[4:8])
        assert np.isfinite(score).all()
        np.testing.assert_array_equal(comparator.scaler.mean_, before)
