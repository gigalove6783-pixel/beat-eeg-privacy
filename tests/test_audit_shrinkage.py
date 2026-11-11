"""Regression for the interpolation cliff in high-dimensional within-user whitening."""

import numpy as np
import pytest

from beat_audit import AuditConfig, Condition, Population, Segment, audit_timing


def scarce_window_stream(dimension):
    """Rank-three identity with IID noise: no population small-variance nuisance axes.

    Nine development users provide 360/180/72 windows at k=10/20/50. At k=20,
    176 mean or rich dimensions nearly exhaust the 171 residual degrees of
    freedom. Random cue counts avoid an artificial exact linear dependence
    between the global and two cue means.
    """
    rng = np.random.default_rng(73)
    users = [f"u{i:02}" for i in range(12)]
    latent = rng.normal(size=(12, 3))
    loadings = rng.normal(size=(3, dimension))
    mean = 0.35 * latent @ loadings / np.sqrt(3)
    return {
        "outputs": mean[np.repeat(np.arange(12), 400)] + rng.normal(size=(4800, dimension)),
        "user": np.repeat(users, 400),
        "day": np.tile(np.repeat(["1", "2"], 200), 12),
        "segment": np.tile(np.repeat(["1", "2"], 100), 24),
        "order": np.tile(np.arange(200), 24),
        "target": rng.choice(["L", "R"], size=4800),
    }


@pytest.mark.parametrize("dimension,suffix", [(176, ""), (44, "_rich")])
def test_lda_does_not_collapse_at_residual_sample_dimension_boundary(tmp_path, dimension, suffix):
    path = tmp_path / "scarce.npz"
    np.savez(path, **scarce_window_stream(dimension))
    users = [f"u{i:02}" for i in range(12)]
    config = AuditConfig(
        k_grid=(10, 20, 50), comparators=("cosine" + suffix, "lda_cosine" + suffix),
        permutations=199, bootstraps=39, threads=1,
        closed_set=False, gallery=False, sequential=False,
    )
    condition = Condition("cross", "cross-day", (((Segment("1"),), (Segment("2"),)),))
    report = audit_timing(
        path, conditions=[condition], config=config,
        population=Population(users, {u: i // 3 for i, u in enumerate(users)}))
    assert all(c["status"] == "ok" for c in report["cells"])
    results = {c["k"]: c["comparators"] for c in report["cells"]}
    name = "lda_cosine" + suffix
    center = results[20][name]["auc"]
    # Allow one AUC point of sampling variation, not the old 13/21-point cliff.
    assert center >= min(results[10][name]["auc"], results[50][name]["auc"]) - 0.01
    assert center >= results[20]["cosine" + suffix]["auc"] - 0.01
    fit = results[20][name]["training"][0]
    assert fit["within_covariance_estimator"] == "LedoitWolf"
    assert 0 < fit["within_covariance_shrinkage"] <= 1
    assert fit["within_covariance_condition"] < 100
