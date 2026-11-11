"""Cue/spread-only identity, frozen rich attackers, and the public timing boundary."""

from dataclasses import replace

import numpy as np
import pytest

from beat_audit import AuditConfig, Condition, Population, Segment, audit_stream, audit_timing
from beat_audit.io import from_arrays, make_windows
from beat_audit.synthetic import stream_arrays


def _condition():
    return Condition("cross", "cross-day", (((Segment("1", "1"),), (Segment("2", "2"),)),))


def _settings():
    return AuditConfig(k_grid=(20,), permutations=1999, bootstraps=39, threads=1,
                       closed_set=False, sequential=False, gallery=False)


@pytest.mark.parametrize("identity", ["cue", "variance"])
def test_rich_detects_identity_invisible_to_window_means(tmp_path, identity):
    rng = np.random.default_rng(520)
    arrays = stream_arrays(users=12, per_segment=40, effect=0, heterogeneous=False)
    vectors = (2 * ((np.arange(12)[:, None] >> np.arange(8)) % 2) - 1
               if identity == "cue" else rng.integers(1, 20, size=(12, 8)))
    user_index = np.repeat(np.arange(12), 160)
    positive = arrays["order"] % 20 < 10
    arrays["outputs"] = vectors[user_index] * np.where(positive, 1, -1)[:, None]
    if identity == "cue":
        arrays["target"] = np.where(positive, "left", "right")
    path = tmp_path / "signal.npz"
    np.savez(path, **arrays)
    report = audit_stream(path, conditions=[_condition()], config=_settings())
    cell = report["cells"][0]
    assert cell["status"] == "ok", cell.get("reason")
    for name in ("cosine", "lda_cosine", "pair_logistic"):
        assert cell["comparators"][name]["auc"] == 0.5
        assert cell["comparators"][name]["p"] == 1
    for name in ("cosine_rich", "lda_cosine_rich"):
        assert cell["comparators"][name]["auc"] > 0.95
        assert cell["comparators"][name]["p"] < 0.05
    if identity == "cue":
        np.testing.assert_array_equal(np.var(arrays["outputs"].reshape(-1, 20, 8), axis=1), 1)


def test_rich_summary_fallback_floor_and_k_one_skip(tmp_path):
    arrays = stream_arrays(users=6, per_segment=10)
    arrays["outputs"] = np.full((len(arrays["user"]), 2), 3.0)
    arrays["target"] = np.repeat("visible", len(arrays["user"]))
    stream = from_arrays(arrays)
    rich = make_windows(stream, 5, ["visible", "absent"], summary="rich")
    for values in rich.values():
        np.testing.assert_allclose(values[:, :6], 3)
        np.testing.assert_allclose(values[:, 6:], np.log(1e-12))
    one = make_windows(stream, 1, ["visible", "absent"], summary="rich")
    for values in one.values():
        assert values.shape[1] == 2
    path = tmp_path / "constant.npz"
    np.savez(path, **arrays)
    report = audit_stream(path, conditions=[_condition()],
                          config=replace(_settings(), k_grid=(1, 5)), out=tmp_path / "report")
    assert set(report["cells"][0]["comparators"]) == {"cosine", "lda_cosine", "pair_logistic"}
    assert set(report["cells"][0]["skipped_comparators"]) == {"cosine_rich", "lda_cosine_rich"}
    assert {"cosine_rich", "lda_cosine_rich"} <= set(report["cells"][1]["comparators"])
    assert (tmp_path / "report" / "curve.png").stat().st_size > 50


def test_public_timing_keeps_full_development_and_rich_fits_ignore_targets(tmp_path):
    arrays = stream_arrays(users=12, per_segment=40)
    arrays["target"] = np.where(arrays["order"] % 2, "left", "right")
    users = sorted(set(arrays["user"]))
    population = Population(users, {u: i // 3 for i, u in enumerate(users)})
    path = tmp_path / "features.npz"
    np.savez(path, **arrays)
    config = replace(_settings(), comparators=("cosine_rich", "lda_cosine_rich"))
    before = audit_timing(path, folds=(1,), conditions=[_condition()],
                          config=config, population=population)
    changed = np.isin(arrays["user"], users[3:6])
    arrays["outputs"][changed] = np.random.default_rng(17).normal(
        1e4, 1e3, size=arrays["outputs"][changed].shape)
    np.savez(path, **arrays)
    after = audit_timing(path, folds=(1,), conditions=[_condition()],
                         config=config, population=population)
    assert before["purpose"] == "timing_only"
    assert before["timing_folds"] == [1] and before["peak_rss_bytes"] > 0
    for name in config.comparators:
        a, b = [r["cells"][0]["comparators"][name] for r in (before, after)]
        assert a["timing_seconds"] > 0
        assert a["training"] == b["training"]
        assert [x["fold"] for x in a["training"]] == [1]
        assert a["training"][0]["fitted_parameters_sha256"]
        assert [x["calibration_parameters"] for x in a["operating_points"]] == [
            x["calibration_parameters"] for x in b["operating_points"]]
        assert [x["development_eer_threshold"] for x in a["operating_points"]] == [
            x["development_eer_threshold"] for x in b["operating_points"]]
        assert a["histogram"] != b["histogram"]
        assert a["partner_auc"][0]["users"] == users[3:6]
    from beat_audit.report import compare_reports
    utility = {"pilot": {"task_ba": 0.5, "hold_rate": 0.1, "wrong_move_rate": 0.1}}
    with pytest.raises(ValueError, match="timing"):
        compare_reports({"pilot": before}, utility)
