"""Integration: explicit physical sides, model seeds, provenance and A10 refusal."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from beat_audit import AuditConfig, Condition, Population, Segment, audit_stream
from beat_audit.engine import _tables, assign_folds
from beat_audit.io import digest
from beat_audit.report import compare_reports
from beat_audit.synthetic import stream_arrays


def test_explicit_conditions_seed_groups_provenance_and_report_files(tmp_path):
    arrays = stream_arrays(users=6, per_segment=1, kind="features")
    path = tmp_path / "features.npz"
    np.savez(path, **arrays)
    users = sorted(set(arrays["user"]))
    folds, _ = assign_folds(users, 7)
    config = AuditConfig(
        k_grid=(1,), bootstraps=39, permutations=39, gallery=True,
        closed_set=False, sequential=False, comparators=("cosine", "lda_cosine", "pair_logistic"),
        threads=1, cohort="synthetic",
    )
    condition = Condition("explicit", "cross-day",
                          (((Segment("1", "1"),), (Segment("2", "2"),)),))
    report = audit_stream(
        {"a": path, "b": path}, conditions=[condition], config=config,
        population=Population(users, folds), out=tmp_path / "report")
    cell = report["cells"][0]
    assert cell["status"] == "ok", cell.get("reason")
    assert set(cell["comparators"]) == {"cosine", "lda_cosine", "pair_logistic"}
    assert report["hashes"]["inputs"]["a"] == report["hashes"]["inputs"]["b"]
    assert report["hashes"]["source"]["sha256"]
    assert digest(cell["population"]) == cell["population_hash"]
    assert cell["population"]["folds"] == folds
    for name in ("report.json", "report.md", "curve.png", "det.png", "scores.png", "gallery.png"):
        assert (tmp_path / "report" / name).stat().st_size > 50
    single = audit_stream(
        path, conditions=[condition], config=replace(config, comparators=("cosine",), gallery=False),
        population=Population(users, folds))
    assert cell["comparators"]["cosine"]["auc"] == single["cells"][0]["comparators"]["cosine"]["auc"]
    assert cell["comparators"]["cosine"]["p"] == single["cells"][0]["comparators"]["cosine"]["p"]


def test_a10_equal_metadata_different_analysed_users_is_refused():
    population = {"users": ["a", "b"], "masks": {"0": [0, 1]}, "folds": {"a": 0, "b": 0}}
    report = {
        "cohort": "same", "days": ["1", "2"],
        "cells": [{"condition": "same", "relation": "same-day", "days": ["2"], "k": 20,
                   "status": "ok", "analysed_users": ["a", "b"], "population": population,
                   "resolved_pairs": [[[["2", "1"]], [["2", "2"]]]],
                   "wording": "linkage detected",
                   "population_hash": digest(population), "strongest": "cosine",
                   "comparators": {"cosine": {"auc": 0.8, "interval": [0.7, 0.9],
                                               "holm_p": 0.01, "wording": "linkage detected"}}}],
    }
    other = deepcopy(report)
    utility = {"a": {"task_ba": 0.8, "hold_rate": 0.1, "wrong_move_rate": 0.1},
               "b": {"task_ba": 0.9, "hold_rate": 0.2, "wrong_move_rate": 0.05}}
    assert len([r for r in compare_reports({"a": report, "b": other}, utility)
                if r["status"] == "ok"]) == 2
    other["cells"][0]["analysed_users"] = ["a", "different"]
    with pytest.raises(ValueError, match="population_hash"):
        compare_reports({"a": report, "b": other}, utility)


def test_feature_comparisons_are_streamed_in_bounded_chunks():
    class RecordingComparator:
        def __init__(self):
            self.batch_sizes = []

        def score(self, a, b):
            self.batch_sizes.append(len(a))
            return np.sum(a * b, axis=1)

    comparator = RecordingComparator()
    users = [str(i) for i in range(12)]
    sides = [{u: np.ones((20, 210)) for u in users} for _ in range(2)]
    mated, nonmated = _tables(comparator, sides, users, 200, 50, 17)
    assert max(comparator.batch_sizes) <= 4096
    assert len(mated[0][0]) == 200 and len(nonmated[0][1]) == 50
    assert all(np.all(value == 210) for row in mated for value in row)


@pytest.mark.parametrize("field,value", [("cohort", "other"), ("days", ["1", "3"])])
def test_comparison_refuses_different_metadata(field, value):
    population = {"users": ["a"], "masks": {"0": [0]}, "folds": {"a": 0}}
    report = {
        "cohort": "same", "days": ["1", "2"],
        "cells": [{"condition": "cross", "relation": "cross-day", "days": ["1", "2"], "k": 1,
                   "status": "ok", "analysed_users": ["a"], "population": population,
                   "resolved_pairs": [[[["1", "1"]], [["2", "1"]]]], "wording": "none",
                   "population_hash": digest(population), "strongest": "cosine",
                   "comparators": {"cosine": {"auc": 0.5, "interval": [0, 1],
                                               "holm_p": 1, "wording": "none"}}}],
    }
    other = deepcopy(report)
    other[field] = value
    u = {"task_ba": 0.5, "hold_rate": 0.2, "wrong_move_rate": 0.3}
    with pytest.raises(ValueError, match="cohort/days"):
        compare_reports({"a": report, "b": other}, {"a": u, "b": u})


def test_engine_pooled_histograms_weight_users_partners_directions_days_and_seeds(tmp_path):
    users = [f"u{u}" for u in range(6)]
    groups = [users[:3], users[3:]]
    inputs, windows = {}, []
    for seed in range(2):
        rng = np.random.default_rng(51 + seed)
        arrays = {k: [] for k in ("outputs", "user", "day", "segment")}
        blocks = {}
        for u, user in enumerate(users):
            for day in range(2):
                for segment in range(2):
                    n = 1 + int((u + 2 * day + segment + seed) % 3 == 0)
                    value = rng.normal(size=(n, 3))
                    blocks[user, day, segment] = value
                    arrays["outputs"].extend(value)
                    arrays["user"].extend([user] * n)
                    arrays["day"].extend([str(day + 1)] * n)
                    arrays["segment"].extend([str(segment + 1)] * n)
        path = tmp_path / f"weighted-{seed}.npz"
        np.savez(path, **{key: np.asarray(value) for key, value in arrays.items()})
        inputs[str(seed)] = path
        windows.append(blocks)
    mated, nonmated, wm, wn = [], [], [], []
    components = 2 * 2 * 2
    for group in groups:
        for blocks in windows:
            development = np.concatenate([
                blocks[u, day, segment] for u in users if u not in group
                for day in range(2) for segment in range(2)
            ])
            standardized = {
                key: (value - development.mean(0)) / development.std(0)
                for key, value in blocks.items()
            }
            normalized = {key: value / np.linalg.norm(value, axis=1)[:, None]
                          for key, value in standardized.items()}
            for day in range(2):
                for a, b in ((0, 1), (1, 0)):
                    for u in group:
                        for v in group:
                            values = (normalized[u, day, a] @ normalized[v, day, b].T).ravel()
                            if u == v:
                                mated.extend(values)
                                wm.extend([1 / (6 * components * len(values))] * len(values))
                            else:
                                nonmated.extend(values)
                                wn.extend([1 / (6 * components * 2 * len(values))] * len(values))
    edges = np.linspace(min(mated + nonmated) - 1e-9, max(mated + nonmated) + 1e-9, 51)
    expected_m = np.histogram(mated, edges, weights=wm)[0]
    expected_n = np.histogram(nonmated, edges, weights=wn)[0]
    assert not np.allclose(expected_m, np.histogram(mated, edges)[0] / len(mated))
    assert not np.allclose(expected_n, np.histogram(nonmated, edges)[0] / len(nonmated))
    condition = Condition("same", "same-day", tuple(
        ((Segment(str(day), "1"),), (Segment(str(day), "2"),)) for day in (1, 2)))
    config = AuditConfig(k_grid=(1,), comparators=("cosine",), permutations=39,
                         bootstraps=39, gallery=False, closed_set=False, sequential=False, threads=1)
    report = audit_stream(inputs, conditions=[condition], config=config,
                          population=Population(users, {u: i // 3 for i, u in enumerate(users)}))
    assert report["cells"][0]["status"] == "ok"
    actual = report["cells"][0]["comparators"]["cosine"]["histogram"]
    np.testing.assert_allclose(actual["edges"], edges, atol=1e-12)
    np.testing.assert_allclose(actual["mated"], expected_m, atol=1e-12)
    np.testing.assert_allclose(actual["nonmated"], expected_n, atol=1e-12)
