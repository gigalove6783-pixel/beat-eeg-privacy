"""Independent-review regressions for physical conditions and builder-facing decisions."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from beat_audit import AuditConfig, Condition, Segment, audit_stream
from beat_audit.cli import main
from beat_audit.io import digest
from beat_audit.report import compare_reports
from beat_audit.synthetic import stream_arrays


def _config():
    return AuditConfig(k_grid=(1,), permutations=39, bootstraps=39, threads=1,
                       comparators=("cosine",), gallery=False, sequential=False, closed_set=False)


def _utility(*names):
    return {name: {"task_ba": 0.8, "hold_rate": 0.1, "wrong_move_rate": 0.1} for name in names}


def _decision_report():
    population = {"users": ["a", "b"], "folds": {"a": 0, "b": 0}, "masks": {"0": [0, 1]}}
    return {
        "cohort": "review-fixture", "days": ["1", "2"],
        "cells": [{
            "condition": "same", "relation": "same-day", "days": ["2"], "k": 20,
            "resolved_pairs": [[[["2", "1"]], [["2", "2"]]]],
            "status": "ok", "analysed_users": ["a", "b"], "excluded_users": [],
            "population": population, "population_hash": digest(population),
            "strongest": "cosine", "wording": "linkage detected",
            "comparators": {
                "cosine": {"auc": 0.81, "interval": [0.7, 0.9], "holm_p": 0.08,
                           "wording": "not detected: insufficient evidence, not absence"},
                "lda_cosine": {"auc": 0.80, "interval": [0.7, 0.9], "holm_p": 0.01,
                               "wording": "linkage detected"},
            },
        }],
    }


def test_compare_refuses_different_physical_partitions_not_different_settings(tmp_path):
    arrays = stream_arrays(users=6, per_segment=6, seed=102, effect=0)
    arrays["segment"] = np.tile(np.repeat(["1", "2", "3", "4"], 3), 12)
    path = tmp_path / "four-segments.npz"
    np.savez(path, **arrays)
    condition = lambda a, b: Condition(
        "same-name", "same-day",
        ((tuple(Segment("1", s) for s in a), tuple(Segment("1", s) for s in b)),),
    )
    first = audit_stream(path, conditions=[condition(("1", "2"), ("3", "4"))], config=_config())
    second = audit_stream(path, conditions=[condition(("1", "3"), ("2", "4"))], config=_config())
    a, b = first["cells"][0], second["cells"][0]
    assert a["population_hash"] == b["population_hash"]
    assert a["comparators"]["cosine"]["auc"] != b["comparators"]["cosine"]["auc"]
    with pytest.raises(ValueError, match="condition.*A/B|A/B.*condition"):
        compare_reports({"first": first, "second": second}, _utility("first", "second"))
    equivalent = audit_stream(
        path, conditions=[condition(("2", "1"), ("4", "3"))],
        config=replace(_config(), seed=41, seconds_per_output=0.25),
    )
    rows = compare_reports({"first": first, "equivalent": equivalent},
                           _utility("first", "equivalent"))
    assert len([row for row in rows if row["status"] == "ok"]) == 2


def test_comparison_preserves_cell_detection_and_names_detecting_comparator():
    report = _decision_report()
    row = next(r for r in compare_reports({"rung": report}, _utility("rung"))
               if r["condition"] == "same")
    assert row["auc"] == 0.81 and row["comparator"] == "cosine"
    assert row["wording"] == row["cell_wording"] == "linkage detected"
    assert row["selected_comparator_wording"].startswith("not detected")
    assert row["detected_comparators"] == "lda_cosine"


def test_comparison_marks_ranking_as_target_observed_post_hoc():
    row = compare_reports({"rung": _decision_report()}, _utility("rung"))[0]
    assert "target" in row["comparator_selection"]
    assert "post hoc" in row["comparator_selection"]
    assert "not selected on development" in row["comparator_selection"]


def test_numeric_days_are_chronological_and_explicit_order_is_recorded(tmp_path):
    arrays = stream_arrays(users=6, days=3, per_segment=1, seed=5)
    arrays["day"] = np.array([{"1": "2", "2": "10", "3": "11"}[d] for d in arrays["day"]])
    path = tmp_path / "days.npz"
    np.savez(path, **arrays)
    report = audit_stream(path, config=_config())
    cross = next(c for c in report["configuration"]["conditions"] if c["relation"] == "cross-day")
    assert [(a[0]["day"], b[0]["day"]) for a, b in cross["pairs"]] == [
        ("2", "10"), ("2", "11")]
    assert report["day_order"] == ["2", "10", "11"]
    explicit = audit_stream(path, config=replace(_config(), day_order=("11", "2", "10")))
    assert explicit["day_order"] == ["11", "2", "10"]
    assert explicit["configuration"]["day_order"] == ["11", "2", "10"]
    assert explicit["day_order_source"] == "explicit"


def test_opaque_days_need_order_for_defaults_but_explicit_conditions_still_work(tmp_path):
    arrays = stream_arrays(users=6, per_segment=1, kind="commands")
    arrays["day"] = np.where(arrays["day"] == "1", "T", "E")
    path = tmp_path / "opaque.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="day_order.*--day-order"):
        audit_stream(path, config=_config())
    condition = Condition("explicit", "cross-day", (((Segment("T"),), (Segment("E"),)),))
    report = audit_stream(path, conditions=[condition], config=_config())
    assert report["cells"][0]["status"] == "ok"
    assert report["configuration"]["conditions"][0]["pairs"][0][0][0]["day"] == "T"
    assert report["day_order_source"] == "explicit_conditions"


def test_cli_accepts_and_records_explicit_day_order(tmp_path):
    arrays = stream_arrays(users=6, per_segment=1, kind="commands")
    arrays["day"] = np.where(arrays["day"] == "1", "T", "E")
    path = tmp_path / "opaque.npz"
    np.savez(path, **arrays)
    out = tmp_path / "report"
    assert main(["run", str(path), "--out", str(out), "--k", "1",
                 "--day-order", "T", "E"]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["day_order"] == ["T", "E"]


def test_missing_same_day_is_an_explicit_report_and_comparison_entry(tmp_path):
    arrays = stream_arrays(users=6, per_segment=2, kind="commands")
    arrays["segment"] = np.repeat("only", len(arrays["user"]))
    path = tmp_path / "single-segment.npz"
    np.savez(path, **arrays)
    report = audit_stream(path, config=_config(), out=tmp_path / "report")
    missing = next(c for c in report.get("unavailable_analyses", [])
                   if c["relation"] == "same-day")
    assert missing["status"] == "not_evaluable" and "segment" in missing["reason"]
    assert "| same-day | 1 |" in (tmp_path / "report" / "report.md").read_text()
    rows = compare_reports({"one": report, "two": deepcopy(report)}, _utility("one", "two"))
    same = [r for r in rows if r["relation"] == "same-day"]
    assert len(same) == 2
    assert all(r["status"] == "not_evaluable" and r["auc"] is None and r["reason"] for r in same)
    assert all(r["wording"] != "linkage detected" for r in same)


@pytest.mark.parametrize("command_auc,declared,flagged", [
    (0.90, True, True), (0.70, True, False), (0.90, False, False),
])
def test_declared_derivation_guard_is_not_an_unconditional_auc_claim(command_auc, declared, flagged):
    feature, command = _decision_report(), _decision_report()
    feature["kind"], command["kind"] = "features", "commands"
    command["cells"][0]["comparators"]["cosine"]["auc"] = command_auc
    rows = compare_reports(
        {"features": feature, "commands": command}, _utility("features", "commands"),
        derived_from={"commands": "features"} if declared else None)
    selected = [row for row in rows if row["condition"] == "same"]
    assert all(bool(row.get("data_processing_guard")) == flagged for row in selected)
    if flagged:
        assert all("feature attacker is too weak" in row["data_processing_guard"] for row in selected)
        assert all("at least" in row["data_processing_guard"] for row in selected)
        assert next(r for r in selected if r["alternative"] == "features")["auc"] == 0.81
