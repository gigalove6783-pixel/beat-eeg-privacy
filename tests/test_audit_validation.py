"""A2 CI whole-procedure lane: all three shapes, bidirectional matching, coverage."""

import json

from beat_audit.validate import analytic_checks, run_validation
from beat_audit.null_validation import COMPARATORS, append_validation, run_representative_nulls


def test_validation_analytic_results_serialize_without_numpy_scalars():
    json.dumps(analytic_checks(), allow_nan=False)


def test_ci_whole_procedure_null_coverage_and_contrast():
    report = run_validation(lane="ci", threads=1)
    assert len(report["whole_procedure"]) == 6
    assert report["passed"], {
        "analytic": report["analytic"], "sequential": report["sequential"],
        "cases": [{k: v for k, v in case.items() if k != "simulations_detail"}
                  for case in report["whole_procedure"]],
    }
    null_p = [v["p"] for c in report["whole_procedure"] if c["effect"] == 0
              for v in c["simulations_detail"]]
    assert any(p < 0.95 for p in null_p), "all-one p-values are not a calibrated null procedure"
    planted = [v for c in report["whole_procedure"] if c["effect"] > 0
               for v in c["simulations_detail"]]
    assert sum(v["p"] <= 0.05 for v in planted) > len(planted) / 2


def test_ci_representative_attacks_use_bounded_multiseed_procedure_and_append(tmp_path):
    report = run_representative_nulls(simulations=1, threads=1, lane="ci")
    assert {c["comparator"] for c in report["whole_procedure"]} == set(COMPARATORS)
    observed = []
    for case in report["whole_procedure"]:
        assert case["status"] == "ok", case
        assert case["binomial_99_count_bounds"] == [0, 1]
        value = case["simulations_detail"][0]
        assert value["draws"] == 199
        assert value["comparison_bounds"]["components"] == 12
        assert len(value["inputs_sha256"]) == 3
        assert value["population_sha256"]
        assert 0 < value["p"] <= 1
        observed.append(value["p"])
    assert any(p < 0.95 for p in observed)
    original = {"passed": True, "source": {"sha256": "earlier-source"},
                "whole_procedure": [{"simulations_detail": [{"p": 0.123}]}]}
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(original))
    append_validation(path, report)
    saved = json.loads(path.read_text())
    assert {key: saved[key] for key in original} == original
    assert saved["additional_null_validation"][0] == report
