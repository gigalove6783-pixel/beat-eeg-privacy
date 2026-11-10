"""Tests for Experiment R (exploratory addendum on L); synthetic inputs, no dataset access."""
import itertools
import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest
import yaml

from beat import exp_l
from beat import exp_r as R
from beat.config import ROOT

LCFG = yaml.safe_load((ROOT / "configs/exp_l.yaml").read_text(encoding="utf-8"))
SEALED_BUDGET = dict(max_iter=LCFG["attackers"]["mlp_max_iter"],
                     max_fun=LCFG["attackers"]["mlp_max_fun"])


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat([1, 2, 3, 4], 15)
    X = rng.normal(size=(len(y), 4)) + y[:, None] * 0.7
    return X, y


def test_mlp_only_matches_sealed_attacker_fit():
    X, y = _toy()
    sealed = exp_l._fit_attack(X, y, LCFG, 3)["mlp"]
    predictions, fits = R.mlp_only(SEALED_BUDGET)(X, y, X[::2], LCFG, 3)
    assert np.array_equal(predictions["mlp"], sealed.predict(X[::2]))
    assert fits["mlp"]["iterations"] == sealed.convergence_["iterations"]
    assert fits["mlp"]["loss"] == sealed.convergence_["loss"]
    assert fits["mlp"]["converged"] == sealed.convergence_["converged"]


def test_mlp_only_reports_budget_exhaustion_and_convergence():
    X, y = _toy(1)
    _, short = R.mlp_only(dict(max_iter=3, max_fun=10000))(X, y, X, LCFG, 0)
    _, long = R.mlp_only(dict(max_iter=20000, max_fun=200000))(X, y, X, LCFG, 0)
    assert not short["mlp"]["converged"] and short["mlp"]["iterations"] == 3
    assert "failed to converge" in short["mlp"]["warning"]
    assert long["mlp"]["converged"] and long["mlp"]["warning"] == ""


def test_confirmatory_jobs_cover_the_claim_inputs():
    counts = Counter(R.claim_of(job) for job in R.confirmatory_jobs(LCFG))
    assert counts == {"C1": 15, "C2": 30, "C3": 30, "C4": 10}


def test_substitution_replaces_only_refitted_mlp_rows():
    def frame(job, attacker, prediction):
        return pd.DataFrame(dict(subject=[1, 2], run=[3, 3], trial=[1, 2], prediction=prediction,
                                 attacker=attacker, job=job))
    frames = {0: pd.concat([frame(0, "mlp", [1, 1]), frame(0, "logistic", [1, 2])]),
              1: pd.concat([frame(1, "mlp", [2, 2]), frame(1, "logistic", [2, 1])])}
    store = R.substituted(frames, {1: frame(1, "mlp", [1, 2])})
    assert store.select(job=0, attacker="mlp").prediction.tolist() == [1, 1]
    assert store.select(job=1, attacker="mlp").prediction.tolist() == [1, 2]
    assert store.select(attacker="logistic").prediction.tolist() == [1, 2, 2, 1]


def test_component_rows_and_side_by_side():
    test = lambda m: dict(mean=m, lower=m - .1, p=.01, n=54, sign_flip=dict(signflip_p=.02))
    decision = dict(claims=dict(
        C2=dict(component_tests={"a/main": {"b/main": test(.3)}}, utility=test(.0)),
        C3=dict(component_tests={"a/main": dict(threshold=test(.1),
                                                differences={"b/main": test(.2)})})))
    rows = R.component_rows(decision)
    assert [r["component"] for r in rows] == [
        "a/main | b/main", "utility", "a/main | threshold", "a/main | differences | b/main"]
    changed = [dict(r, mean=r["mean"] + .5) for r in rows]
    table = R.side_by_side(rows, changed, ["claim", "component"])
    assert np.allclose(table.mean_change, .5) and np.allclose(table.p_change, 0)


def test_sealed_decision_matches_itself_with_missing_effect_requirements():
    decision = json.loads((exp_l.RESULTS / "decision.json").read_text("utf-8"))
    assert R.max_deviation(decision, decision) == (0.0, True)
    claims = R.side_by_side(R.claim_rows(decision), R.claim_rows(decision), ["claim"])
    assert not claims.decision_changed.any() and not claims.effect_requirement_changed.any()


def test_fit_targets_respects_the_compute_cap(monkeypatch, tmp_path):
    """Refits start only while the longest refit fits; reproductions only while time remains;
    a rerun resumes from the per-job checkpoints."""
    monkeypatch.setattr(R, "CHECKPOINTS", tmp_path)
    jobs = [dict(id=i, kind="representation", key="P3_lambda1", mode="feature", tau=-1.0, k=1,
                 enrolment="all", enrolled=54, scenario="main", direction="offline_to_online",
                 seed=i) for i in range(3)]
    frame = pd.DataFrame(dict(subject=[1, 2], run=[3, 3], trial=[1, 2], prediction=[1, 2],
                              attacker="mlp"))
    clock = itertools.count(0, 100)
    budgets = []

    def fake_refit(job, state, budget):
        budgets.append(budget["max_iter"])
        return frame, dict(converged=True, iterations=7, loss=.5, warning="", other_warnings="",
                           max_iter=budget["max_iter"], max_fun=budget["max_fun"], seconds=1.0,
                           ), dict(peak_mib=1.0)

    monkeypatch.setattr(R.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(R, "refit", fake_refit)
    monkeypatch.setattr(R, "sealed_frame", lambda job, state: frame)
    monkeypatch.setattr(R, "guard", lambda cfg, deadline: 9999.0)
    monkeypatch.setattr(R, "job_ba", lambda frame: .5)
    state = dict(cfg=dict(attackers=dict(mlp_max_iter=500, mlp_max_fun=10000)),
                 convergence={R.fit_key(j): dict(converged=False, iterations=7, loss=.5)
                              for j in jobs})
    cfg = dict(mlp_budget=dict(max_iter=20000, max_fun=200000),
               runtime=dict(first_refit_seconds=200, reproduction_seconds=50))
    rows, replacements = R.fit_targets(jobs, state, cfg, deadline=750)
    # The clock advances 100 per call: jobs 0 and 1 are refit (checks at 0 and 300), job 2 is
    # not (150 left at 600 < 200); job 0 is reproduced (50 left at 700), job 1 not (at 800).
    assert [r["refit_run"] for r in rows] == [True, True, False]
    assert sorted(replacements) == [0, 1]
    assert [r["reproduced"] for r in rows] == [True, None, None]
    assert budgets == [20000, 20000, 500]
    clock = itertools.count(0, 100)
    rows, replacements = R.fit_targets(jobs, state, cfg, deadline=10_000)
    assert [r["refit_run"] for r in rows] == [True, True, True]
    assert all(r["reproduced"] for r in rows) and sorted(replacements) == [0, 1, 2]
    assert replacements[0].equals(frame)
    assert budgets[3:] == [20000, 500, 500]


def test_mlp_refits_are_consistent_with_sealed_records():
    path = R.RESULTS / "mlp_refits.csv"
    if not path.exists():
        pytest.skip("results/exp_r not produced yet")
    fits = pd.read_csv(path)
    diagnostics = json.loads((exp_l.RESULTS / "attacker_diagnostics.json").read_text("utf-8"))
    sealed = {R.fit_key(r): r for r in diagnostics if r.get("record") == "convergence"}
    assert len(fits) == 45 and fits.reproduced.dropna().astype(bool).all()
    for row in fits.to_dict("records"):
        record = sealed[R.fit_key(row)]
        assert not record["converged"] and row["sealed_iterations"] == record["iterations"]
    claims = pd.read_csv(R.RESULTS / "mlp_sensitivity_claims.csv")
    recorded = json.loads((exp_l.RESULTS / "decision.json").read_text("utf-8"))["claims"]
    for row in claims.to_dict("records"):
        assert row["holm_p_sealed"] == pytest.approx(recorded[row["claim"]]["holm_p"], abs=1e-12)
        assert row["decision_changed"] == (row["decision_sealed"] != row["decision_refit"])
