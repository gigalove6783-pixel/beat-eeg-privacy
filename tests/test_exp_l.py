"""Lee2019 confirmatory contracts; real session 2 is never used."""
import copy
import json
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from beat.config import load_config
from beat.data import LeakageError, Trials, bandpass_sos
from beat.exp_c import fit_temperature, identify_head
from beat.exp_g import FourStageDefense
from beat.exp_l import (candidate_vectors, grouped, holm, infer_claims, paired,
                        participant_distribution, run_dev, run_final, session1_only,
                        verify_seal, attack, _enrolment, _fit_attack,
                        analysis_jobs, attack_job_count, append_predictions,
                        replay_control, summary_rows, head_summary_rows)
from beat.attacks import fit_attackers
from beat.features import covariances
from beat.lee2019 import load_trials
from beat.models import FeatureMap, fit_defended, fit_encoder, fit_linear
from scripts.prepare_lee2019 import MOTOR, reduce_phase


def small_config():
    cfg = copy.deepcopy(load_config("exp_l"))
    cfg.update(subjects=[1, 2, 3], seeds=[0], descriptive_seeds=[0], workers=2,
               dimensions=[1, 8], lambdas=[.1, 1., 3.],
               enrolled_counts=[2, 3])
    cfg["training"].update(max_epochs=2, encoder_hidden=8, identity_hidden=4,
                           batch_size=128)
    cfg["attackers"].update(svm_C=[1.], svm_gamma=["scale"], inner_folds=2,
                            mlp_hidden=[4], mlp_max_iter=15, mlp_max_fun=50)
    cfg["statistics"]["bootstrap_samples"] = 40
    return cfg


def test_direct_fixed_estimators_match_single_candidate_refit():
    cfg = small_config()
    rng = np.random.default_rng(27)
    X = rng.normal(size=(60, 5))
    subjects = np.repeat([1, 2, 3], 20)
    X[:, 0] += subjects
    test = rng.normal(size=(21, 5))
    with threadpool_limits(limits=1):
        expected = fit_attackers(X, subjects, cfg["attackers"], 0)
        direct = _fit_attack(X, subjects, cfg, 0)
    for name in ("logistic", "rbf_svm", "mlp"):
        np.testing.assert_array_equal(direct[name].predict(test), expected[name].predict(test))
    np.testing.assert_allclose(direct["rbf_svm"].decision_function(test),
                               expected["rbf_svm"].decision_function(test), rtol=1e-12)
    assert isinstance(direct["rbf_svm"].steps[-1][1], __import__("sklearn.svm", fromlist=["SVC"]).SVC)


def test_restricted_manifest_keeps_confirmatory_seeds_and_exact_descriptive_cells():
    cfg = load_config("exp_l")
    jobs = pd.DataFrame(analysis_jobs(cfg))
    assert attack_job_count(cfg) == dict(representation=90, feature_groups=108, command_groups=228,
                                         enrolment=112, N_scaling=48, benchmark=2, total=588)
    representation = jobs[jobs.kind == "representation"]
    expected_keys = {"P0", "P2_d1", "P2_d8", "P2-RC_d1", "P2-RC_d8",
                     "P3_lambda0.1", "P3_lambda1", "P3_lambda3"}
    assert set(representation.key) == expected_keys
    for key, part in representation.groupby("key"):
        assert set(part.seed) == (set(range(5)) if key in ("P0", "P2-RC_d8", "P3_lambda1") else {0, 1, 2})
        assert set(part.direction) == {"T_to_E", "offline_to_online", "online_to_offline"}
    obs = jobs[jobs.kind == "observation"]
    assert set(obs.key) == {"P0", "P2-RC_d8", "P3_lambda1"}
    assert set(obs[obs["mode"] == "command"].tau) == {.5, .8}
    assert set(obs.k) == {1, 5, 10, 20}
    for (key, mode, tau, k), part in obs.groupby(["key", "mode", "tau", "k"]):
        primary = key == "P2-RC_d8" and mode == "command" and tau == .5 and k in (1, 20)
        assert set(part.seed) == (set(range(5)) if primary else {0, 1, 2})
    enrol = jobs[jobs.kind == "enrolment"]
    assert set(enrol.key) == {"G", "G-raw", "P0-RC"}
    for (key, n), part in enrol.groupby(["key", "enrolment"]):
        assert set(part.seed) == (set(range(5)) if key == "G" and n == "5" else {0, 1, 2})
    scaled = jobs[jobs.kind == "enrolled"]
    assert set(scaled.key) == {"P0", "P2-RC_d8"}
    assert set(scaled.enrolled) == {9, 18, 27, 54}
    assert set(scaled.scenario) == {"main"}
    assert set(scaled.seed) == {0, 1, 2}


def test_prediction_csv_preserves_fields_across_worker_column_orders(tmp_path):
    first = pd.DataFrame([dict(subject=1, run=1, trial=1, prediction=1,
                               kind="representation", key="P0", mode="feature", tau=-1., k=1,
                               enrolment="all", enrolled=54, scenario="auxiliary",
                               direction="T_to_E", seed=0, attacker="logistic")])
    second = first.assign(kind="benchmark", key="G", attacker="paper_ts_svm")
    columns = list(second.columns)
    columns.remove("seed")
    columns.insert(columns.index("scenario"), "seed")
    path = tmp_path/"predictions.csv.gz"
    append_predictions(path, first, True)
    append_predictions(path, second[columns], False)
    restored = pd.read_csv(path)
    assert restored.seed.tolist() == [0, 0]
    assert restored.scenario.tolist() == ["auxiliary", "auxiliary"]
    assert restored.direction.tolist() == ["T_to_E", "T_to_E"]


def synthetic(session="T"):
    rng = np.random.default_rng(42 if session == "T" else 43)
    rows, signals = [], []
    for subject in (1, 2, 3):
        for run in (1, 2, 3, 4):
            for position in range(1, 26):
                label = position % 2
                X = rng.normal(size=(4, 80))
                X[subject-1] *= 1.3
                X[label] *= 1.5
                signals.append(X)
                rows.append(dict(subject=subject, session=session, run=run,
                                 trial=position + (25 if run in (2, 4) else 0),
                                 label=label, artifact=False, finite=True))
    return Trials(np.asarray(signals, dtype=np.float32), pd.DataFrame(rows))


def write_reduced(path, session=1):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 20, 750)).astype(np.float32)
    np.savez(path / f"sess{session}_subj01.npz", X=X, label=np.tile([0, 1], 100),
             phase=np.repeat(["train", "test"], 100), trial=np.tile(np.arange(1, 101), 2),
             finite=np.ones(200, dtype=bool))


def test_loader_blocks_order_and_pre_io_seal(tmp_path, monkeypatch):
    write_reduced(tmp_path)
    t = load_trials(tmp_path, [1], 1)
    np.testing.assert_array_equal(t.run, np.repeat([1, 2, 3, 4], 50))
    np.testing.assert_array_equal(t.meta.trial, np.tile(np.arange(1, 101), 2))
    assert set(t.session) == {"T"}
    assert not t.meta.artifact.any()
    assert t.meta.finite.all()
    opened = []
    def forbidden(*args, **kwargs):
        opened.append(args[0])
        raise AssertionError("Guard ran too late")
    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(LeakageError):
        load_trials(tmp_path, [1], 2)
    assert not opened


def test_reducer_nan_remains_excluded_after_zero_filling(tmp_path):
    cfg = load_config("data")
    raw = np.zeros((400000, len(MOTOR)), dtype=np.float32)
    raw[500, 0] = np.nan  # First trial's guard, before its 1–4 s epoch.
    run = SimpleNamespace(x=raw, chan=MOTOR, t=1+np.arange(100)*4000,
                          y_dec=np.tile([2, 1], 50))
    sos = bandpass_sos(dict(cfg, sfreq=1000))
    dirty, labels, finite = reduce_phase(run, sos, cfg)
    raw[500, 0] = 0.
    clean, _, clean_finite = reduce_phase(run, sos, cfg)
    assert not finite[0] and finite[1:].all() and np.isfinite(dirty).all()
    arrays = dict(X=np.concatenate([dirty, clean]), label=np.tile(labels, 2),
                  finite=np.r_[finite, clean_finite], phase=np.repeat(["train", "test"], 100),
                  trial=np.tile(np.arange(1, 101), 2))
    path = tmp_path/"sess1_subj01.npz"
    np.savez(path, **arrays)
    loaded = load_trials(tmp_path, [1], 1)
    assert loaded.meta.finite.sum() == 199 and not loaded.meta.finite.iloc[0]
    assert np.isfinite(loaded.X).all()
    arrays["X"][1, 0, 0] = np.nan
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="retained"):
        load_trials(tmp_path, [1], 1)


def test_control_episodes_do_not_bridge_excluded_trials():
    meta = pd.DataFrame(dict(subject=1, session="E", run=1, trial=np.arange(1, 41),
                             label=1, artifact=False, finite=np.arange(1, 41) != 10))
    scores = np.tile([-10., 10.], (40, 1))
    controller = small_config()["controller"]
    episodes, commands, angles, _ = replay_control(meta, scores, 1., .5, controller)
    assert episodes[["first_trial", "last_trial"]].values.tolist() == [[11, 30]]
    assert episodes.trials.tolist() == [20] and len(commands) == 20
    assert angles[0] == 5.
    complete = meta.assign(finite=True)
    _, _, reset_angles, _ = replay_control(complete, scores, 1., .5, controller)
    assert reset_angles[0] == reset_angles[20] == 5.
    empty, commands, _, _ = replay_control(meta.assign(finite=False), scores, 1., .5, controller)
    assert empty.empty and len(commands) == 0


@pytest.mark.parametrize("missing", ["participant", "seed", "direction", "fixed_subset"])
def test_incomplete_summary_never_reduces_declared_cells(missing):
    cfg = load_config("exp_l")
    tag = dict(kind="representation", key="P0", mode="feature", tau=-1.,
               k=1, enrolment="all", enrolled=54)
    subjects, seeds, directions = cfg["subjects"], cfg["seeds"], ["T_to_E"]
    scenario = "auxiliary"
    if missing in ("direction", "fixed_subset"):
        scenario, directions = "main", ["offline_to_online", "online_to_offline"]
    if missing == "fixed_subset":
        tag.update(kind="enrolled", enrolled=9)
        subjects = sorted(np.random.default_rng(42).permutation(cfg["subjects"])[:9])
        seeds = cfg["descriptive_seeds"]
    frame = pd.DataFrame([dict(subject=s, prediction=s, seed=seed, direction=direction,
                               attacker="logistic", scenario=scenario)
                          for s in subjects for seed in seeds for direction in directions])
    if missing in ("participant", "fixed_subset"):
        broken = frame[frame.subject != subjects[-1]].copy()
        if missing == "fixed_subset":
            replacement = next(s for s in cfg["subjects"] if s not in subjects)
            extra = frame[frame.subject == subjects[-1]].assign(subject=replacement, prediction=replacement)
            broken = pd.concat([broken, extra], ignore_index=True)
    elif missing == "seed":
        broken = frame[frame.seed != seeds[-1]]
    else:
        broken = frame[frame.direction != directions[-1]]
    row = next(r for r in summary_rows(tag, broken, cfg)
               if r["attacker"] == "logistic" and r["scenario"] == scenario)
    assert row.get("status", "ok").startswith("not_evaluable")
    assert pd.isna(row["ba"]) and pd.isna(row["ci_low"]) and pd.isna(row["ci_high"])
    assert not row["strongest"] and json.loads(row["missing_cells"])
    restored = next(r for r in summary_rows(tag, frame.iloc[::-1], cfg)
                    if r["attacker"] == "logistic" and r["scenario"] == scenario)
    assert restored["status"] == "ok"
    assert restored["ba"] == restored["ci_low"] == restored["ci_high"] == 1.


def test_all_defended_fits_reject_e_before_preprocessing():
    cfg = small_config()
    e = synthetic("E")
    with threadpool_limits(limits=1):
        C = covariances(e)
        fits = [
            lambda: FeatureMap(False).fit(C),
            lambda: FeatureMap(True).fit(C),
            lambda: fit_defended(C, "P2", cfg["training"], 0, 1),
            lambda: fit_linear(C, FeatureMap(False), cfg["training"], 0),
            lambda: fit_encoder(C.with_X(np.zeros((len(C), 3))), cfg["training"], 0, 1, 1.),
            lambda: FourStageDefense(cfg["g_strengths"]).fit(e),
            lambda: fit_temperature(e.with_X(np.zeros((len(e), 2))), [.05, 20.]),
            lambda: identify_head(e.with_X(np.zeros((len(e), 2))), np.zeros((len(e), 2)), .001),
            lambda: run_dev(e, cfg),
            lambda: run_final(e, e, cfg, {})]
        for fit in fits:
            with pytest.raises(LeakageError):
                fit()


def test_absent_head_candidates_are_reported_not_evaluable():
    heads = pd.DataFrame(columns=["key", "subject", "seed", "direction", "prediction"])
    independent = pd.DataFrame(columns=["key", "kind", "strongest", "ba"])
    rows = head_summary_rows(heads, independent, small_config())
    assert {r["key"] for r in rows} == {"P3_lambda0.1", "P3_lambda1", "P3_lambda3"}
    assert all(r["status"] == "not_evaluable_incomplete_cells" for r in rows)
    assert all(r["head_ba"] is None and json.loads(r["missing_cells"]) for r in rows)


def test_missing_sixth_candidate_blocks_strongest_head_plots_and_claim(tmp_path):
    from beat.exp_l import PredictionStore, make_decision
    cfg = load_config("exp_l")
    cfg["statistics"]["bootstrap_samples"] = 40
    tag = dict(kind="representation", key="P3_lambda1", mode="feature",
               tau=-1., k=1, enrolment="all", enrolled=54)
    records = []
    for attacker in ("logistic", "rbf_svm", "mlp"):
        for scenario, directions in (("auxiliary", ["T_to_E"]),
                                     ("main", ["offline_to_online", "online_to_offline"])):
            for seed in cfg["seeds"]:
                for direction in directions:
                    for subject in cfg["subjects"]:
                        for trial in (1, 2):
                            correct = trial == 1 or (attacker == "mlp" and scenario == "main")
                            records.append(dict(tag, attacker=attacker, scenario=scenario, seed=seed,
                                                direction=direction, subject=subject, trial=trial,
                                                run=3 if direction == "offline_to_online" else 1,
                                                prediction=subject if correct else subject % 54+1))
    complete = pd.DataFrame(records)
    missing = ((complete.attacker == "mlp") & (complete.scenario == "main") &
               (complete.subject == 54) & (complete.seed == 0) &
               (complete.direction == "offline_to_online"))
    broken = complete.loc[~missing]
    summary = pd.DataFrame(summary_rows(tag, broken, cfg, descriptive=False))
    assert len(summary[summary.status == "ok"]) == 5
    assert summary.loc[summary.status == "ok", "ba"].eq(.5).all()
    assert not summary.strongest.any()
    assert summary.strongest_ba.isna().all()
    assert summary.strongest_status.eq("not_evaluable_incomplete_candidate_set").all()
    absent = json.loads(summary.missing_candidates.iloc[0])
    assert [(r["attacker"], r["scenario"]) for r in absent] == [("mlp", "main")]
    assert absent[0]["missing_cells"] == [dict(seed=0, direction="offline_to_online", subjects=[54])]

    heads = complete[(complete.attacker == "logistic") & (complete.scenario == "main")].copy()
    heads["prediction"] = heads.subject % 54+1
    descriptive = pd.DataFrame(summary_rows(tag, broken[broken.seed.isin([0, 1, 2])], cfg, descriptive=True))
    comparison = next(r for r in head_summary_rows(heads, descriptive, cfg) if r["key"] == "P3_lambda1")
    assert comparison["head_ba"] == 0.
    assert comparison["independent_ba"] is None
    assert comparison["status"] == "not_evaluable_incomplete_candidate_set"
    assert json.loads(comparison["missing_candidates"]) == absent

    from beat.exp_l import strongest_plot_rows
    plot = strongest_plot_rows(summary, "representation")
    assert len(plot) == 1 and plot.ba.isna().all()
    assert plot.strongest_status.eq("not_evaluable_incomplete_candidate_set").all()

    store = PredictionStore(tmp_path/"predictions")
    store.append(broken)
    for key in ("P0", "P2-RC_d8"):
        store.append(complete.assign(key=key))
    for k in (1, 20):
        store.append(complete.assign(kind="observation", key="P2-RC_d8", mode="command", tau=.5, k=k))
    enrol = complete[(complete.attacker == "logistic") & (complete.scenario == "main")]
    for attacker in ("paper_ts_svm", "beat_logistic", "beat_rbf_svm", "beat_mlp"):
        store.append(enrol.assign(kind="enrolment", key="G", enrolment="5", attacker=attacker))
    task = pd.DataFrame([dict(key=key, seed=seed, subject=subject, ba=.75)
                         for key in ("P0", "P2-RC_d8") for seed in cfg["seeds"] for subject in cfg["subjects"]])
    claim = make_decision(store, task, heads, {"accuracy": .024}, cfg)["claims"]["C1"]
    assert claim["decision"] == "not_evaluable_missing_cells" and claim["p"] == 1.
    assert claim["observed_difference"] is None and claim["effect_requirement"] is None
    assert claim["component_tests"] is None and claim["missing_cells"]


def test_phase_groups_can_cross_blocks_but_not_phases():
    trial = synthetic().subset(np.arange(100))
    messages = np.arange(100)[:, None].astype(float)
    X, meta = grouped(messages, trial.meta, 20, "feature", 20)
    # Two windows per 50-trial phase; window 21..40 spans blocks 1/2.
    np.testing.assert_allclose(X[:, 0], [9.5, 29.5, 59.5, 79.5])
    assert meta.run.tolist() == [1, 1, 3, 3]
    assert meta.last_trial.tolist() == [20, 40, 20, 40]
    shortened = trial.subset(trial.meta.trial != 10)
    _, m = grouped(np.zeros((len(shortened), 1)), shortened.meta, 20, "feature", 1)
    assert not ((m.trial < 10) & (m.last_trial > 10)).any()


def test_strongest_is_one_global_classifier_not_subjectwise_or_seedwise():
    cfg = small_config()
    cfg.update(subjects=[1, 2], seeds=[0, 1], descriptive_seeds=[0, 1])
    tag = dict(kind="representation", key="P0", mode="feature", tau=-1.,
               k=1, enrolment="all", enrolled=2)
    frames = []
    for attacker, correctness in (("logistic", [[1, 1, 1], [0, 0, 0]]),
                                  ("mlp", [[0, 0, 0], [1, 1, 0]]),
                                  ("rbf_svm", [[0, 0, 0], [1, 1, 0]])):
        for scenario, directions in (("auxiliary", ["T_to_E"]),
                                     ("main", ["offline_to_online", "online_to_offline"])):
            for direction in directions:
                for seed in (0, 1):
                    for subject in (1, 2):
                        for trial, correct in enumerate(correctness[subject-1], 1):
                            frames.append(dict(attacker=attacker, scenario=scenario, seed=seed,
                                               direction=direction, subject=subject, run=1, trial=trial,
                                               prediction=subject if correct else 3-subject))
    rows = summary_rows(tag, pd.DataFrame(frames), cfg)
    winners = [row for row in rows if row["strongest"]]
    assert len(winners) == 1 and winners[0]["attacker"] == "logistic"
    assert winners[0]["ba"] == .5 and all(row["strongest_ba"] == .5 for row in rows)


def test_paired_holm_and_paired_participant_bootstrap():
    assert paired([.2, .2, .2])["lower"] == pytest.approx(.2)
    assert paired([-.04, -.04, -.04], -.05)["p"] == 0.
    assert holm({"C1": .01, "C2": .03, "C3": .04, "C4": .2}) == {
        "C1": .04, "C2": .09, "C3": .09, "C4": .2}
    first = participant_distribution([0., .25, 1.], small_config())
    second = participant_distribution([1., .75, 0.], small_config())
    np.testing.assert_allclose(first+second, 1.)
    assert np.ptp(first) > 0


def test_revised_claim_formulas_match_independent_t_tests():
    from scipy.stats import ttest_1samp
    cfg = small_config()
    cfg["subjects"] = list(range(1, 55))
    x = np.linspace(-.03, .03, 54)
    vec = lambda base, j: base + x*(j+1)
    r0 = {str(j): vec(.6+j*.005, j) for j in range(6)}
    r2 = {str(j): vec(.2+j*.003, j) for j in range(6)}
    r3 = {str(j): vec(.45+j*.003, j) for j in range(6)}
    h, u = vec(.1, 0), vec(-.01, 0)
    c1 = {str(j): vec(.04+j*.002, j) for j in range(6)}
    c20 = {str(j): vec(.08+j*.003, j) for j in range(6)}
    g = {str(j): vec(.3+j*.004, j) for j in range(4)}
    t = lambda v, a=0.: float(ttest_1samp(v, a, alternative="greater").pvalue)
    expected = [
        min(1., 6*min(t(v-h) for v in r3.values())),
        max(t(u, -.05), min(1., 6*min(max(t(v-w) for w in r2.values()) for v in r0.values()))),
        min(1., 6*min(max(t(v, 2/54), max(t(v-w) for w in c1.values())) for v in c20.values())),
        min(1., 4*min(t(v, 10/54) for v in g.values()))]
    actual = infer_claims(r0, r2, r3, h, u, c1, c20, g, .024, cfg)
    np.testing.assert_allclose([actual[f"C{i}"]["p"] for i in range(1, 5)], expected, rtol=1e-10)
    low = {key: value-.3 for key, value in r3.items()}
    failed = infer_claims(r0, r0, low, h, u, c1, c20, g, .2, cfg)
    assert failed["C1"]["p"] == failed["C2"]["p"] == failed["C4"]["p"] == 1.
    assert not failed["C4"]["benchmark_gate"]


def test_missing_candidate_cells_do_not_shrink_family():
    rows = []
    for subject in (1, 2):
        rows.append(dict(attacker="logistic", scenario="auxiliary", seed=0,
                         direction="T_to_E", subject=subject, prediction=subject))
    vectors, missing = candidate_vectors(pd.DataFrame(rows), small_config())
    assert len(vectors) == 6
    assert any(row["candidate"] == "logistic/auxiliary" and row["subjects"] == [3] for row in missing)
    assert len(missing) == 9
    four, missing_four = candidate_vectors(
        pd.DataFrame(rows), small_config(),
        ("paper_ts_svm", "beat_logistic", "beat_rbf_svm", "beat_mlp"), ("main",))
    assert len(four) == 4 and len(missing_four) == 8


def test_missing_enrolment_and_exhausted_runtime_are_recorded():
    cfg = small_config()
    cfg["enrolment_n"] = [5]
    t = synthetic()
    e = synthetic("E").subset(np.isin(np.arange(300) % 25, [0, 1]))
    frames, failures = _enrolment(t, e, cfg, "G")
    assert not frames
    assert len(failures) == 2
    assert all(f["record"] == "failure" for f in failures)
    cfg["_deadline"] = 0.
    prediction, failure = attack(np.ones((300, 2)), t.meta, np.ones((300, 2)), t.meta,
                                 cfg, 0, dict(enrolled=3, key="P0"))
    assert prediction.empty and failure[0]["error"] == "TimeoutError"


def test_dev_audit_guard_blocks_sess2_and_final_requires_seal(tmp_path):
    write_reduced(tmp_path)
    with session1_only() as access:
        trials = load_trials(tmp_path, [1], 1)
        with pytest.raises(LeakageError):
            np.load(tmp_path / "sess2_subj01.npz")
    assert set(trials.session) == {"T"}
    assert len(access["blocked"]) == 1
    assert all("sess1_" in path for path in access["opened"])
    with pytest.raises(LeakageError):
        verify_seal(None, small_config())


def test_synthetic_final_all_outputs_and_pickling(tmp_path):
    cfg = small_config()
    cfg["seeds"] = [0, 1]
    T, E = synthetic(), synthetic("E")
    out, art = tmp_path / "results", tmp_path / "artifacts"
    with threadpool_limits(limits=1):
        with session1_only() as access:
            selection = run_dev(T, cfg, out, art, timing=False)
        assert not access["blocked"]
        assert selection["folds"] == [1, 2, 3, 4]
        assert len(selection["budgets"]) == 4
        assert set(selection["temperatures"]) >= {"P3_lambda1", "P3_lambda3", "P2-RC_d1"}
        decision = run_final(T, E, cfg, selection, out, art)
        assert set(decision["claims"]) == {"C1", "C2", "C3", "C4"}
        expected = [
            "identity_summary.csv", "participant_recall.csv", "identity_predictions.csv.gz",
            "task.csv", "head_predictions.csv.gz", "control.csv", "processing_timing.csv",
            "enrolment.csv", "enrolled_participants.csv", "observation_length.csv",
            "dimension_sweep.csv", "lambda_sweep.csv", "head_vs_independent.csv",
            "smallest_k_twice_chance.csv", "decision.json", "reproduction.json",
            "attacker_diagnostics.json", "tradeoff.png", "observation_length.png",
            "head_vs_independent.png", "enrolment.png"]
        assert all((out / name).stat().st_size > 0 for name in expected)
        enrol = pd.read_csv(out / "enrolment.csv")
        assert set(enrol.key) == {"G", "G-raw", "P0-RC"}
        assert set(enrol.enrolment.astype(str)) == {"1", "2", "5", "10", "20", "all"}
        assert set(enrol[(enrol.key == "G") & (enrol.enrolment.astype(str) == "1")].attacker) == {
            "beat_logistic", "beat_rbf_svm", "beat_mlp", "paper_ts_svm"}
        assert set(enrol[(enrol.key == "G") & (enrol.enrolment.astype(str) == "5")].attacker) == {
            "paper_ts_svm", "beat_logistic", "beat_rbf_svm", "beat_mlp"}
        predictions = pd.read_csv(out / "identity_predictions.csv.gz")
        assert set(predictions[(predictions.kind == "representation") & (predictions.key == "P3_lambda1")].seed) == {0, 1}
        assert set(predictions[(predictions.kind == "representation") & (predictions.key == "P3_lambda3")].seed) == {0}
        descriptive = pd.read_csv(out/"descriptive_summary.csv", dtype={"seeds": str})
        assert set(descriptive.seeds) == {"0"}
        scaled = predictions[(predictions.kind == "enrolled") & (predictions.enrolled == 2)]
        expected_ids = set(np.random.default_rng(42).permutation([1, 2, 3])[:2])
        assert set(scaled.subject) == expected_ids and set(scaled.scenario) == {"main"}
        execution = json.loads((out/"execution.json").read_text())
        assert execution["scheduling"]["workers"] <= 2
        assert len(execution["jobs"]) == attack_job_count(cfg)["total"]
        assert all(row["status"] == "ok" for row in execution["jobs"])
        primary = predictions[(predictions.kind == "observation") &
                              (predictions.key == "P2-RC_d8") &
                              (predictions["mode"] == "command") & (predictions.tau == .5)]
        units = ["attacker", "scenario", "seed", "direction", "subject"]
        one = primary[primary.k == 1].groupby(units).size()
        twenty = primary[primary.k == 20].groupby(units).size()
        np.testing.assert_array_equal(one, 20*twenty)
        counts = pd.read_csv(out / "usable_group_counts.csv")
        assert counts.usable_groups.ge(0).all()
        excluded = pd.read_csv(out / "exclusions.csv")
        assert excluded.excluded.sum() == 0 and excluded.total.sum() == 600
        control = pd.read_csv(out/"control.csv")
        assert set(control.key) == {"P0", "P2-RC_d8", "P3_lambda1"} and set(control.seed) == {0}
        assert pd.read_csv(out / "processing_timing.csv").trajectories_equal.all()
        reprod = json.loads((out / "reproduction.json").read_text())
        if abs(reprod["accuracy"] - .024) > .05:
            assert decision["claims"]["C4"]["decision"] == "not_evaluable_benchmark_gate"
        model = joblib.load(art / "models/P2_d8_0.joblib")
        assert type(model).__module__ == "beat.models"
        C = covariances(T)
        before = pickle.dumps(model)
        outputs = Parallel(n_jobs=2)(delayed(model)(C.subset(C.subject == s)) for s in (1, 2, 3))
        np.testing.assert_allclose(np.concatenate([p[0] for p in outputs]), model(C)[0], atol=1e-6)
        assert pickle.dumps(model) == before
        cfg["attacker_status"] = "approved"
        # Approval is non-numerical; no tuning changes are allowed after development.
        changed = copy.deepcopy(cfg)
        changed["training"]["max_epochs"] += 1
        with pytest.raises(ValueError, match="Configuration changed"):
            run_final(T, E, changed, selection, out, art)


def test_module_entrypoint_refuses_final_before_data_access():
    result = subprocess.run([sys.executable, "-m", "beat.exp_l", "final"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "Seal 3 commit" in result.stderr
