"""Experiment G contracts; no real Lee2019 files are opened."""
import pickle

import numpy as np
import pandas as pd
import pytest
from joblib import Parallel, delayed
from pyriemann.geometry.mean import mean_riemann
from threadpoolctl import threadpool_limits

from beat.data import LeakageError, Trials
from beat.exp_g import (FourStageDefense, OVRCSP, aggregate_features, apply_rma,
                        chronological_halves, clipped_inverse_tangent, load_lee2019,
                        rma_maps, sealed_lee_plan, spatial_cov, subject_scatter,
                        evaluate_split, mechanism, paired_recall_test, write_outputs,
                        first_labelled_trials, write_paper_comparison)


@pytest.fixture
def signals():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(24, 4, 100))
    X[12:, 0] *= 3
    meta = pd.DataFrame(dict(subject=np.repeat([1, 2], 12), session="T",
                             run=np.tile(np.repeat([1, 2], 6), 2),
                             trial=np.tile(np.arange(1, 7), 4), label=np.tile([0, 1], 12),
                             artifact=False, finite=True))
    return Trials(X, meta)


def test_fit_rejects_evaluation_and_mixed_sessions(signals):
    for sessions in (["E"] * 24, ["T"] * 23 + ["E"]):
        other = Trials(signals.X, signals.meta.assign(session=sessions))
        with pytest.raises(LeakageError):
            FourStageDefense().fit(other)


def test_zero_stages_are_signal_identity(signals):
    defense = FourStageDefense((0, 0, 0, 0)).fit(signals)
    np.testing.assert_array_equal(defense.transform(signals).X, signals.X)


def test_full_rma_matches_population_covariance(signals):
    C = spatial_cov(signals.X)
    maps = rma_maps(C, signals.subject, 1.)
    target = mean_riemann(C)
    for s, W in maps.items():
        np.testing.assert_allclose(W @ mean_riemann(C[signals.subject == s]) @ W.T,
                                   target, atol=1e-7)
    X = apply_rma(signals.X, signals.subject, maps)
    # Lower-triangular positive-diagonal W makes equation 2 exactly W X.
    for s, W in maps.items():
        np.testing.assert_allclose(X[signals.subject == s], W @ signals.X[signals.subject == s],
                                   atol=1e-10)


def test_scatter_shrinkage_hand_calculated():
    # Cov([[-1,0],[1,0]]) = diag(2,0); n=d invokes Appendix C shrinkage.
    result = subject_scatter(np.array([[-1., 0.], [1., 0.]]))
    np.testing.assert_allclose(result, np.diag([1.9001, .1001]), atol=1e-12)


def test_inverse_tangent_clips_matrix_eigenvalues():
    # vech order [00,01,11], sqrt(2) on the off diagonal.
    C = clipped_inverse_tangent(np.array([[50., 0., -50.]]), np.eye(2))
    np.testing.assert_allclose(C[0], np.diag([np.exp(40), np.exp(-40)]))


def test_all_stages_fixed_map_and_module_serialization(signals):
    with threadpool_limits(limits=1):
        defense = FourStageDefense((.9, .35, .6, .2)).fit(signals)
        before = pickle.dumps(defense)
        evaluation = Trials(signals.X * 1.1, signals.meta.assign(session="E"))
        output = defense.transform(evaluation)
        assert np.isfinite(output.X).all()
        assert not np.allclose(output.X, evaluation.X)
        assert pickle.dumps(defense) == before
        for s in (1, 2):
            ix = evaluation.subject == s
            np.testing.assert_allclose(defense.transform(evaluation.subset(ix)).X, output.X[ix])
        restored = pickle.loads(before)
        assert type(restored).__module__ == "beat.exp_g"
        # Real process transfer exercises worker-used class pickling.
        pieces = Parallel(n_jobs=2)(delayed(restored.transform)(evaluation.subset(
            evaluation.subject == s)) for s in (1, 2))
        np.testing.assert_allclose(np.concatenate([p.X for p in pieces]), output.X)


def test_unseen_subject_is_rejected(signals):
    defense = FourStageDefense((0, 0, 0, 0)).fit(signals)
    with pytest.raises(ValueError, match="Subject absent"):
        defense.transform(Trials(signals.X, signals.meta.assign(subject=99)))


def test_recording_halves_and_consecutive_groups(signals):
    evaluation = Trials(signals.X, signals.meta.assign(session="E"))
    a, b = chronological_halves(evaluation)
    assert set(a.run) == {1}
    assert set(b.run) == {2}
    features = np.arange(len(b))[:, None].astype(float)
    grouped, meta = aggregate_features(features, b.meta, 5)
    np.testing.assert_array_equal(grouped[:, 0], [2., 8.])
    assert list(meta.subject) == [1, 2]
    assert list(meta.last_trial) == [5, 5]
    missing = b.subset(b.meta.trial != 3)
    with pytest.raises(ValueError, match="No consecutive groups"):
        aggregate_features(np.zeros((len(missing), 1)), missing.meta, 5)


def test_csp_discriminates_variance_signatures():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(40, 4, 100))
    y = np.repeat([1, 2], 20)
    X[y == 1, 0] *= 4
    X[y == 2, 3] *= 4
    csp = OVRCSP(pairs=1).fit(X, y)
    F = csp.transform(X)
    assert F.shape == (40, 4)
    assert np.linalg.norm(F[y == 1].mean(0) - F[y == 2].mean(0)) > 2


def test_enrolment_selects_first_trials_only_in_supplied_half(signals):
    _, half = chronological_halves(signals)
    # Deliberately shuffled storage must not change recording-order enrolment.
    half = half.subset(np.arange(len(half))[::-1])
    selected = first_labelled_trials(half, 2)
    assert list(selected.subject) == [1, 1, 2, 2]
    assert list(selected.meta.trial) == [1, 2, 1, 2]
    assert set(selected.run) == {2}
    assert len(first_labelled_trials(half, "all")) == len(half)
    with pytest.raises(ValueError, match="Insufficient enrolment"):
        first_labelled_trials(half, 7)


def test_lee_session2_guard_precedes_io(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Session-2 guard allowed file access")
    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(LeakageError):
        load_lee2019(tmp_path, [1], 2)
    with pytest.raises(LeakageError):
        sealed_lee_plan(None)


def test_lee_synthetic_loader_and_four_stage_analysis(tmp_path):
    rng = np.random.default_rng(5)
    # Synthetic sessions only, in pytest's private directory.
    for session in (1, 2):
        for subject in (1, 2):
            np.savez(tmp_path / f"sess{session}_subj{subject:02d}.npz",
                     X=rng.normal(size=(40, 20, 750)).astype("float32") * subject,
                     label=np.tile([0, 1], 20), finite=np.ones(40, dtype=bool),
                     phase=np.repeat(["train", "test"], 20), trial=np.tile(np.arange(1, 21), 2))
    train, rejected = load_lee2019(tmp_path, [1, 2], 1)
    test, _ = load_lee2019(tmp_path, [1, 2], 2, allow_session2=True)
    assert rejected == 0 and set(train.session) == {"T"} and set(test.session) == {"E"}
    with threadpool_limits(limits=1):
        model = FourStageDefense((.9, .60, .6, .2)).fit(train)
        released = model.transform(test)
    first, second = chronological_halves(released)
    assert len(first) == len(second) == 40
    assert set(first.run) == {1} and set(second.run) == {2}
    assert np.isfinite(released.X).all()
    cfg = dict(csp_pairs=2, xdawn_filters=1, seed=42, k=[1, 10], bootstrap_samples=5,
               enrolment_n=[1, 2, 5, 10, 20, "all"], enrolment_k=[1, 10])
    attack = dict(inner_folds=3, logistic_C=1., logistic_max_iter=1000,
                  svm_C=[1.], svm_gamma=["scale"], mlp_hidden=[4], mlp_alpha=.01,
                  mlp_solver="lbfgs", mlp_max_iter=1000, mlp_max_fun=10000)
    outputs = []
    mechanisms = {}
    for condition, (a, b) in {"undefended": (train, test),
                              "defended": (model.transform(train), released)}.items():
        mechanisms[condition] = mechanism(a, b)
        h1, h2 = chronological_halves(b)
        for scenario, direction, tr, te in [
                ("reference_session", "S1_to_S2", a, b),
                ("reference_session", "S2_to_S1", b, a),
                ("post_deployment", "first_to_second", h1, h2),
                ("post_deployment", "second_to_first", h2, h1)]:
            outputs.append(evaluate_split(tr, te, cfg, attack, condition, scenario, direction))
    rows = pd.DataFrame([r for output in outputs for r in output[0]])
    assert len(rows) == 2 * 4 * 8 * 2
    assert rows.balanced_accuracy.between(0, 1).all()
    post = rows[(rows.scenario == "post_deployment") & (rows.k == 1)]
    assert post.ba_ci_low.isna().all() and post.ba_ci_high.isna().all()
    ref = rows[(rows.scenario == "reference_session") & (rows.k == 1)]
    assert ref.ba_ci_low.notna().all() and ref.ba_ci_high.notna().all()
    enrolment_outputs = [o[3] for o in outputs if o[3] is not None]
    enrolment = pd.DataFrame([r for o in enrolment_outputs for r in o[0]])
    assert len(enrolment) == 2 * 2 * 6 * 4 * 2
    unavailable = enrolment[(enrolment.enrolment_n.isin(["1", "2"]))
                            & enrolment.attacker.isin(["beat_rbf_svm", "beat_mlp"])]
    assert len(unavailable) == 32
    assert (unavailable.status == "not_evaluable").all()
    assert unavailable.balanced_accuracy.isna().all()
    eligible = enrolment.drop(unavailable.index)
    assert (eligible.status == "evaluable").all()
    assert eligible.balanced_accuracy.between(0, 1).all()
    for output in outputs:
        if output[3] is None:
            continue
        all_rows = [r for r in output[3][0] if r["enrolment_n"] == "all"]
        original = {(r["attacker"], r["k"]): r for r in output[0]}
        for r in all_rows:
            assert r["balanced_accuracy"] == original[r["attacker"], r["k"]]["balanced_accuracy"]
        for n, member in output[3][3].groupby("enrolment_n"):
            count = 20 if n == "all" else n
            assert member.groupby("subject").size().tolist() == [count, count]
            assert sorted(member.trial.unique()) == list(range(1, count + 1))
    write_outputs(tmp_path / "output", rows,
                  pd.concat([o[1] for o in outputs], ignore_index=True), mechanisms,
                  dict(dataset="lee2019", chance=.5, seed=42, configuration=cfg),
                  enrolment_outputs)
    assert (tmp_path / "output" / "summary.json").is_file()
    assert (tmp_path / "output" / "identity_vs_k.png").is_file()
    assert (tmp_path / "output" / "enrolment_curve.png").is_file()
    import json
    summary = json.loads((tmp_path / "output" / "summary.json").read_text())
    assert len(summary["enrolment_curve"]) == 2 * 6 * 4 * 2
    assert sum(r["balanced_accuracy"] is None for r in summary["enrolment_curve"]) == 16


def test_lee_full_cohort_statistics_are_bounded():
    # Positive subject differences: extreme tail; never enumerate 2^54 signs.
    result = paired_recall_test(np.ones(54), "lee2019")
    assert 0 < result["signflip_p"] <= .001


def test_paper_comparison_withholds_unmatched_configuration_differences(tmp_path):
    results = pd.DataFrame([
        dict(condition="undefended", direction="S1_to_S2", attacker="paper_ts_svm",
             k=1, accuracy=.8),
        dict(condition="undefended", direction="S1_to_S2", attacker="paper_xdawn_svm",
             k=1, accuracy=.8),
        dict(condition="defended", direction="S1_to_S2", attacker="paper_xdawn_svm",
             k=1, accuracy=.2),
        dict(condition="defended", direction="S1_to_S2", attacker="paper_csp_svm",
             k=1, accuracy=.2),
        dict(condition="defended", direction="S1_to_S2", attacker="paper_csp_lda",
             k=1, accuracy=.2),
    ])
    rows = write_paper_comparison(tmp_path, results, dict(
        dataset="bnci", configuration=dict(xdawn_filters=1, csp_pairs=4)))
    saved = pd.read_csv(tmp_path / "paper_table1_comparison.csv")
    matched = saved[saved.comparable]
    assert len(matched) == 1
    assert matched.iloc[0].attacker == "paper_ts_svm"
    assert matched.iloc[0].difference_pp == pytest.approx(-10.8)
    assert saved.loc[~saved.comparable, "difference_pp"].isna().all()
    assert saved.comparison_note.notna().all()
    for row in rows:
        assert row["paper_attacker_configuration"] and row["observed_attacker_configuration"]
        assert row["paper_selection_policy"] and row["observed_selection_policy"]
        if not row["comparable"]:
            assert row["difference_pp"] is None
    xd = saved[(saved.condition == "undefended") & (saved.attacker == "paper_xdawn_svm")].iloc[0]
    assert "oracle (8)" in xd.paper_attacker_configuration
    assert "fixed nfilter=1" in xd.observed_attacker_configuration
    # Matching numerical counts do not make oracle and fixed selection comparable.
    defended = saved[(saved.condition == "defended") & (saved.direction == "S1_to_S2")
                     & (saved.attacker == "paper_xdawn_svm")].iloc[0]
    assert not defended.comparable and pd.isna(defended.difference_pp)
