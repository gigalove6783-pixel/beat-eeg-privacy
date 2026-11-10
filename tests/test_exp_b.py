"""Experiment B contracts. Final evaluation uses only generated covariances."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from beat.attacks import ATTACKERS, fit_attackers
from beat.config import load_config
from beat.data import META_COLUMNS, LeakageError, Trials
from beat.exp_b import (FAMILIES, _identity_interval, _identity_scores,
                        choose_epochs, choose_operating_points,
                        evaluate_attack_scenarios, load_frozen, load_oof,
                        load_outputs, make_decision, run_dev, run_final, settings)
from beat.models import (FeatureMap, fit_defended, fit_encoder, fit_linear,
                         ramp, reverse_gradient)


def synthetic(session="T", subjects=9, per_run=4):
    rng = np.random.default_rng(10 if session == "T" else 11)
    rows, cov = [], []
    for subject in range(1, subjects + 1):
        for run in range(1, 7):
            for trial in range(per_run):
                label = trial % 2
                matrix = rng.normal(0, 0.03, (3, 3))
                diagonal = np.exp([0.6 * (2 * label - 1), subject / 10, -subject / 12])
                cov.append(matrix @ matrix.T + np.diag(diagonal))
                rows.append(dict(subject=subject, session=session, run=run, trial=trial + 1,
                                 label=label, artifact=False, finite=True))
    return Trials(np.stack(cov), pd.DataFrame(rows))


@pytest.fixture
def small_config():
    cfg = copy.deepcopy(load_config("exp_b"))
    cfg.update(seeds=[0], lambdas=[0.01, 0.3], workers=1)
    cfg["training"].update(max_epochs=3, checkpoint_interval=1, batch_size=128)
    cfg["attackers"].update(svm_C=[1.0], svm_gamma=["scale"], inner_folds=2)
    cfg["statistics"].update(bootstrap_samples=30, utility_margin=1.0)
    return cfg


@pytest.fixture(autouse=True)
def restore_torch_settings():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    with torch.random.fork_rng():
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


@pytest.mark.parametrize("entry", ["features", "linear", "encoder", "pipeline", "dev"])
def test_defended_fit_entries_reject_e_before_fit(entry, small_config, tmp_path):
    e = synthetic("E")
    cfg = small_config["training"]
    with pytest.raises(LeakageError):
        if entry == "features":
            FeatureMap(False).fit(e)
        elif entry == "linear":
            fit_linear(e, FeatureMap(False), cfg, 0)
        elif entry == "encoder":
            fit_encoder(e.with_X(np.ones((len(e), 6))), cfg, 0, 1, 0.3)
        elif entry == "pipeline":
            fit_defended(e, "P3", cfg, 0, 1, 0.3)
        else:
            run_dev(e, small_config, tmp_path / "results", tmp_path / "artifacts")


def test_reversal_changes_only_encoder_gradient_and_ramps():
    x = torch.tensor([1.0, -2.0], requires_grad=True)
    head_weight = torch.tensor(3.0, requires_grad=True)
    z = reverse_gradient(x, 0.4)
    assert torch.equal(z, x)
    (z * head_weight).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([-1.2, -1.2]))
    assert head_weight.grad.item() == -1
    assert ramp(3, 0) == 0
    assert ramp(3, 1) == pytest.approx(2.9997276128)


def test_same_seed_budget_and_zero_lambda_are_identical(small_config):
    train = synthetic(subjects=3)
    cfg = small_config["training"]
    p2 = fit_defended(train, "P2", cfg, 4, 2)
    p3zero = fit_defended(train, "P3", cfg, 4, 2, 0.0)
    np.testing.assert_array_equal(p2(train)[0], p3zero(train)[0])
    np.testing.assert_array_equal(p2(train)[1], p3zero(train)[1])
    assert not p2.predictor.training
    assert not any(parameter.requires_grad for parameter in p2.predictor.parameters())
    _, X = p2.features.transform(train)
    long = fit_encoder(train.with_X(X), cfg, 4, 2, 0, checkpoints=[1])
    short = fit_encoder(train.with_X(X), cfg, 4, 1, 0)
    for name, tensor in short[1].items():
        torch.testing.assert_close(long[1][name], tensor, rtol=0, atol=0)
    defended = fit_defended(train, "P3", cfg, 4, 2, 3.0)
    assert not np.array_equal(p2(train)[0], defended(train)[0])


def test_feature_reference_and_scale_do_not_update_during_e_inference(small_config):
    train, test = synthetic(subjects=3), synthetic("E", subjects=3)
    model = fit_defended(train, "P0-RC", small_config["training"], 0, 1)
    mean = model.features.scale.mean_.copy()
    whiteners = copy.deepcopy(model.features.reference.whiteners_)
    before = model(test)[1]
    altered = test.with_X(test.X * 20)
    model(altered)
    np.testing.assert_array_equal(model.features.scale.mean_, mean)
    for subject, w in whiteners.items():
        np.testing.assert_array_equal(model.features.reference.whiteners_[subject], w)
    np.testing.assert_array_equal(model(test)[1], before)


def test_attackers_learn_identity_and_tune_only_on_supplied_training(small_config):
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(1, 10), 12)
    X = np.eye(9)[labels - 1] * 4 + rng.normal(0, 0.02, (len(labels), 9))
    models = fit_attackers(X, labels, small_config["attackers"], 5)
    assert tuple(models) == ATTACKERS
    heldout = np.eye(9) * 4
    for model in models.values():
        np.testing.assert_array_equal(model.predict(heldout), np.arange(1, 10))
    svm = models["rbf_svm"]
    assert svm.best_params_["classifier__C"] == 1
    assert svm.best_params_["classifier__gamma"] == "scale"
    np.testing.assert_allclose(svm.best_estimator_["scale"].mean_, X.mean(axis=0))
    before = models["mlp"]["scale"].mean_.copy()
    models["mlp"].predict(heldout * 100)
    np.testing.assert_array_equal(before, models["mlp"]["scale"].mean_)
    with pytest.raises(ValueError, match="inner_folds"):
        fit_attackers(X[:2], labels[:2], small_config["attackers"])
    assert models["mlp"].convergence_["converged"]
    assert models["mlp"].convergence_["warning"] == ""
    limited = dict(small_config["attackers"], mlp_max_iter=1)
    capped = fit_attackers(X, labels, limited, 5)["mlp"]
    assert not capped.convergence_["converged"]
    assert capped.convergence_["iterations"] == 1
    assert capped.convergence_["warning"]


def test_separate_adversary_api_preserves_identity_labels_and_task_outputs(small_config):
    train = synthetic(subjects=3)
    train.meta["subject"] *= 10
    model = fit_defended(train, "P3", small_config["training"], 0, 1, 0.1)
    before = model(train)
    assert len(before) == 2
    with torch.no_grad():
        for layer in model.predictor.adversary:
            if isinstance(layer, nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        model.predictor.adversary[-1].bias[1] = 4
    scores = model.adversary_logits(train)
    np.testing.assert_array_equal(model.identity_classes_[scores.argmax(axis=1)],
                                  np.full(len(train), 20))
    for original, after in zip(before, model(train)):
        np.testing.assert_array_equal(original, after)


def test_epoch_choice_averages_folds_and_seeds_and_breaks_ties_early():
    rows = []
    for family in FAMILIES:
        for seed in (0, 1):
            for fold in range(1, 7):
                for epoch, score in ((5, 0.7), (10, 0.8), (15, 0.8)):
                    rows.append(dict(family=family, seed=seed, fold=fold,
                                     epoch=epoch, ba=score + (seed - 0.5) * 0.02))
    assert choose_epochs(pd.DataFrame(rows)) == {"common": 10, "rc": 10}


def test_operating_point_uses_strongest_attacker_utility_filter_and_ties(small_config):
    cfg = copy.deepcopy(small_config)
    cfg["statistics"]["utility_margin"] = 0.05
    tasks, identities = [], []
    for family in FAMILIES:
        suffix = "-RC" if family == "rc" else ""
        for condition, target, utility, exposure in (
                ("P2", 0, 0.8, 0.8), ("P3", 0.01, 0.74, 0.1),
                ("P3", 0.03, 0.76, 0.3), ("P3", 0.1, 0.78, 0.3),
                ("P3", 0.3, 0.79, 0.4)):
            tags = dict(condition=condition + suffix, target=target, family=family)
            tasks.append(dict(**tags, ba=utility))
            for attacker, value in zip(ATTACKERS, [exposure, exposure - 0.02, exposure - 0.05]):
                identities.append(dict(**tags, attacker=attacker, ba=value))
            if condition == "P3":
                identities.append(dict(**tags, attacker="adversary_head", ba=1.0))
    task, identity = pd.DataFrame(tasks), pd.DataFrame(identities)
    points, _ = choose_operating_points(task, identity, cfg)
    assert points["common"]["target"] == 0.1
    assert 0.01 not in points["rc"]["eligible_targets"]
    task.loc[(task.family == "rc") & task.condition.str.startswith("P3"), "ba"] = 0.7
    points, _ = choose_operating_points(task, identity, cfg)
    assert points["rc"]["target"] is None
    assert points["rc"]["status"] == "no_operating_point"


def test_main_direction_means_are_not_pooled_by_trial_count():
    rows = []
    for subject in (1, 2):
        for direction, count, prediction in ((0, 3, subject), (1, 1, 3 - subject)):
            for _ in range(count):
                rows.append(dict(subject=subject, prediction=prediction,
                                 direction=direction, run=direction + 1,
                                 attacker="logistic", scenario="main"))
    frame = pd.DataFrame(rows)
    scores, recall = _identity_scores(frame)
    assert scores.ba.mean() == 0.5
    assert (recall.groupby("subject").recall.mean() == 0.5).all()
    interval = _identity_interval(frame, 20, 0, 0.95)
    assert interval["ba_low"] == interval["ba_high"] == 0.5
    assert interval["macro_f1_low"] == interval["macro_f1_high"] == 0.5


def test_scenarios_use_disjoint_e_halves_and_auxiliary_t(small_config):
    train, test = synthetic(), synthetic("E")
    t_z, e_z = np.eye(9)[train.subject - 1], np.eye(9)[test.subject - 1]
    head_prediction = np.ones(len(test), dtype=int)
    scores, directional, recalls, predictions, tuning = evaluate_attack_scenarios(
        t_z, train, e_z, test, small_config, 0, head_prediction)
    assert set(scores.scenario) == {"main", "auxiliary"}
    assert len(scores) == 8 and len(directional) == 12 and len(recalls) == 72
    assert (scores.loc[scores.attacker.isin(ATTACKERS), "ba"] == 1).all()
    np.testing.assert_allclose(scores.loc[scores.attacker == "adversary_head", "ba"], 1 / 9)
    assert set(scores.loc[scores.attacker == "adversary_head", "evidence_role"]) == {
        "training_adversary_diagnostic"}
    assert set(predictions.session) == {"E"}
    main = predictions[predictions.scenario == "main"]
    assert set(main.loc[main.direction == 0, "run"]) == {4, 5, 6}
    assert set(main.loc[main.direction == 1, "run"]) == {1, 2, 3}
    assert len(tuning) == 6
    for scenario, direction in (("main", 0), ("main", 1), ("auxiliary", 0)):
        subset = predictions[(predictions.scenario == scenario) & (predictions.direction == direction)]
        head_meta = subset.loc[subset.attacker == "adversary_head", META_COLUMNS].reset_index(drop=True)
        independent_meta = subset.loc[subset.attacker == "logistic", META_COLUMNS].reset_index(drop=True)
        pd.testing.assert_frame_equal(head_meta, independent_meta)


def decision_tables():
    tasks, recalls = [], []
    for condition in ("P0", "P0-RC", "P2", "P2-RC", "P3", "P3-RC"):
        target = 0.1 if condition.startswith("P3") else 0.0
        for seed in (0, 1):
            for subject in range(1, 10):
                defended = condition.startswith("P3")
                tasks.append(dict(condition=condition, target=target, subject=subject,
                                  seed=seed, ba=0.78 if defended else 0.8))
                for scenario in ("main", "auxiliary"):
                    for i, attacker in enumerate(ATTACKERS):
                        recalls.append(dict(condition=condition, target=target,
                                            subject=subject, seed=seed, scenario=scenario,
                                            attacker=attacker,
                                            recall=(0.5 if defended else 0.7) - i * 0.02))
    selection = {"families": {f: {"target": 0.1} for f in FAMILIES}}
    return pd.DataFrame(tasks), pd.DataFrame(recalls), selection


def test_decision_requires_both_families_and_seed_mean_subject_pairs():
    tasks, recalls, selection = decision_tables()
    cfg = load_config("exp_b")
    verdict = make_decision(tasks, recalls, selection, cfg)
    assert verdict["h2_accepted"]
    assert verdict["families"]["common"]["utility"]["lower_95"] == pytest.approx(-0.02)
    assert len(verdict["families"]["common"]["utility"]["per_subject"]) == 9
    heads = recalls[recalls.attacker == "logistic"].copy()
    heads["attacker"] = "adversary_head"
    heads["recall"] = 1.0
    assert make_decision(tasks, pd.concat([recalls, heads]), selection, cfg) == verdict
    tasks.loc[tasks.condition == "P3-RC", "ba"] = 0.7
    verdict = make_decision(tasks, recalls, selection, cfg)
    assert not verdict["h2_accepted"]
    assert not verdict["families"]["rc"]["utility"]["passed"]


def test_decision_withholds_chance_baseline_and_auxiliary_only_improvement():
    tasks, recalls, selection = decision_tables()
    cfg = load_config("exp_b")
    recalls.loc[recalls.condition == "P0", "recall"] = 1 / 9
    verdict = make_decision(tasks, recalls, selection, cfg)
    assert verdict["baseline_at_chance"] and not verdict["h2_accepted"]
    recalls.loc[recalls.condition == "P0", "recall"] = 0.8
    main = recalls.scenario == "main"
    recalls.loc[main & recalls.condition.str.startswith("P2"), "recall"] = 0.4
    recalls.loc[main & recalls.condition.str.startswith("P3"), "recall"] = 0.4
    verdict = make_decision(tasks, recalls, selection, cfg)
    assert verdict["families"]["common"]["protection"]["strongest_both_scenarios"]["passed"]
    assert not verdict["families"]["common"]["protection"]["main"]["passed"]
    assert not verdict["h2_accepted"]


def test_synthetic_dev_final_and_c_api_roundtrip(small_config, tmp_path):
    train, test = synthetic(per_run=6), synthetic("E", per_run=6)
    # Real data flags are kept in exports, but excluded from all B fitting/scoring.
    train.meta.loc[0, "artifact"] = True
    test.meta.loc[0, "artifact"] = True
    test.meta.loc[1, "finite"] = False
    test.X[1] = np.nan
    results, artifacts = tmp_path / "results", tmp_path / "artifacts"
    selection = run_dev(train, small_config, results, artifacts)
    assert selection["sessions"] == ["T"]
    assert selection["retained_trials"] == len(train) - 1
    history = pd.read_csv(results / "epoch_cv.csv")
    assert set(history.fold) == set(range(1, 7))
    assert set(history.family) == {"common", "rc"}
    assert set(history.epoch) == {1, 2, 3}
    for family in FAMILIES:
        assert selection["budgets"][family] in (1, 2, 3)
    cv_scores = pd.read_csv(results / "identity_cv.csv")
    assert len(cv_scores[cv_scores.attacker == "adversary_head"]) == 4 * 6 * 2
    assert set(cv_scores.loc[cv_scores.attacker == "adversary_head", "condition"]) == {"P3", "P3-RC"}
    for phase, count in ((selection, 8 * 6 * 2),):
        assert phase["attacker_convergence"]["mlp_fits"] == count
        assert phase["attacker_convergence"]["mlp_nonconverged"] == 0
    for condition, target in selection["selected"].items():
        if target is None:
            continue
        oof = load_oof(condition, 0, artifacts)
        assert len(oof["logits"]) == len(train) - 1
        assert set(oof["session"]) == {"T"}
        assert not oof["artifact"].any()
        assert set(oof) == set(META_COLUMNS) | {"logits"}
        keys = list(zip(oof["subject"], oof["run"], oof["trial"]))
        assert len(set(keys)) == len(keys)
    combined = Trials(np.concatenate([train.X, test.X]),
                      pd.concat([train.meta, test.meta], ignore_index=True))
    verdict = run_final(combined, selection, small_config, results, artifacts)
    assert isinstance(verdict["h2_accepted"], bool)
    for _, condition, target in settings(small_config):
        model = load_frozen(condition, 0, target, artifacts)
        outputs = load_outputs(condition, 0, target, artifacts)
        z, logits = model(combined)
        np.testing.assert_array_equal(z, outputs["representation"])
        np.testing.assert_array_equal(logits, outputs["logits"])
        assert set(META_COLUMNS) | {"representation", "logits"} <= set(outputs)
        assert len(logits) == len(combined)
        assert outputs["artifact"].sum() == 2
        assert np.isnan(logits[len(train) + 1]).all()
        assert np.isfinite(logits[len(train)]).all()
        assert z.shape[1] == (6 if condition.startswith("P0") else 8)
        if model.neural:
            assert not any(p.requires_grad for p in model.predictor.parameters())
        if condition.startswith("P3"):
            np.testing.assert_array_equal(model.adversary_logits(combined),
                                          outputs["adversary_logits"])
            np.testing.assert_array_equal(outputs["adversary_classes"], np.arange(1, 10))
            assert np.isnan(outputs["adversary_logits"][len(train) + 1]).all()
    assert (results / "utility_exposure.png").stat().st_size > 1000
    assert (results / "adversary_vs_independent.png").stat().st_size > 1000
    comparison = pd.read_csv(results / "adversary_comparison.csv")
    assert len(comparison) == 4
    assert set(comparison.strongest_attacker) <= set(ATTACKERS)
    assert json.loads((results / "decision.json").read_text()) == verdict
    scores = pd.read_csv(results / "identity_final.csv")
    assert len(scores) == len(list(settings(small_config))) * 6 + 4 * 2
    assert set(scores.scenario) == {"main", "auxiliary"}
    assert len(pd.read_csv(results / "task_final.csv")) == len(list(settings(small_config))) * 9
    manifest = json.loads((artifacts / "final" / "manifest.json").read_text())
    assert manifest["attacker_convergence"]["mlp_fits"] == 8 * 3
    assert manifest["attacker_convergence"]["mlp_nonconverged"] == 0
    changed = copy.deepcopy(small_config)
    changed["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="config/source"):
        run_final(combined, selection, changed, results, artifacts)
