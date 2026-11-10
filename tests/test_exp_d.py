"""Experiment D contracts. Final evaluation uses only generated covariances."""
import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from beat.attacks import ATTACKERS
from beat.data import LeakageError, Trials
from beat.exp_b import FAMILIES
from beat.exp_d import (EnsembleEncoder, choose_variants, fit_variant,
                        fit_variant_pipeline, frontier, load_d_config,
                        make_decision_d, refresh_index, run_dev, run_final,
                        tradeoff_table, variant_loss)
from beat.models import initialized_encoder
from test_exp_b import synthetic


@pytest.fixture
def small_config():
    cfg = copy.deepcopy(load_d_config())
    cfg.update(seeds=[0], lambdas=[0.1, 1.0], workers=1, bottlenecks=[2, 8],
               epoch_grid=[1, 2])
    cfg["training"].update(max_epochs=3, checkpoint_interval=1)
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


def _training(cfg):
    return dict(cfg["training"], bottleneck=8, ensemble_heads=cfg["variants"]["ensemble"]["heads"])


def test_config_shares_b_sections_and_plan_grids():
    cfg = load_d_config()
    from beat.config import load_config
    base = load_config("exp_b")
    for key in ("seeds", "lambdas", "training", "attackers", "statistics"):
        assert cfg[key] == base[key]
    assert cfg["bottlenecks"] == [1, 2, 4, 8, 16, 32]
    assert cfg["epoch_grid"] == [5, 10, 20, 40]
    assert cfg["variants"]["ensemble"]["heads"] == 5
    assert cfg["workers"] <= 4


@pytest.mark.parametrize("entry", ["network", "pipeline", "dev"])
def test_defended_fit_entries_reject_e(entry, small_config, tmp_path):
    e = synthetic("E")
    with pytest.raises(LeakageError):
        if entry == "network":
            fit_variant(e.with_X(np.ones((len(e), 6))), "maxent", _training(small_config), 0, 1, 0.3)
        elif entry == "pipeline":
            fit_variant_pipeline(e, "rc", "ensemble", _training(small_config), 0, 1, 0.3)
        else:
            run_dev(e, small_config, tmp_path)


def _batch(cfg, seed=3):
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(rng.normal(size=(20, 6)), dtype=torch.float32)
    y = torch.as_tensor(rng.integers(0, 2, 20))
    identity = torch.as_tensor(np.arange(20) % 3)
    return x, y, identity


def _grads(loss, parameters):
    return torch.autograd.grad(loss, list(parameters), allow_unused=True, retain_graph=True)


def test_maxent_adversary_learns_identity_and_encoder_targets_uniform(small_config):
    cfg = _training(small_config)
    model = initialized_encoder(6, 3, cfg, 0)
    x, y, identity = _batch(cfg)
    strength = 0.7
    actual_encoder = _grads(variant_loss(model, "maxent", x, y, identity, strength),
                            model.encoder.parameters())
    actual_adversary = _grads(variant_loss(model, "maxent", x, y, identity, strength),
                              model.adversary.parameters())
    z = model.encoder(x)
    uniform_ce = -F.log_softmax(model.adversary(z), dim=1).mean()
    expected_encoder = _grads(F.cross_entropy(model.task(z), y) + strength * uniform_ce,
                              model.encoder.parameters())
    expected_adversary = _grads(F.cross_entropy(model.adversary(z.detach()), identity),
                                model.adversary.parameters())
    for a, b in zip(actual_encoder + actual_adversary, expected_encoder + expected_adversary):
        torch.testing.assert_close(a, b)


def test_ensemble_heads_train_on_own_ce_and_encoder_reverses_mean(small_config):
    cfg = _training(small_config)
    torch.manual_seed(0)
    model = EnsembleEncoder(6, 3, cfg)
    x, y, identity = _batch(cfg)
    strength = 0.7
    loss = variant_loss(model, "ensemble", x, y, identity, strength)
    actual_encoder = _grads(loss, model.encoder.parameters())
    z = model.encoder(x)
    heads = model.adversary.heads
    mean_ce = sum(F.cross_entropy(h(z), identity) for h in heads) / len(heads)
    task_grad = _grads(F.cross_entropy(model.task(z), y), model.encoder.parameters())
    adversary_grad = _grads(mean_ce, model.encoder.parameters())
    for a, t, g in zip(actual_encoder, task_grad, adversary_grad):
        torch.testing.assert_close(a, t - strength * g)
    for head in heads:
        actual = _grads(loss, head.parameters())
        expected = _grads(F.cross_entropy(head(z), identity), head.parameters())
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)
    posterior = model.adversary(z).exp()
    mean = torch.stack([h(z).softmax(1) for h in heads]).mean(0)
    torch.testing.assert_close(posterior, mean)


def test_ensemble_refresh_rotates_one_head_and_init_matches_b(small_config):
    assert [refresh_index(e, 5) for e in range(1, 8)] == [None, 0, 1, 2, 3, 4, 0]
    cfg = dict(_training(small_config), learning_rate=0.0)
    train = synthetic(subjects=3).with_X(np.random.default_rng(0).normal(size=(72, 6)))
    torch.manual_seed(4)
    initial = EnsembleEncoder(6, 3, cfg).state_dict()
    b_model = initialized_encoder(6, 3, cfg, 4).state_dict()
    for name, tensor in b_model.items():
        ours = name.replace("adversary.", "adversary.heads.0.")
        torch.testing.assert_close(initial[ours], tensor, rtol=0, atol=0)
    trained = fit_variant(train, "ensemble", cfg, 4, 3, 1.0).state_dict()
    changed = {k.split(".")[2] for k in trained if not torch.equal(trained[k], initial[k])}
    assert changed == {"0", "1"}
    assert all(k.startswith("adversary.heads.") for k in trained
               if not torch.equal(trained[k], initial[k]))


def test_variant_fits_are_deterministic_and_lambda_matters(small_config):
    cfg = _training(small_config)
    train = synthetic(subjects=3)
    for variant in ("maxent", "ensemble"):
        a = fit_variant_pipeline(train, "common", variant, cfg, 2, 2, 1.0)
        b = fit_variant_pipeline(train, "common", variant, cfg, 2, 2, 1.0)
        c = fit_variant_pipeline(train, "common", variant, cfg, 2, 2, 3.0)
        np.testing.assert_array_equal(a(train)[0], b(train)[0])
        assert not np.array_equal(a(train)[0], c(train)[0])
        assert a(train)[0].shape[1] == 8
        assert a.adversary_logits(train).shape == (len(train), 3)
        assert not any(p.requires_grad for p in a.predictor.parameters())


def _summary_row(kind, family, target, epochs, task, identity, d=8):
    return dict(kind=kind, condition=kind, family=family, d=d, target=target, epochs=epochs,
                task_ba=task, identity_ba=identity, adversary_ba=0.1)


def test_joint_selection_uses_d8_reference_ties_and_per_lambda_curve(small_config):
    cfg = copy.deepcopy(small_config)
    cfg["statistics"]["utility_margin"] = 0.05
    rows = []
    for family in FAMILIES:
        rows += [_summary_row("task", family, 0.0, 7, 0.80, 0.6),
                 _summary_row("task", family, 0.0, 9, 0.95, 0.9, d=32)]
        for variant in ("maxent", "ensemble"):
            rows += [_summary_row(variant, family, 0.1, 1, 0.76, 0.4),
                     _summary_row(variant, family, 0.1, 2, 0.78, 0.4),
                     _summary_row(variant, family, 1.0, 1, 0.70, 0.2),
                     _summary_row(variant, family, 1.0, 2, 0.74, 0.3)]
    summary = pd.DataFrame(rows)
    choice = choose_variants(summary, cfg)
    common = choice["maxent"]["common"]
    assert (common["target"], common["epochs"], common["reference_task_ba"]) == (0.1, 2, 0.8)
    assert common["curve"] == [dict(target=0.1, epochs=2, eligible=True),
                               dict(target=1.0, epochs=2, eligible=False)]
    summary.loc[summary.kind != "task", "task_ba"] = 0.5
    choice = choose_variants(summary, cfg)
    assert choice["ensemble"]["rc"]["status"] == "no_operating_point"
    assert choice["ensemble"]["rc"]["target"] is None


def _decision_tables():
    tasks, recalls, settings = [], [], []
    spec = [("linear", "P0", 0, 0.0, 0.95), ("task", "D1-d8", 8, 0.0, 0.7),
            ("task", "D1-d2", 2, 0.0, 0.3), ("maxent", "MaxEnt", 8, 0.1, 0.5),
            ("ensemble", "Ensemble", 8, 0.1, 0.7)]
    for family in FAMILIES:
        suffix = "-RC" if family == "rc" else ""
        for kind, condition, d, target, exposure in spec:
            setting = dict(family=family, kind=kind, condition=condition + suffix, d=d,
                           target=target, epochs=0 if kind == "linear" else 5)
            settings.append(setting)
            for seed in (0, 1):
                for subject in range(1, 10):
                    task = 0.6 if condition == "D1-d2" else 0.8
                    tasks.append(dict(setting, seed=seed, subject=subject, ba=task, kappa=0.5))
                    for scenario in ("main", "auxiliary"):
                        for i, attacker in enumerate(ATTACKERS):
                            recalls.append(dict(setting, seed=seed, subject=subject,
                                                scenario=scenario, attacker=attacker,
                                                recall=exposure - i * 0.02))
    selection = {"variants": {v: {f: {"target": 0.1, "epochs": 5} for f in FAMILIES}
                              for v in ("maxent", "ensemble")}}
    grl = pd.DataFrame([dict(condition=c, target=t, selected=s, task_ba=0.8, identity_ba=i)
                        for c, t, s, i in (("P2", 0.0, False, 0.7), ("P3", 0.01, True, 0.71),
                                           ("P2-RC", 0.0, False, 0.7),
                                           ("P3-RC", 0.01, True, 0.71))])
    return pd.DataFrame(tasks), pd.DataFrame(recalls), settings, selection, grl


def test_decision_relabels_b_rule_and_answers_both_questions(small_config):
    tasks, recalls, settings, selection, grl = _decision_tables()
    scores = pd.DataFrame(columns=["condition", "target", "attacker", "scenario", "ba"])
    tradeoff = tradeoff_table(tasks, scores, recalls, settings, selection)
    assert tradeoff.selected.sum() == 4
    decision = make_decision_d(tasks, recalls, tradeoff, grl, selection, small_config)
    for family in FAMILIES:
        maxent = decision["variants"]["maxent"]["families"][family]
        ensemble = decision["variants"]["ensemble"]["families"][family]
        assert maxent["protection_rule_passed"] and maxent["utility"]["passed"]
        assert maxent["protection"]["main"]["mean"] == pytest.approx(0.2)
        assert not ensemble["protection_rule_passed"]
        points = {p["variant"]: p for p in decision["question_ii"]["families"][family]["points"]}
        assert points["maxent"]["dominates"] and points["maxent"]["comparator_d"] == [8]
        assert not points["ensemble"]["dominates"] and not points["grl_ce"]["dominates"]
        assert decision["exp_b_p2_consistency"][family]["task_ba_diff"] == pytest.approx(0)
    assert decision["question_i"]["answer"] and decision["question_ii"]["answer"]
    json.dumps(decision, allow_nan=False)
    recalls.loc[recalls.condition.str.startswith("MaxEnt"), "recall"] = 0.7
    tradeoff = tradeoff_table(tasks, scores, recalls, settings, selection)
    decision = make_decision_d(tasks, recalls, tradeoff, grl, selection, small_config)
    assert not decision["question_i"]["answer"] and not decision["question_ii"]["answer"]


def test_question_i_requires_protection_in_both_families(small_config):
    tasks, recalls, settings, selection, grl = _decision_tables()
    recalls.loc[recalls.condition == "MaxEnt-RC", "recall"] = 0.7
    scores = pd.DataFrame(columns=["condition", "target", "attacker", "scenario", "ba"])
    tradeoff = tradeoff_table(tasks, scores, recalls, settings, selection)
    decision = make_decision_d(tasks, recalls, tradeoff, grl, selection, small_config)
    maxent = decision["variants"]["maxent"]
    assert maxent["families"]["common"]["protection_rule_passed"]
    assert not maxent["families"]["rc"]["protection_rule_passed"]
    assert maxent["families"]["rc"]["utility"]["passed"]
    assert not maxent["protection_both_families"]
    assert not decision["question_i"]["answer"]


def test_frontier_empty_comparator_is_not_dominance(small_config):
    tradeoff = pd.DataFrame([
        dict(kind="task", family=f, d=8, target=0.0, epochs=5, selected=False,
             task_ba=0.7, identity_ba=0.5) for f in FAMILIES] + [
        dict(kind="maxent", family=f, d=8, target=0.1, epochs=5, selected=True,
             task_ba=0.75, identity_ba=0.1) for f in FAMILIES])
    grl = pd.DataFrame(columns=["condition", "target", "selected", "task_ba", "identity_ba"])
    result = frontier(tradeoff, grl, small_config)
    assert not result["answer"]
    point = result["families"]["common"]["points"][0]
    assert point["comparator_d"] == [] and point["comparator_min_identity_ba"] is None


def test_synthetic_dev_and_final_roundtrip(small_config, tmp_path):
    train, test = synthetic(per_run=6), synthetic("E", per_run=6)
    train.meta.loc[0, "artifact"] = True
    test.meta.loc[1, "finite"] = False
    test.X[1] = np.nan
    results, artifacts = tmp_path / "results", tmp_path / "artifacts"
    selection = run_dev(train, small_config, results)
    assert selection["sessions"] == ["T"] and selection["retained_trials"] == len(train) - 1
    for family in FAMILIES:
        assert set(selection["d1_budgets"][family]) == {"2", "8"}
        assert all(e in (1, 2, 3) for e in selection["d1_budgets"][family].values())
        for variant in ("maxent", "ensemble"):
            choice = selection["variants"][variant][family]
            assert choice["status"] == "selected" and choice["epochs"] in (1, 2)
            assert [c["target"] for c in choice["curve"]] == [0.1, 1.0]
    identity = pd.read_csv(results / "identity_cv.csv")
    assert (identity.attacker == "adversary_head").sum() == 2 * 2 * 6 * 4 * 2
    assert set(identity.loc[identity.attacker == "adversary_head", "kind"]) == {"maxent", "ensemble"}
    summary = pd.read_csv(results / "summary_cv.csv")
    assert len(summary) == 2 * (2 + 2 * 4)
    assert selection["attacker_convergence"]["mlp_fits"] == 6 * 2 * (2 + 2 * 4) * 2
    grl = tmp_path / "grl.csv"
    pd.DataFrame([dict(condition=c, target=t, selected=s, task_ba=0.6, identity_ba=0.5)
                  for c, t, s in (("P0", 0.0, False), ("P2", 0.0, False), ("P3", 0.1, True),
                                  ("P3", 1.0, False), ("P2-RC", 0.0, False),
                                  ("P3-RC", 0.1, True))]).to_csv(grl, index=False)
    combined = Trials(np.concatenate([train.X, test.X]),
                      pd.concat([train.meta, test.meta], ignore_index=True))
    decision = run_final(combined, selection, small_config, results, artifacts, grl)
    assert isinstance(decision["question_i"]["answer"], bool)
    assert isinstance(decision["question_ii"]["answer"], bool)
    assert set(decision["variants"]) == {"maxent", "ensemble"}
    assert json.loads((results / "decision.json").read_text()) == decision
    tradeoff = pd.read_csv(results / "tradeoff.csv")
    assert len(tradeoff) == 2 * (1 + 2 + 2 * 2) and tradeoff.selected.sum() == 4
    assert tradeoff.loc[tradeoff.kind.isin(["maxent", "ensemble"]), "head_main_ba"].notna().all()
    assert set(tradeoff.strongest_attacker) <= set(ATTACKERS)
    assert (results / "frontier.png").stat().st_size > 1000
    assert len(pd.read_csv(results / "task_final.csv")) == len(tradeoff) * 9
    manifest = json.loads((artifacts / "final" / "manifest.json").read_text())
    assert len(manifest["models"]) == len(tradeoff)
    assert manifest["attacker_convergence"]["mlp_fits"] == len(tradeoff) * 3
    for entry in manifest["models"]:
        folder = artifacts / "final" / entry["key"]
        model = joblib.load(folder / "seed0.joblib")
        z, logits = model(combined)
        with np.load(folder / "seed0_outputs.npz") as saved:
            np.testing.assert_array_equal(z, saved["representation"])
            np.testing.assert_array_equal(logits, saved["logits"])
        assert np.isnan(logits[len(train) + 1]).all()
        assert z.shape[1] == {"linear": 6, "task": entry["d"]}.get(entry["kind"], 8)
    changed = copy.deepcopy(small_config)
    changed["variants"]["ensemble"]["heads"] = 3
    with pytest.raises(ValueError, match="config/source"):
        run_final(combined, selection, changed, results, artifacts, grl)
