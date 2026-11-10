"""Synthetic-only Experiment E contracts: never load real session E."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from beat.config import load_config
from beat.data import LeakageError, Trials
from beat import exp_e


@pytest.fixture
def config():
    c = copy.deepcopy(load_config("exp_e"))
    c["workers"] = 1
    c["pilot"]["epoch_candidates"] = [1]
    c["pilot"]["held_out_runs"] = [1]
    c["statistics"]["bootstrap_samples"] = 10
    return c


@pytest.fixture
def trials():
    rng = np.random.default_rng(2026)
    rows = [dict(subject=s, session=session, run=run, trial=y + 1, label=y,
                 artifact=False, finite=True)
            for session in ("T", "E") for s in range(1, 10) for run in range(1, 7) for y in (0, 1)]
    X = rng.normal(size=(len(rows), 22, 750)).astype(np.float32)
    for i, row in enumerate(rows):
        X[i] *= 0.5 + row["subject"] / 9
        X[i, row["label"]] += np.sin(np.arange(750) * 0.3) * 2
    return Trials(X, pd.DataFrame(rows))


@pytest.fixture(autouse=True)
def restore_torch_state():
    threads, state = torch.get_num_threads(), torch.random.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    yield
    torch.set_num_threads(threads)
    torch.random.set_rng_state(state)
    torch.use_deterministic_algorithms(deterministic)


def test_retention_and_exact_decimation(trials, config):
    meta = trials.meta.copy()
    meta.loc[0, "artifact"] = True
    meta.loc[1, "finite"] = False
    kept = exp_e.prepare(Trials(trials.X, meta), config)
    assert kept.X.shape == (len(trials) - 2, 22, 375)
    np.testing.assert_array_equal(kept.X, trials.X[2:, :, ::2])


def test_alignment_whitens_training_covariance_and_does_not_refit(trials, config):
    prepared = exp_e.prepare(trials, config)
    train = prepared.subset(prepared.session == "T")
    reference = exp_e.Alignment("ea").fit(train)
    aligned = reference.transform(train)
    for subject in np.unique(train.subject):
        X = aligned.X[aligned.subject == subject].astype(float)
        X -= X.mean(axis=2, keepdims=True)
        mean = (X @ X.transpose(0, 2, 1) / 374).mean(axis=0)
        np.testing.assert_allclose(mean, np.eye(22), atol=2e-6)
    before = copy.deepcopy(reference.references)
    reference.transform(prepared.subset(prepared.session == "E"))
    for subject in before:
        np.testing.assert_array_equal(reference.references[subject], before[subject])


@pytest.mark.parametrize("entry", ["alignment_none", "alignment_ea", "network", "pipeline", "dev"])
def test_every_defended_fit_rejects_e(entry, trials, config, tmp_path):
    E = exp_e.prepare(trials.subset(trials.session == "E"), config)
    with pytest.raises(LeakageError):
        if entry.startswith("alignment"):
            exp_e.Alignment(entry.split("_")[1]).fit(E)
        elif entry == "network":
            exp_e.fit_network(E, config, 0, 1, 0)
        elif entry == "pipeline":
            exp_e.fit_defended(E, config, "ea", 0, 1, 0)
        else:
            exp_e.run_dev(trials, config, tmp_path, tmp_path)
    assert not list(tmp_path.iterdir())


def test_training_is_deterministic_frozen_and_trains_zero_lambda_identifier(trials, config):
    train = exp_e.prepare(trials.subset((trials.session == "T") & (trials.run == 1)), config)
    torch.manual_seed(4)
    initial = exp_e.EEGNet(config, 9)
    first, _, _, _ = exp_e.fit_network(train, config, 4, 1, 0)
    second, _, _, _ = exp_e.fit_network(train, config, 4, 1, 0)
    assert not torch.equal(initial.identifier[0].weight, first.identifier[0].weight)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
    z, task, head = exp_e.infer(first, train, 8)
    assert z.shape == (18, 176) and task.shape == (18, 2) and head.shape == (18, 9)
    assert not first.training and not any(p.requires_grad for p in first.parameters())
    assert first.spatial.weight.flatten(1).norm(dim=1).max() <= 1.000001
    assert first.task.weight.norm(dim=1).max() <= 0.250001


def test_identifier_gradient_reverses_backbone_not_identifier(config):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = exp_e.EEGNet(config, 9).eval()
    X = torch.randn(2, 22, 375)
    gradients = []
    for strength in (1.0, -1.0, 0.0):
        model.zero_grad(set_to_none=True)
        _, _, head = model(X, strength)
        nn_loss = torch.nn.functional.cross_entropy(head, torch.tensor([0, 1]))
        nn_loss.backward()
        gradients.append((model.encoder[1].weight.grad.clone(), model.identifier[0].weight.grad.clone()))
    torch.testing.assert_close(gradients[0][0], -gradients[1][0])
    torch.testing.assert_close(gradients[0][1], gradients[1][1])
    assert torch.count_nonzero(gradients[2][0]) == 0
    assert torch.count_nonzero(gradients[2][1]) > 0


def test_synthetic_dev_and_full_final_grid(trials, config, tmp_path):
    attackers = copy.deepcopy(load_config("exp_b")["attackers"])
    attackers.update(svm_C=[1.0], svm_gamma=["scale"], inner_folds=2,
                     mlp_max_iter=10, mlp_max_fun=30, logistic_max_iter=100)
    train = trials.subset(trials.session == "T")
    path = tmp_path / "config.yaml"
    selection = exp_e.run_dev(train, config, tmp_path / "dev", tmp_path / "artifacts",
                             config_path=path, attackers=attackers)
    chosen = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert selection["session_scope"] == ["T"]
    assert selection["budgets"] == {"none": 1, "ea": 1}
    result = tmp_path / "final"
    summary = exp_e.run_final(trials, selection, chosen, result, tmp_path / "artifacts", attackers)
    assert summary["model_count"] == 36
    assert summary["mlp_fits"] == 108
    assert len(summary["per_lambda"]) == 12
    task = pd.read_csv(result / "task_final.csv")
    identity = pd.read_csv(result / "identity_final.csv")
    directional = pd.read_csv(result / "identity_directional.csv")
    assert len(task) == 36 * 9
    assert len(identity) == 36 * 2 * 3
    assert set(identity.attacker) == {"logistic", "rbf_svm", "mlp"}
    assert set(identity.scenario) == {"main", "auxiliary"}
    assert len(directional[directional.scenario == "main"]) == 36 * 2 * 3
    for row in summary["per_lambda"]:
        selected = identity[(identity.alignment == row["alignment"]) & (identity.target == row["target"])]
        expected = selected.groupby(["scenario", "attacker"]).ba.mean().max()
        assert row["strongest_ba"] == pytest.approx(expected)
    for alignment in ("none", "ea"):
        assert (result / f"identifier_vs_independent_{alignment}.png").stat().st_size > 1000
    assert json.loads((result / "summary.json").read_text())["model_count"] == 36
    checkpoint = torch.load(tmp_path / "artifacts/final/ea/lambda0/seed0/model.pt", weights_only=True)
    model = exp_e.EEGNet(chosen, 9)
    model.load_state_dict(checkpoint["state"])
    assert len(checkpoint["references"]) == 9
    changed = copy.deepcopy(chosen)
    changed["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="changed since dev"):
        exp_e.run_final(trials, selection, changed, result, tmp_path / "artifacts", attackers)
    shifted = Trials(trials.X + 1, trials.meta)
    with pytest.raises(ValueError, match="Retained T differs"):
        exp_e.run_final(shifted, selection, chosen, result, tmp_path / "artifacts", attackers)
