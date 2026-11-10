"""Experiment D (plan v2, exploratory): bottleneck sweep and stronger adversaries.

``python -m beat.exp_d dev`` uses session T only and writes the selection;
``python -m beat.exp_d final`` (after seal 2) retrains on retained T, freezes,
and evaluates session E exactly like Experiment B. Experiment B's frozen
GRL-CE models and results are reused as the reference and never retrained.

D1: task-only encoders 253 -> 64 -> d for every d in ``bottlenecks``; the
epoch budget per (d, family) is B's choose_epochs on fold/seed-mean T-CV task
BA. D2 at d = 8:
- MaxEnt: simultaneous updates with one Adam optimizer. The adversary
  minimizes identity cross-entropy on the detached bottleneck; the encoder
  minimizes ramp(lambda) x CE(uniform, adversary posterior), evaluated with the
  adversary's parameters detached.
- Ensemble-refresh: 5 heads behind gradient reversal of strength
  ramp(lambda) / 5 with summed head losses, so every head trains on its own CE
  and the encoder receives the reversed gradient of the heads' mean CE. At the
  start of epoch e >= 2, head (e - 2) mod 5 is re-initialized and its Adam
  state cleared.
Epoch budget (from ``epoch_grid``) and lambda are chosen jointly by B's
operating-point rule against the same family's d = 8 task-only encoder.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from joblib import delayed
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F

from .attacks import ATTACKERS
from .config import ROOT, load_config
from .data import Trials, quality_mask, require_training
from .exp_b import (FAMILIES, _condition, _convergence_summary, _dev_attacks, _hash,
                    _json, _load_covariances, _parallel, _save_outputs, _task_rows,
                    _training_hash, _winner, choose_epochs, evaluate_attack_scenarios,
                    make_decision)
from .models import (Encoder, FeatureMap, FrozenModel, fit_defended, fit_encoder,
                     freeze_encoder, ramp, reverse_gradient)
from .splits import t_run_folds

RESULTS = ROOT / "results" / "exp_d"
ARTIFACTS = ROOT / "artifacts" / "exp_d"
VARIANTS = ("maxent", "ensemble")
LABELS = {"maxent": "MaxEnt", "ensemble": "Ensemble"}
SUMMARY_KEYS = ["kind", "condition", "family", "d", "target", "epochs"]
ORDER = (["identity_ba", "task_ba", "target", "epochs"], [True, False, True, True])


def load_d_config() -> dict:
    """Experiment D settings merged with the B sections D must share."""
    config = load_config("exp_d")
    base = load_config("exp_b")
    for key in config["inherit_from_exp_b"]:
        config[key] = copy.deepcopy(base[key])
    return config


def _source_hash():
    digest = hashlib.sha256()
    for name in ("exp_d.py", "exp_b.py", "models.py", "attacks.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def _suffix(family):
    return "-RC" if family == "rc" else ""


def task_condition(d, family):
    return f"D1-d{int(d)}{_suffix(family)}"


def variant_condition(variant, family):
    return LABELS[variant] + _suffix(family)


def setting_key(setting):
    key = setting["condition"]
    if setting["kind"] in VARIANTS:
        key += f"_lambda{float(setting['target']):g}"
    return key + (f"_e{setting['epochs']}" if setting["epochs"] else "")


def _training(config, d):
    return dict(config["training"], bottleneck=int(d))


def _variant_training(config):
    return dict(_training(config, config["reference_bottleneck"]),
                ensemble_heads=config["variants"]["ensemble"]["heads"])


class EnsembleHeads(nn.Module):
    """K identity heads; forward returns the log of their mean posterior."""

    def __init__(self, bottleneck, hidden, identities, count):
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(bottleneck, hidden), nn.ReLU(), nn.Linear(hidden, identities))
            for _ in range(count))

    def forward(self, z):
        log_posteriors = torch.stack([F.log_softmax(head(z), dim=1) for head in self.heads])
        return torch.logsumexp(log_posteriors, dim=0) - np.log(len(self.heads))


class EnsembleEncoder(nn.Module):
    """B's encoder and task head with an ensemble adversary.

    Modules are created in B's order, so for one seed the encoder, task head and
    first adversary head start from exactly B's P2/P3 initialization.
    """

    def __init__(self, dimension, identities, config):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dimension, config["encoder_hidden"]), nn.ReLU(),
            nn.Linear(config["encoder_hidden"], config["bottleneck"]))
        self.task = nn.Linear(config["bottleneck"], 2)
        self.adversary = EnsembleHeads(config["bottleneck"], config["identity_hidden"],
                                       identities, config["ensemble_heads"])

    def forward(self, x, strength=0.0):
        representation = self.encoder(x)
        return representation, self.task(representation), self.adversary(representation)


def variant_loss(model, variant, x, y, identity, strength):
    """One minibatch objective; gradients reach each module only as documented."""
    z = model.encoder(x)
    loss = F.cross_entropy(model.task(z), y)
    if variant == "maxent":
        adversary = model.adversary
        loss = loss + F.cross_entropy(adversary(z.detach()), identity)
        frozen = {name: p.detach() for name, p in adversary.named_parameters()}
        posterior = torch.func.functional_call(adversary, frozen, (z,))
        return loss - strength * F.log_softmax(posterior, dim=1).mean()
    heads = model.adversary.heads
    reversed_z = reverse_gradient(z, strength / len(heads))
    return loss + sum(F.cross_entropy(head(reversed_z), identity) for head in heads)


def refresh_index(epoch, heads):
    """Head re-initialized at the start of ``epoch`` (None for the first epoch)."""
    return None if epoch < 2 else (epoch - 2) % heads


def fit_variant(train: Trials, variant: str, config: dict, seed: int, epochs: int,
                target: float):
    """Train a D2 network on standardized T features; returns the trained module.

    Initialization, optimizer, minibatch order and lambda ramp follow B's
    fit_encoder for the same seed.
    """
    require_training(train)
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant}")
    if epochs < 1 or target <= 0:
        raise ValueError("epochs and lambda must be positive")
    if config["optimizer"] != "adam":
        raise ValueError("Only the prespecified adam optimizer is supported")
    classes, identity = np.unique(train.subject, return_inverse=True)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = (Encoder if variant == "maxent" else EnsembleEncoder)(
        train.X.shape[1], len(classes), config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    X = torch.as_tensor(train.X, dtype=torch.float32)
    y = torch.as_tensor(train.y, dtype=torch.long)
    identity = torch.as_tensor(identity, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    size = config["batch_size"]
    steps = (len(train) + size - 1) // size
    for epoch in range(1, epochs + 1):
        if variant == "ensemble":
            index = refresh_index(epoch, len(model.adversary.heads))
            if index is not None:
                head = model.adversary.heads[index]
                for layer in head:
                    if isinstance(layer, nn.Linear):
                        layer.reset_parameters()
                for parameter in head.parameters():
                    optimizer.state.pop(parameter, None)
        model.train()
        order = torch.randperm(len(train), generator=generator)
        for batch, start in enumerate(range(0, len(train), size)):
            idx = order[start:start + size]
            progress = ((epoch - 1) * steps + batch) / max(epochs * steps - 1, 1)
            loss = variant_loss(model, variant, X[idx], y[idx], identity[idx],
                                ramp(target, progress))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def fit_variant_pipeline(train: Trials, family: str, variant: str, config: dict, seed: int,
                         epochs: int, target: float) -> FrozenModel:
    """Fit features and a D2 network on T covariances; session E is rejected first."""
    require_training(train)
    features = FeatureMap(family == "rc").fit(train)
    _, X = features.transform(train)
    model = fit_variant(train.with_X(X), variant, config, seed, epochs, target)
    return FrozenModel(features, model, True, np.unique(train.subject))


def _head_prediction(model, trials):
    return model.identity_classes_[model.adversary_logits(trials).argmax(axis=1)]


def _pilot(train, val, family, seed, run, config):
    torch.set_num_threads(config["torch_threads"])
    with threadpool_limits(limits=1):
        features = FeatureMap(family == "rc").fit(train)
        _, X = features.transform(train)
        _, Xv = features.transform(val)
        Xv = torch.as_tensor(Xv)
        identities = len(np.unique(train.subject))
        maximum = config["training"]["max_epochs"]
        interval = config["training"]["checkpoint_interval"]
        checkpoints = sorted(set(range(interval, maximum + 1, interval)) | {maximum})
        rows = []
        for d in config["bottlenecks"]:
            training = _training(config, d)
            states = fit_encoder(train.with_X(X), training, seed, maximum, 0.0, checkpoints)
            network = Encoder(X.shape[1], identities, training).eval()
            for epoch, state in states.items():
                network.load_state_dict(state)
                with torch.inference_mode():
                    logits = network(Xv)[1].numpy()
                rows.append(dict(family=family, d=int(d), seed=seed, fold=run, epoch=epoch,
                                 ba=float(np.mean([r["ba"] for r in _task_rows(val, logits)]))))
        return rows


def _evaluate_dev(models, val, seed, config):
    task_rows, identity_rows, tuning_rows = [], [], []
    for tags, model, head in models:
        z, logits = model(val)
        task_rows.extend(_task_rows(val, logits, **tags))
        scores, tuning = _dev_attacks(z, val, config, seed,
                                      _head_prediction(model, val) if head else None)
        identity_rows.extend(dict(**tags, **row) for row in scores)
        tuning_rows.extend(dict(**tags, **row) for row in tuning)
    return task_rows, identity_rows, tuning_rows


def _dev_task_only(train, val, family, seed, run, budgets, config):
    torch.set_num_threads(config["torch_threads"])
    with threadpool_limits(limits=1):
        features = FeatureMap(family == "rc").fit(train)
        _, X = features.transform(train)
        models = []
        for d in config["bottlenecks"]:
            epochs = budgets[str(d)]
            training = _training(config, d)
            state = fit_encoder(train.with_X(X), training, seed, epochs, 0.0)[epochs]
            model = freeze_encoder(features, state, X.shape[1], np.unique(train.subject), training)
            tags = dict(kind="task", condition=task_condition(d, family), family=family,
                        d=int(d), target=0.0, epochs=epochs, seed=seed, fold=run)
            models.append((tags, model, False))
        return _evaluate_dev(models, val, seed, config)


def _dev_variant(train, val, family, variant, seed, run, config):
    torch.set_num_threads(config["torch_threads"])
    with threadpool_limits(limits=1):
        features = FeatureMap(family == "rc").fit(train)
        _, X = features.transform(train)
        training = _variant_training(config)
        models = []
        for epochs in config["epoch_grid"]:
            for target in config["lambdas"]:
                network = fit_variant(train.with_X(X), variant, training, seed, epochs, target)
                model = FrozenModel(features, network, True, np.unique(train.subject))
                tags = dict(kind=variant, condition=variant_condition(variant, family),
                            family=family, d=config["reference_bottleneck"],
                            target=float(target), epochs=int(epochs), seed=seed, fold=run)
                models.append((tags, model, True))
        return _evaluate_dev(models, val, seed, config)


def summarize_cv(tasks, identities):
    """Fold/seed means: task BA, max over attackers of the attacker-wise mean BA."""
    task = tasks.groupby(SUMMARY_KEYS, as_index=False).ba.mean().rename(columns={"ba": "task_ba"})
    independent = identities[identities.attacker.isin(ATTACKERS)]
    attack = independent.groupby(SUMMARY_KEYS + ["attacker"], as_index=False).ba.mean()
    strongest = attack.groupby(SUMMARY_KEYS, as_index=False).ba.max().rename(
        columns={"ba": "identity_ba"})
    head = identities[identities.attacker == "adversary_head"].groupby(
        SUMMARY_KEYS, as_index=False).ba.mean().rename(columns={"ba": "adversary_ba"})
    summary = task.merge(strongest, on=SUMMARY_KEYS, validate="one_to_one")
    return summary.merge(head, on=SUMMARY_KEYS, how="left", validate="one_to_one")


def _optional(value):
    return None if pd.isna(value) else float(value)


def choose_variants(summary, config):
    """Joint (epochs, lambda) choice per variant and family, solely from T CV."""
    margin = config["statistics"]["utility_margin"]
    choices = {}
    for variant in VARIANTS:
        choices[variant] = {}
        for family in FAMILIES:
            reference = summary[(summary.kind == "task") & (summary.family == family)
                                & (summary.d == config["reference_bottleneck"])]
            baseline = float(reference.task_ba.iloc[0])
            rows = summary[(summary.kind == variant) & (summary.family == family)].copy()
            rows["eligible"] = rows.task_ba >= baseline - margin - 1e-12
            eligible = rows[rows.eligible].sort_values(ORDER[0], ascending=ORDER[1])
            curve = []
            for target, group in rows.groupby("target", sort=True):
                passing = group[group.eligible]
                pick = (passing.sort_values(ORDER[0], ascending=ORDER[1]) if len(passing)
                        else group.sort_values(["task_ba", "epochs"], ascending=[False, True]))
                curve.append(dict(target=float(target), epochs=int(pick.epochs.iloc[0]),
                                  eligible=bool(len(passing))))
            best = eligible.iloc[0] if len(eligible) else None
            choices[variant][family] = dict(
                status="selected" if best is not None else "no_operating_point",
                target=float(best.target) if best is not None else None,
                epochs=int(best.epochs) if best is not None else None,
                task_ba=float(best.task_ba) if best is not None else None,
                identity_ba=float(best.identity_ba) if best is not None else None,
                adversary_ba=_optional(best.adversary_ba) if best is not None else None,
                reference_task_ba=baseline,
                eligible=[dict(target=float(r.target), epochs=int(r.epochs))
                          for r in eligible.itertuples()],
                curve=curve)
    return choices


def run_dev(trials: Trials, config=None, result_dir=RESULTS):
    """All T-only development on covariance Trials; never touches session E."""
    require_training(trials)
    config = config or load_d_config()
    result_dir = Path(result_dir)
    start = time.perf_counter()
    train = trials.subset(quality_mask(trials))
    folds = list(t_run_folds(train))
    pilot = _parallel(config, (
        delayed(_pilot)(fit, val, family, seed, run, config)
        for family in FAMILIES for seed in config["seeds"] for run, fit, val in folds))
    history = pd.DataFrame([row for rows in pilot for row in rows])
    budgets = {family: {} for family in FAMILIES}
    for d, frame in history.groupby("d", sort=True):
        for family, epochs in choose_epochs(frame).items():
            budgets[family][str(int(d))] = epochs
    result_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(result_dir / "d1_epoch_cv.csv", index=False)
    print(f"D1 budgets (T-only): {budgets}", flush=True)
    jobs = [delayed(_dev_variant)(fit, val, family, variant, seed, run, config)
            for variant in reversed(VARIANTS) for family in FAMILIES
            for seed in config["seeds"] for run, fit, val in folds]
    jobs += [delayed(_dev_task_only)(fit, val, family, seed, run, budgets[family], config)
             for family in FAMILIES for seed in config["seeds"] for run, fit, val in folds]
    evaluated = _parallel(config, jobs)
    tables = [pd.DataFrame([row for bundle in evaluated for row in bundle[i]]) for i in range(3)]
    tasks, identities, tuning = tables
    for name, table in zip(("task_cv", "identity_cv", "attacker_tuning_cv"), tables):
        table.to_csv(result_dir / f"{name}.csv", index=False)
    summary = summarize_cv(tasks, identities)
    summary.to_csv(result_dir / "summary_cv.csv", index=False)
    variants = choose_variants(summary, config)
    selection = dict(
        schema=1, phase="T_only_development", sessions=["T"], config=config,
        config_hash=_hash(config), source_hash=_source_hash(),
        training_hash=_training_hash(train), retained_trials=len(train),
        d1_budgets=budgets, variants=variants,
        attacker_convergence=_convergence_summary(tuning),
        wall_seconds=time.perf_counter() - start,
        rules=dict(epoch_tie="earliest", joint_tie=config["selection"]["tie_order"],
                   curve_epochs=config["selection"]["curve_epochs"],
                   task_weighting="equal_subject_fold_seed",
                   identity_weighting="equal_direction_fold_seed_then_max_attacker"))
    _json(result_dir / "selection.json", selection)
    print(json.dumps({k: selection[k] for k in ("d1_budgets", "wall_seconds",
                                                "attacker_convergence")}), flush=True)
    for variant, families in variants.items():
        for family, choice in families.items():
            print(f"{variant}/{family}: {choice['status']} lambda={choice['target']} "
                  f"epochs={choice['epochs']} task={choice['task_ba']} "
                  f"identity={choice['identity_ba']}", flush=True)
    return selection


def final_settings(selection, config):
    """P0 references, the D1 sweep, and each variant's per-lambda curve."""
    for family in FAMILIES:
        yield dict(family=family, kind="linear", condition=_condition("P0", family),
                   d=0, target=0.0, epochs=0)
        for d in config["bottlenecks"]:
            yield dict(family=family, kind="task", condition=task_condition(d, family),
                       d=int(d), target=0.0, epochs=selection["d1_budgets"][family][str(d)])
        for variant in VARIANTS:
            for point in selection["variants"][variant][family]["curve"]:
                yield dict(family=family, kind=variant,
                           condition=variant_condition(variant, family),
                           d=config["reference_bottleneck"], target=point["target"],
                           epochs=point["epochs"])


def _final_model(all_trials, setting, seed, config, artifact_dir):
    torch.set_num_threads(config["torch_threads"])
    with threadpool_limits(limits=1):
        t_idx = (all_trials.session == "T") & quality_mask(all_trials)
        e_idx = (all_trials.session == "E") & quality_mask(all_trials)
        train, test = all_trials.subset(t_idx), all_trials.subset(e_idx)
        family, kind = setting["family"], setting["kind"]
        if kind == "linear":
            model = fit_defended(train, setting["condition"], config["training"], seed, 1)
        elif kind == "task":
            model = fit_defended(train, _condition("P2", family),
                                 _training(config, setting["d"]), seed, setting["epochs"])
        else:
            model = fit_variant_pipeline(train, family, kind, _variant_training(config), seed,
                                         setting["epochs"], setting["target"])
        folder = artifact_dir / "final" / setting_key(setting)
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, folder / f"seed{seed}.joblib")
        z, logits = model(all_trials)
        head_logits = model.adversary_logits(all_trials) if kind in VARIANTS else None
        _save_outputs(folder / f"seed{seed}_outputs.npz", all_trials, logits, z,
                      head_logits, model.identity_classes_)
        tags = dict(setting, seed=seed)
        tasks = _task_rows(test, logits[e_idx], **tags)
        scores, directional, recalls, predictions, tuning = evaluate_attack_scenarios(
            z[t_idx], train, z[e_idx], test, config, seed,
            model.identity_classes_[head_logits[e_idx].argmax(axis=1)]
            if head_logits is not None else None)
        predictions.to_csv(folder / f"seed{seed}_attack_predictions.csv.gz", index=False)
        tables = [scores, directional, recalls, pd.DataFrame(tuning)]
        for table in tables:
            for name, value in tags.items():
                table[name] = value
        return tasks, *(table.to_dict("records") for table in tables)


def _selected(selection, setting):
    if setting["kind"] not in VARIANTS:
        return False
    return selection["variants"][setting["kind"]][setting["family"]]["target"] == setting["target"]


def tradeoff_table(tasks, scores, recalls, settings, selection):
    """Seed-mean task BA/kappa; strongest independent attackers by scenario."""
    rows = []
    auxiliary = recalls[recalls.scenario == "auxiliary"]
    for setting in settings:
        condition, target = setting["condition"], setting["target"]
        task = tasks[(tasks.condition == condition) & (tasks.target == target)]
        _, both, _ = _winner(recalls, condition, target)
        _, main, _ = _winner(recalls, condition, target, main_only=True)
        _, aux, _ = _winner(auxiliary, condition, target)
        head = scores[(scores.condition == condition) & (scores.target == target)
                      & (scores.attacker == "adversary_head")].groupby("scenario").ba.mean()
        rows.append(dict(setting, selected=_selected(selection, setting),
                         task_ba=float(task.ba.mean()), kappa=float(task.kappa.mean()),
                         identity_ba=both["ba"], strongest_attacker=both["attacker"],
                         strongest_scenario=both["scenario"],
                         identity_main_ba=main["ba"], main_attacker=main["attacker"],
                         identity_auxiliary_ba=aux["ba"], auxiliary_attacker=aux["attacker"],
                         head_main_ba=float(head.get("main", np.nan)),
                         head_auxiliary_ba=float(head.get("auxiliary", np.nan))))
    return pd.DataFrame(rows)


def frontier(tradeoff, grl, config):
    """Question (ii) per family; empty comparator sets are not dominance."""
    families = {}
    for family in FAMILIES:
        task_only = tradeoff[(tradeoff.kind == "task") & (tradeoff.family == family)]
        points = [dict(variant=r.kind, source="exp_d", target=float(r.target),
                       epochs=int(r.epochs), task_ba=float(r.task_ba),
                       identity_ba=float(r.identity_ba))
                  for r in tradeoff[(tradeoff.family == family) & tradeoff.selected].itertuples()]
        points += [dict(variant="grl_ce", source="exp_b", target=float(r.target), epochs=None,
                        task_ba=float(r.task_ba), identity_ba=float(r.identity_ba))
                   for r in grl[(grl.condition == _condition("P3", family))
                                & grl.selected.astype(bool)].itertuples()]
        for point in points:
            comparators = task_only[task_only.task_ba >= point["task_ba"] - 1e-12]
            point["comparator_d"] = [int(d) for d in comparators.d]
            point["comparator_min_identity_ba"] = (float(comparators.identity_ba.min())
                                                   if len(comparators) else None)
            point["dominates"] = bool(len(comparators)) and bool(
                point["identity_ba"] < comparators.identity_ba.min())
        families[family] = dict(points=points,
                                any_dominates=any(p["dominates"] for p in points))
    return dict(rule=config["decision"]["frontier_empty_comparator_set"], families=families,
                answer=any(f["any_dominates"] for f in families.values()))


def make_decision_d(tasks, recalls, tradeoff, grl, selection, config):
    """B's make_decision per variant, relabelled: d = 8 task-only as P2, variant as P3."""
    reference = config["reference_bottleneck"]
    variants = {}
    for variant in VARIANTS:
        rename = {}
        for family in FAMILIES:
            rename[task_condition(reference, family)] = _condition("P2", family)
            rename[variant_condition(variant, family)] = _condition("P3", family)
        keep = list(rename) + ["P0"]
        view_tasks = tasks[tasks.condition.isin(keep)].assign(
            condition=lambda f: f.condition.replace(rename))
        view_recalls = recalls[recalls.condition.isin(keep)].assign(
            condition=lambda f: f.condition.replace(rename))
        view_selection = {"families": {f: {"target": selection["variants"][variant][f]["target"]}
                                       for f in FAMILIES}}
        verdict = make_decision(view_tasks, view_recalls, view_selection, config)
        families = {}
        for family, report in verdict["families"].items():
            choice = selection["variants"][variant][family]
            protection = report.get("protection", {})
            point = tradeoff[(tradeoff.kind == variant) & (tradeoff.family == family)
                             & tradeoff.selected]
            families[family] = dict(
                report, epochs=choice["epochs"],
                protection_rule_passed=bool(protection) and not verdict["baseline_at_chance"]
                and all(v["passed"] for v in protection.values()),
                selected_point=None if point.empty else dict(
                    task_ba=float(point.task_ba.iloc[0]),
                    identity_ba=float(point.identity_ba.iloc[0]),
                    identity_main_ba=float(point.identity_main_ba.iloc[0]),
                    identity_auxiliary_ba=float(point.identity_auxiliary_ba.iloc[0]),
                    head_main_ba_diagnostic=_optional(point.head_main_ba.iloc[0]),
                    head_auxiliary_ba_diagnostic=_optional(point.head_auxiliary_ba.iloc[0])))
        # B's protection bullet accepts a contribution only if it holds in both families.
        variants[variant] = dict(families=families, protection_both_families=all(
                                     f["protection_rule_passed"] for f in families.values()),
                                 baseline_p0=verdict["baseline_p0"],
                                 baseline_at_chance=verdict["baseline_at_chance"],
                                 accepted_both_families=verdict["h2_accepted"])
    consistency = {}
    for family in FAMILIES:
        ours = tradeoff[tradeoff.condition == task_condition(reference, family)].iloc[0]
        theirs = grl[grl.condition == _condition("P2", family)].iloc[0]
        consistency[family] = dict(task_ba_diff=float(ours.task_ba - theirs.task_ba),
                                   identity_ba_diff=float(ours.identity_ba - theirs.identity_ba))
    return dict(
        schema=1, status="exploratory", reference_bottleneck=reference, variants=variants,
        question_i=dict(rule=config["decision"]["question_i"], answer=any(
            v["protection_both_families"] for v in variants.values())),
        question_ii=frontier(tradeoff, grl, config),
        exp_b_p2_consistency=consistency,
        interpretation="Protection is judged only with independent attackers against the same "
                       "family's d = 8 task-only encoder; co-trained head BA is diagnostic.",
        selection_hash=_hash(selection))


def _figure(tradeoff, grl, config, result_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    styles = {"maxent": ("tab:red", "o"), "ensemble": ("tab:purple", "D")}
    for family, ax in zip(FAMILIES, axes):
        rows = tradeoff[tradeoff.family == family]
        sweep = rows[rows.kind == "task"].sort_values("d")
        ax.plot(sweep.task_ba, sweep.identity_ba, "-^", color="tab:green",
                label="Task-only bottleneck d sweep")
        for r in sweep.itertuples():
            ax.annotate(f"d={r.d}", (r.task_ba, r.identity_ba), xytext=(4, -10),
                        textcoords="offset points", fontsize=7, color="tab:green")
        for variant, (color, marker) in styles.items():
            curve = rows[rows.kind == variant].sort_values("target")
            ax.plot(curve.task_ba, curve.identity_ba, "-" + marker, color=color, markersize=4,
                    linewidth=1, label=f"{LABELS[variant]} lambda curve")
            for r in curve.itertuples():
                ax.annotate(f"{r.target:g}", (r.task_ba, r.identity_ba), xytext=(3, 3),
                            textcoords="offset points", fontsize=6, color=color)
            chosen = curve[curve.selected]
            ax.scatter(chosen.task_ba, chosen.identity_ba, marker="*", s=220, color=color,
                       edgecolor="black", zorder=5, label=f"{LABELS[variant]} T-selected")
        reference = grl[grl.condition == _condition("P3", family)].sort_values("target")
        ax.plot(reference.task_ba, reference.identity_ba, "--x", color="grey", linewidth=1,
                label="GRL-CE lambda curve (Exp. B)")
        chosen = reference[reference.selected.astype(bool)]
        ax.scatter(chosen.task_ba, chosen.identity_ba, marker="*", s=220, color="grey",
                   edgecolor="black", zorder=5, label="GRL-CE T-selected (Exp. B)")
        p0 = rows[rows.kind == "linear"]
        ax.scatter(p0.task_ba, p0.identity_ba, marker="s", color="black",
                   label=f"{_condition('P0', family)} tangent vector")
        ax.axhline(config["statistics"]["chance"], color="grey", linestyle=":",
                   label="Identity chance (1/9)")
        ax.set(title="Common reference" if family == "common" else "Per-subject recentering",
               xlabel="E task balanced accuracy (higher is better)", ylim=(0, 1.02))
        ax.legend(fontsize=7, loc="best")
    axes[0].set_ylabel("Strongest independent-attacker identity BA (lower is better)")
    fig.suptitle("Experiment D (exploratory): bottleneck sweep vs adversarial variants")
    fig.text(0.5, 0.01, "Seed means over 5 seeds; identity = max over attackers and main/auxiliary "
             "scenarios. Stars: T-only selected operating points.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(result_dir / "frontier.png", dpi=180)
    plt.close(fig)


def run_final(trials: Trials, selection: dict, config=None, result_dir=RESULTS,
              artifact_dir=ARTIFACTS, grl_reference=None):
    """Retrain/freeze on retained T and evaluate E. Use synthetic data before seal 2."""
    config = config or load_d_config()
    if set(trials.session) != {"T", "E"}:
        raise ValueError("Final requires both T and E covariance Trials")
    if selection["config_hash"] != _hash(config) or selection["source_hash"] != _source_hash():
        raise ValueError("Selection config/source differs; rerun T-only dev before sealing")
    train = trials.subset((trials.session == "T") & quality_mask(trials))
    if selection["training_hash"] != _training_hash(train):
        raise ValueError("Final retained T differs from development")
    grl = pd.read_csv(grl_reference or ROOT / config["decision"]["grl_reference"])
    result_dir, artifact_dir = Path(result_dir), Path(artifact_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    settings = list(final_settings(selection, config))
    evaluated = _parallel(config, (
        delayed(_final_model)(trials, setting, seed, config, artifact_dir)
        for setting in settings for seed in config["seeds"]))
    tables = [pd.DataFrame([row for bundle in evaluated for row in bundle[i]]) for i in range(5)]
    for name, table in zip(("task_final", "identity_final", "identity_directional",
                            "identity_recall", "attacker_tuning_final"), tables):
        table.to_csv(result_dir / f"{name}.csv", index=False)
    tasks, scores, _, recalls, tuning = tables
    subject_means = tasks.groupby(["condition", "target", "subject"],
                                  as_index=False)[["ba", "kappa"]].mean()
    subject_means.to_csv(result_dir / "task_subject_means.csv", index=False)
    subject_means.groupby(["condition", "target"])[["ba", "kappa"]].agg(
        ["mean", "median", "std"]).to_csv(result_dir / "task_summary.csv")
    tradeoff = tradeoff_table(tasks, scores, recalls, settings, selection)
    tradeoff.to_csv(result_dir / "tradeoff.csv", index=False)
    decision = make_decision_d(tasks, recalls, tradeoff, grl, selection, config)
    _json(result_dir / "decision.json", decision)
    _figure(tradeoff, grl, config, result_dir)
    manifest = dict(schema=1, seeds=config["seeds"], selection_hash=_hash(selection),
                    config_hash=_hash(config), source_hash=_source_hash(),
                    wall_seconds=time.perf_counter() - start,
                    attacker_convergence=_convergence_summary(tuning),
                    all_trials=len(trials), retained_T=len(train),
                    retained_E=int(((trials.session == "E") & quality_mask(trials)).sum()),
                    models=[dict(s, key=setting_key(s)) for s in settings])
    _json(artifact_dir / "final" / "manifest.json", manifest)
    print(json.dumps(dict(question_i=decision["question_i"]["answer"],
                          question_ii=decision["question_ii"]["answer"],
                          attacker_convergence=manifest["attacker_convergence"],
                          wall_seconds=manifest["wall_seconds"])), flush=True)
    return decision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["dev", "final"])
    args = parser.parse_args(argv)
    config = load_d_config()
    if args.phase == "dev":
        start = time.perf_counter()
        selection = run_dev(_load_covariances(True), config)
        selection["command_wall_seconds"] = time.perf_counter() - start
        _json(RESULTS / "selection.json", selection)
        print(f"Full dev wall time: {selection['command_wall_seconds']:.1f}s", flush=True)
    else:
        selection = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
        run_final(_load_covariances(False), selection, config)


if __name__ == "__main__":
    main()
