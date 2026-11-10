"""Experiment B CLI and frozen-model interface for Experiment C.

Run ``python -m beat.exp_b dev`` before the E seal. Only ``final`` loads E.
All public fit/development entry points reject E. Final selection is read-only:
E results never change budgets, lambdas, thresholds, or selected OOF logits.

C API (trusted local artifacts only):
    model = load_frozen("P3-RC", seed=0)
    representation, logits = model(covariance_trials)
    saved = load_outputs("P3-RC", seed=0)  # all T/E rows, flags preserved
    oof = load_oof("P3-RC", seed=0)       # retained T, each row held out once

NPZ mappings contain representation (except OOF), logits, and META_COLUMNS.
Final P3 outputs additionally contain adversary_logits and adversary_classes.
The frozen model's separate adversary_logits(trials) method is diagnostic only;
its ordinary two-item return value and the independent-attacker API are unchanged.
Logit columns are left/right. No-operating-point families have no default
frozen model or OOF file; callers must handle that explicit failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from .attacks import ATTACKERS, fit_attackers
from .config import ROOT, load_config
from .data import META_COLUMNS, Trials, load_trials, quality_mask, require_training
from .features import covariances
from .models import (FeatureMap, fit_defended, fit_encoder, fit_linear,
                     freeze_encoder)
from .splits import E_HALVES, t_run_folds
from .stats import paired_t_lower_bound, sign_flip_pvalue

RESULTS = ROOT / "results" / "exp_b"
ARTIFACTS = ROOT / "artifacts" / "exp_b"
FAMILIES = ("common", "rc")


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _source_hash():
    digest = hashlib.sha256()
    for name in ("exp_b.py", "models.py", "attacks.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def _training_hash(trials):
    digest = hashlib.sha256(np.ascontiguousarray(trials.X).tobytes())
    digest.update(trials.meta[META_COLUMNS].to_csv(index=False).encode())
    return digest.hexdigest()


def _condition(prefix, family):
    return prefix + ("-RC" if family == "rc" else "")


def setting_key(condition, target=0.0):
    return condition + (f"_lambda{float(target):g}" if condition.startswith("P3") else "")


def settings(config):
    for family in FAMILIES:
        for prefix in ("P0", "P2"):
            yield family, _condition(prefix, family), 0.0
        for target in config["lambdas"]:
            yield family, _condition("P3", family), float(target)


def _save_outputs(path, trials, logits, representation=None, adversary_logits=None,
                  adversary_classes=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {c: trials.meta[c].to_numpy() for c in META_COLUMNS}
    arrays["session"] = arrays["session"].astype("U1")
    arrays["logits"] = logits
    if representation is not None:
        arrays["representation"] = representation
    if adversary_logits is not None:
        arrays["adversary_logits"] = adversary_logits
        arrays["adversary_classes"] = adversary_classes
    np.savez_compressed(path, **arrays)


def _load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _task_rows(trials, logits, **tags):
    prediction = logits.argmax(axis=1)
    rows = []
    for subject in np.unique(trials.subject):
        idx = trials.subject == subject
        matrix = confusion_matrix(trials.y[idx], prediction[idx], labels=[0, 1])
        rows.append(dict(**tags, subject=int(subject),
                         ba=float(balanced_accuracy_score(trials.y[idx], prediction[idx])),
                         kappa=float(cohen_kappa_score(trials.y[idx], prediction[idx])),
                         n=int(idx.sum()), tn=int(matrix[0, 0]), fp=int(matrix[0, 1]),
                         fn=int(matrix[1, 0]), tp=int(matrix[1, 1])))
    return rows


def _task_mean(trials, logits):
    return float(np.mean([row["ba"] for row in _task_rows(trials, logits)]))


def _attack_predictions(train_z, train_meta, test_z, test_meta, config, seed,
                        scenario, direction):
    fitted = fit_attackers(train_z, train_meta["subject"].to_numpy(),
                          config["attackers"], seed)
    predictions, tuning = [], []
    for name, attacker in fitted.items():
        frame = test_meta[META_COLUMNS].copy()
        frame["prediction"] = attacker.predict(test_z)
        frame["attacker"], frame["scenario"], frame["direction"] = name, scenario, direction
        predictions.append(frame)
        if name == "rbf_svm":
            tuning.append(dict(scenario=scenario, direction=direction, attacker=name,
                               C=float(attacker.best_params_["classifier__C"]),
                               gamma=attacker.best_params_["classifier__gamma"],
                               inner_ba=float(attacker.best_score_)))
        elif name == "mlp":
            tuning.append(dict(scenario=scenario, direction=direction, attacker=name,
                               **attacker.convergence_))
    return pd.concat(predictions, ignore_index=True), tuning


def _head_predictions(meta, prediction, scenario, direction):
    frame = meta[META_COLUMNS].copy()
    frame["prediction"] = prediction
    frame["attacker"], frame["scenario"], frame["direction"] = "adversary_head", scenario, direction
    return frame


def _convergence_summary(tuning):
    mlp = tuning[tuning.attacker == "mlp"]
    failed = mlp[mlp.converged == False]
    return dict(mlp_fits=len(mlp), mlp_nonconverged=len(failed),
                warning_counts={str(k): int(v) for k, v in failed.warning.value_counts().items()})


def _identity_scores(predictions):
    """Equal-direction means, not a size-weighted concatenation of E halves."""
    metrics, recalls = [], []
    for (scenario, attacker, direction), group in predictions.groupby(
            ["scenario", "attacker", "direction"], sort=True):
        tags = dict(scenario=scenario, attacker=attacker, direction=int(direction),
                    evidence_role="independent_attack" if attacker in ATTACKERS
                    else "training_adversary_diagnostic")
        metrics.append(dict(**tags,
                            ba=float(balanced_accuracy_score(group.subject, group.prediction)),
                            macro_f1=float(f1_score(group.subject, group.prediction,
                                                   average="macro", zero_division=0))))
        for subject, sub in group.groupby("subject"):
            recalls.append(dict(**tags, subject=int(subject),
                                recall=float((sub.subject == sub.prediction).mean())))
    return pd.DataFrame(metrics), pd.DataFrame(recalls)


def _dev_attacks(z, val, config, seed, head_prediction=None):
    predictions, tuning = [], []
    split = StratifiedKFold(2, shuffle=True, random_state=seed)
    for direction, (train, test) in enumerate(split.split(z, val.subject)):
        pred, tuned = _attack_predictions(
            z[train], val.meta.iloc[train], z[test], val.meta.iloc[test],
            config, seed, "validation", direction)
        predictions.append(pred)
        if head_prediction is not None:
            predictions.append(_head_predictions(val.meta.iloc[test], head_prediction[test],
                                                 "validation", direction))
        tuning.extend(tuned)
    scores, _ = _identity_scores(pd.concat(predictions, ignore_index=True))
    return scores.to_dict("records"), tuning


def _pilot(train, val, family, seed, run, config, artifact_dir):
    torch.set_num_threads(config["torch_threads"])
    with threadpool_limits(limits=1):
        training = config["training"]
        features = FeatureMap(family == "rc").fit(train)
        linear = fit_linear(train, features, training, seed)
        _, X = features.transform(train)
        checkpoints = list(range(training["checkpoint_interval"],
                                 training["max_epochs"] + 1,
                                 training["checkpoint_interval"]))
        if training["max_epochs"] not in checkpoints:
            checkpoints.append(training["max_epochs"])
        states = fit_encoder(train.with_X(X), training, seed,
                             training["max_epochs"], 0.0, checkpoints)
        rows = []
        for epoch, state in states.items():
            model = freeze_encoder(features, state, X.shape[1],
                                   np.unique(train.subject), training)
            rows.append(dict(family=family, seed=seed, fold=run, epoch=epoch,
                             ba=_task_mean(val, model(val)[1])))
        path = artifact_dir / "dev" / f"{family}_seed{seed}_fold{run}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(features=features, linear=linear, states=states), path)
        return rows


def choose_epochs(history):
    """Single family budget: maximize fold/seed mean; ties choose fewer epochs."""
    budgets = {}
    for family, frame in history.groupby("family"):
        means = frame.groupby("epoch").ba.mean()
        budgets[family] = int(means.sort_index().idxmax())
    return budgets


def _dev_fold(train, val, family, seed, run, epochs, config, artifact_dir):
    torch.set_num_threads(config["torch_threads"])
    with threadpool_limits(limits=1):
        packet = joblib.load(artifact_dir / "dev" / f"{family}_seed{seed}_fold{run}.joblib")
        features = packet["features"]
        _, X = features.transform(train)
        models = [(_condition("P0", family), 0.0, packet["linear"]),
                  (_condition("P2", family), 0.0,
                   freeze_encoder(features, packet["states"][epochs], X.shape[1],
                                  np.unique(train.subject), config["training"]))]
        for target in config["lambdas"]:
            state = fit_encoder(train.with_X(X), config["training"], seed, epochs, target)[epochs]
            models.append((_condition("P3", family), float(target),
                           freeze_encoder(features, state, X.shape[1],
                                          np.unique(train.subject), config["training"])))
        task_rows, identity_rows, tuning_rows = [], [], []
        for condition, target, model in models:
            tags = dict(condition=condition, target=target, family=family, seed=seed, fold=run)
            z, logits = model(val)
            task_rows.extend(_task_rows(val, logits, **tags))
            head_prediction = (model.identity_classes_[model.adversary_logits(val).argmax(axis=1)]
                               if condition.startswith("P3") else None)
            scores, tuning = _dev_attacks(z, val, config, seed, head_prediction)
            identity_rows.extend(dict(**tags, **row) for row in scores)
            tuning_rows.extend(dict(**tags, **row) for row in tuning)
            _save_outputs(artifact_dir / "dev" / "oof" / setting_key(condition, target) /
                          f"seed{seed}_fold{run}.npz", val, logits)
        return task_rows, identity_rows, tuning_rows


def choose_operating_points(tasks, identities, config):
    """Choose solely on T CV: eligible task BA, then minimum strongest attack."""
    keys = ["condition", "target", "family"]
    task = tasks.groupby(keys, as_index=False).ba.mean().rename(columns={"ba": "task_ba"})
    independent = identities[identities.attacker.isin(ATTACKERS)]
    attack = independent.groupby(keys + ["attacker"], as_index=False).ba.mean()
    strongest = attack.groupby(keys, as_index=False).ba.max().rename(
        columns={"ba": "identity_ba"})
    summary = task.merge(strongest, on=keys, validate="one_to_one")
    head = identities[identities.attacker == "adversary_head"].groupby(
        keys, as_index=False).ba.mean().rename(columns={"ba": "adversary_ba"})
    summary = summary.merge(head, on=keys, how="left", validate="one_to_one")
    choices = {}
    for family in FAMILIES:
        family_rows = summary[summary.family == family]
        baseline = float(family_rows.loc[
            family_rows.condition == _condition("P2", family), "task_ba"].iloc[0])
        candidates = family_rows[family_rows.condition == _condition("P3", family)]
        eligible = candidates[candidates.task_ba >=
                              baseline - config["statistics"]["utility_margin"] - 1e-12]
        eligible = eligible.sort_values(["identity_ba", "task_ba", "target"],
                                        ascending=[True, False, True])
        choices[family] = dict(
            status="selected" if len(eligible) else "no_operating_point",
            target=float(eligible.iloc[0].target) if len(eligible) else None,
            p2_task_ba=baseline,
            eligible_targets=eligible.target.tolist())
    return choices, summary


def _parallel(config, jobs):
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return Parallel(n_jobs=config["workers"])(jobs)


def run_dev(trials: Trials, config=None, result_dir=RESULTS, artifact_dir=ARTIFACTS):
    """Run all T-only development on covariance Trials, never session-E arrays."""
    require_training(trials)
    config = config or load_config("exp_b")
    result_dir, artifact_dir = Path(result_dir), Path(artifact_dir)
    start = time.perf_counter()
    train = trials.subset(quality_mask(trials))
    folds = list(t_run_folds(train))
    pilot = _parallel(config, (
        delayed(_pilot)(fit, val, family, seed, run, config, artifact_dir)
        for family in FAMILIES for seed in config["seeds"] for run, fit, val in folds))
    history = pd.DataFrame([row for rows in pilot for row in rows])
    budgets = choose_epochs(history)
    result_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(result_dir / "epoch_cv.csv", index=False)
    print(f"P2 family budgets (T-only): {budgets}", flush=True)
    evaluated = _parallel(config, (
        delayed(_dev_fold)(fit, val, family, seed, run, budgets[family], config, artifact_dir)
        for family in FAMILIES for seed in config["seeds"] for run, fit, val in folds))
    tables = [pd.DataFrame([row for bundle in evaluated for row in bundle[i]]) for i in range(3)]
    tasks, identities, tuning = tables
    for name, table in zip(("task_cv", "identity_cv", "attacker_tuning_cv"), tables):
        table.to_csv(result_dir / f"{name}.csv", index=False)
    choices, summary = choose_operating_points(tasks, identities, config)
    summary.to_csv(result_dir / "summary_cv.csv", index=False)
    selected = {}
    for family in FAMILIES:
        for prefix in ("P0", "P2", "P3"):
            condition = _condition(prefix, family)
            target = choices[family]["target"] if prefix == "P3" else 0.0
            selected[condition] = target
            if target is None:
                continue
            for seed in config["seeds"]:
                arrays = [_load_npz(artifact_dir / "dev" / "oof" /
                                    setting_key(condition, target) / f"seed{seed}_fold{run}.npz")
                          for run, _, _ in folds]
                joined = {key: np.concatenate([a[key] for a in arrays]) for key in arrays[0]}
                meta = pd.DataFrame({key: joined[key] for key in META_COLUMNS})
                order = meta.sort_values(["subject", "run", "trial"]).index.to_numpy()
                _save_outputs(artifact_dir / "oof" / f"{condition}_seed{seed}.npz",
                              Trials(np.zeros(len(order)), meta.iloc[order]),
                              joined["logits"][order])
    selection = dict(schema=1, phase="T_only_development", sessions=["T"], config=config,
                     config_hash=_hash(config), source_hash=_source_hash(),
                     training_hash=_training_hash(train), retained_trials=len(train),
                     budgets=budgets, families=choices, selected=selected,
                     attacker_convergence=_convergence_summary(tuning),
                     wall_seconds=time.perf_counter() - start,
                     rules=dict(epoch_tie="earliest", lambda_tie="higher_task_then_lower_lambda",
                                task_weighting="equal_subject_fold_seed",
                                identity_weighting="equal_direction_fold_seed_then_max_attacker"))
    _json(result_dir / "selection.json", selection)
    _json(artifact_dir / "oof" / "manifest.json",
          dict(schema=1, selected=selected, seeds=config["seeds"],
               selection_hash=_hash(selection)))
    print(json.dumps({k: selection[k] for k in (
        "budgets", "families", "wall_seconds", "attacker_convergence")}), flush=True)
    return selection


def _identity_interval(predictions, samples, seed, confidence):
    """Run-within-subject bootstrap, keeping the main directions equally weighted.

    Resampling confusion-count units rather than trial arrays computes exactly
    the same recall/F1 statistics and makes thousands of replicates practical.
    """
    identities = np.sort(predictions.subject.unique())
    lookup = {subject: i for i, subject in enumerate(identities)}
    rng = np.random.default_rng(seed)
    ba, f1 = [], []
    for _, direction in predictions.groupby("direction"):
        confusion = np.zeros((samples, len(identities), len(identities)))
        for subject, group in direction.groupby("subject"):
            units = []
            for _, unit in group.groupby("run"):
                counts = np.zeros(len(identities))
                for predicted, count in unit.prediction.value_counts().items():
                    counts[lookup[predicted]] = count
                units.append(counts)
            units = np.asarray(units)
            sampled = rng.integers(len(units), size=(samples, len(units)))
            confusion[:, lookup[subject], :] = units[sampled].sum(axis=1)
        diagonal = confusion.diagonal(axis1=1, axis2=2)
        true_counts, predicted_counts = confusion.sum(axis=2), confusion.sum(axis=1)
        ba.append((diagonal / true_counts).mean(axis=1))
        f1.append((2 * diagonal / (true_counts + predicted_counts)).mean(axis=1))
    q = [(1 - confidence) / 2, (1 + confidence) / 2]
    ba_lo, ba_hi = np.quantile(np.mean(ba, axis=0), q)
    f1_lo, f1_hi = np.quantile(np.mean(f1, axis=0), q)
    return dict(ba_low=float(ba_lo), ba_high=float(ba_hi),
                macro_f1_low=float(f1_lo), macro_f1_high=float(f1_hi))


def evaluate_attack_scenarios(train_z, train: Trials, test_z, test: Trials, config, seed,
                             head_prediction=None):
    """Evaluate frozen outputs; no defended fit is performed here."""
    if set(train.session) != {"T"} or set(test.session) != {"E"}:
        raise ValueError("Final scenarios require T training outputs and E collection outputs")
    predictions, tuning = [], []
    for direction, (fit_runs, test_runs) in enumerate(E_HALVES):
        fit_idx, test_idx = np.isin(test.run, fit_runs), np.isin(test.run, test_runs)
        pred, tuned = _attack_predictions(
            test_z[fit_idx], test.meta.loc[fit_idx], test_z[test_idx], test.meta.loc[test_idx],
            config, seed, "main", direction)
        predictions.append(pred)
        if head_prediction is not None:
            predictions.append(_head_predictions(test.meta.loc[test_idx], head_prediction[test_idx],
                                                 "main", direction))
        tuning.extend(tuned)
    pred, tuned = _attack_predictions(train_z, train.meta, test_z, test.meta,
                                     config, seed, "auxiliary", 0)
    predictions.append(pred)
    if head_prediction is not None:
        predictions.append(_head_predictions(test.meta, head_prediction, "auxiliary", 0))
    tuning.extend(tuned)
    predictions = pd.concat(predictions, ignore_index=True)
    directional, recalls = _identity_scores(predictions)
    scenario = directional.groupby(["scenario", "attacker", "evidence_role"],
                                   as_index=False)[["ba", "macro_f1"]].mean()
    intervals = []
    for (name, attacker), group in predictions.groupby(["scenario", "attacker"]):
        intervals.append(dict(scenario=name, attacker=attacker, **_identity_interval(
            group, config["statistics"]["bootstrap_samples"], seed,
            config["statistics"]["confidence"])))
    scenario = scenario.merge(pd.DataFrame(intervals), on=["scenario", "attacker"])
    recalls = recalls.groupby(["scenario", "attacker", "evidence_role", "subject"],
                              as_index=False).recall.mean()
    return scenario, directional, recalls, predictions, tuning


def _final_model(all_trials, family, condition, target, seed, selection, config, artifact_dir):
    torch.set_num_threads(config["torch_threads"])
    with threadpool_limits(limits=1):
        train = all_trials.subset((all_trials.session == "T") & quality_mask(all_trials))
        test = all_trials.subset((all_trials.session == "E") & quality_mask(all_trials))
        model = fit_defended(train, condition, config["training"], seed,
                             selection["budgets"][family], target)
        key = setting_key(condition, target)
        folder = artifact_dir / "final" / key
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, folder / f"seed{seed}.joblib")
        z, logits = model(all_trials)
        head_logits = model.adversary_logits(all_trials) if condition.startswith("P3") else None
        _save_outputs(folder / f"seed{seed}_outputs.npz", all_trials, logits, z,
                      head_logits, model.identity_classes_)
        t_idx = (all_trials.session == "T") & quality_mask(all_trials)
        e_idx = (all_trials.session == "E") & quality_mask(all_trials)
        tags = dict(condition=condition, target=target, family=family, seed=seed)
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


def _winner(recalls, condition, target, main_only=False):
    subset = recalls[(recalls.condition == condition) & (recalls.target == target)
                     & recalls.attacker.isin(ATTACKERS)]
    if main_only:
        subset = subset[subset.scenario == "main"]
    aggregate = subset.groupby(["scenario", "attacker"]).recall.mean().sort_values(ascending=False)
    scenario, attacker = aggregate.index[0]
    chosen = subset[(subset.scenario == scenario) & (subset.attacker == attacker)]
    per_subject = chosen.groupby("subject").recall.mean().sort_index()
    return per_subject, dict(scenario=scenario, attacker=attacker,
                             ba=float(per_subject.mean())), chosen


def _paired_report(difference):
    return dict(mean=float(difference.mean()),
                lower_95=paired_t_lower_bound(difference),
                sign_flip_pvalue=sign_flip_pvalue(difference),
                per_subject={str(k): float(v) for k, v in difference.items()})


def make_decision(tasks, recalls, selection, config):
    """Plan tests use seed-mean per-subject pairs, never seed pseudoreplication."""
    rules = config["statistics"]
    _, p0, _ = _winner(recalls, "P0", 0.0)
    at_chance = p0["ba"] <= rules["chance"]
    families = {}
    for family in FAMILIES:
        choice = selection["families"][family]
        if choice["target"] is None:
            families[family] = dict(status="no_operating_point", accepted=False)
            continue
        target = choice["target"]
        baseline, defended = _condition("P2", family), _condition("P3", family)
        btask = tasks[(tasks.condition == baseline) & (tasks.target == 0)].groupby("subject").ba.mean()
        dtask = tasks[(tasks.condition == defended) & (tasks.target == target)].groupby("subject").ba.mean()
        utility = _paired_report(dtask - btask)
        utility["passed"] = utility["lower_95"] > -rules["utility_margin"]
        protection = {}
        for label, main_only in (("strongest_both_scenarios", False), ("main", True)):
            brecall, bwinner, _ = _winner(recalls, baseline, 0.0, main_only)
            drecall, dwinner, _ = _winner(recalls, defended, target, main_only)
            report = _paired_report(brecall - drecall)
            report.update(baseline=bwinner, defended=dwinner)
            report["passed"] = (report["mean"] >= rules["protection_drop"] - 1e-12
                                and report["lower_95"] > 0)
            protection[label] = report
        accepted = (utility["passed"] and not at_chance
                    and all(value["passed"] for value in protection.values()))
        families[family] = dict(target=target, utility=utility, protection=protection,
                                accepted=bool(accepted),
                                status="withheld_baseline_at_chance" if at_chance
                                else ("accepted" if accepted else "not_accepted"))
    return dict(
        schema=1, baseline_p0=p0, baseline_at_chance=bool(at_chance), families=families,
        h2_accepted=bool(all(f["accepted"] for f in families.values())),
        interpretation="Both families must pass utility and protection. Auxiliary-only improvement "
                       "cannot establish protection: the main scenario must also pass.",
        selection_hash=_hash(selection))


def _tradeoff(tasks, recalls, selection, config, result_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    rows = []
    rng = np.random.default_rng(0)
    samples = config["statistics"]["bootstrap_samples"]
    for family in FAMILIES:
        color = "tab:blue" if family == "common" else "tab:orange"
        curve = []
        for fam, condition, target in settings(config):
            if fam != family:
                continue
            task = tasks[(tasks.condition == condition) & (tasks.target == target)]
            _, winner, identity = _winner(recalls, condition, target)
            a = task.pivot(index="seed", columns="subject", values="ba").sort_index()
            b = identity.pivot(index="seed", columns="subject", values="recall").reindex_like(a)
            seed_idx = rng.integers(len(a), size=(samples, len(a), 1))
            subject_idx = rng.integers(a.shape[1], size=(samples, 1, a.shape[1]))
            values = np.stack([a.to_numpy()[seed_idx, subject_idx].mean(axis=(1, 2)),
                               b.to_numpy()[seed_idx, subject_idx].mean(axis=(1, 2))])
            q = [(1 - config["statistics"]["confidence"]) / 2,
                 (1 + config["statistics"]["confidence"]) / 2]
            low, high = np.quantile(values, q, axis=1)
            x, y = float(a.to_numpy().mean()), winner["ba"]
            selected = condition.startswith("P3") and selection["families"][family]["target"] == target
            marker = "*" if selected else ("o" if condition.startswith("P3")
                                           else ("s" if condition.startswith("P0") else "^"))
            ax.errorbar(x, y, xerr=[[max(0, x - low[0])], [max(0, high[0] - x)]],
                        yerr=[[max(0, y - low[1])], [max(0, high[1] - y)]],
                        fmt=marker, color=color, alpha=0.8, capsize=2,
                        markersize=13 if selected else 6)
            label = f"{target:g}" if condition.startswith("P3") else condition
            ax.annotate(label, (x, y), xytext=(4, 5), textcoords="offset points", fontsize=8)
            if condition.startswith("P3"):
                curve.append((target, x, y))
            rows.append(dict(condition=condition, target=target, selected=selected,
                             task_ba=x, identity_ba=y, task_low=low[0], task_high=high[0],
                             identity_low=low[1], identity_high=high[1], **{
                                 "strongest_attacker": winner["attacker"],
                                 "strongest_scenario": winner["scenario"]}))
        curve.sort()
        ax.plot([p[1] for p in curve], [p[2] for p in curve], color=color,
                linewidth=1, label=f"P3{'-RC' if family == 'rc' else ''} lambda curve")
    ax.axhline(config["statistics"]["chance"], color="grey", linestyle="--", label="Identity chance")
    ax.set(xlabel="E task balanced accuracy (higher is better)",
           ylabel="Strongest-attacker identity BA (lower is better)",
           title="Experiment B: frozen-model utility / identity exposure", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="best", fontsize=8)
    fig.text(0.5, 0.01, "95% subject/seed bootstrap intervals; stars: T-selected lambda. "
             "Identity intervals condition on the strongest attacker/scenario.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(result_dir / "utility_exposure.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(result_dir / "tradeoff.csv", index=False)


def _adversary_comparison(scores, recalls, selection, config, result_dir):
    """Contrast the co-trained diagnostic head with independent attack evidence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    rows = []
    for family, ax in zip(FAMILIES, axes):
        condition = _condition("P3", family)
        family_rows = []
        for target in config["lambdas"]:
            head = scores[(scores.condition == condition) & (scores.target == target)
                          & (scores.attacker == "adversary_head")].groupby("scenario").ba.mean()
            _, winner, _ = _winner(recalls, condition, target)
            row = dict(condition=condition, target=float(target),
                       adversary_main_ba=float(head["main"]),
                       adversary_auxiliary_ba=float(head["auxiliary"]),
                       independent_ba=winner["ba"], strongest_attacker=winner["attacker"],
                       strongest_scenario=winner["scenario"],
                       selected=selection["families"][family]["target"] == target)
            rows.append(row)
            family_rows.append(row)
        table = pd.DataFrame(family_rows).sort_values("target")
        ax.plot(table.target, table.adversary_main_ba, "o--", label="Adversary head: main")
        ax.plot(table.target, table.adversary_auxiliary_ba, ".:", label="Adversary head: T-to-E")
        ax.plot(table.target, table.independent_ba, "s-", label="Strongest independent: both scenarios")
        chosen = table[table.selected]
        ax.scatter(chosen.target, chosen.independent_ba, marker="*", s=180,
                   color="black", label="T-selected lambda", zorder=5)
        ax.axhline(config["statistics"]["chance"], color="grey", linestyle="--", label="Chance (1/9)")
        ax.set(xscale="log", xlabel="Lambda", title=condition, ylim=(0, 1))
        ax.legend(fontsize=7, loc="best")
    axes[0].set_ylabel("Identity balanced accuracy")
    fig.suptitle("Frozen adversary head vs independent identity attacks")
    fig.text(0.5, 0.01, "Seed means; main averages both E-half directions. "
             "The co-trained head is diagnostic only, never protection evidence.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(result_dir / "adversary_vs_independent.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(result_dir / "adversary_comparison.csv", index=False)


def run_final(trials: Trials, selection: dict, config=None, result_dir=RESULTS,
              artifact_dir=ARTIFACTS):
    """Retrain/freeze on retained T; evaluate E. Use synthetic data before seal."""
    config = config or load_config("exp_b")
    if set(trials.session) != {"T", "E"}:
        raise ValueError("Final requires both T and E covariance Trials")
    if selection["config_hash"] != _hash(config) or selection["source_hash"] != _source_hash():
        raise ValueError("Selection config/source differs; rerun T-only dev before sealing")
    train = trials.subset((trials.session == "T") & quality_mask(trials))
    if selection["training_hash"] != _training_hash(train):
        raise ValueError("Final retained T differs from development")
    result_dir, artifact_dir = Path(result_dir), Path(artifact_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    evaluated = _parallel(config, (
        delayed(_final_model)(trials, family, condition, target, seed, selection, config, artifact_dir)
        for family, condition, target in settings(config) for seed in config["seeds"]))
    tables = [pd.DataFrame([row for bundle in evaluated for row in bundle[i]]) for i in range(5)]
    for name, table in zip(("task_final", "identity_final", "identity_directional",
                            "identity_recall", "attacker_tuning_final"), tables):
        table.to_csv(result_dir / f"{name}.csv", index=False)
    tasks, _, _, recalls, _ = tables
    subject_means = tasks.groupby(["condition", "target", "subject"], as_index=False)[["ba", "kappa"]].mean()
    subject_means.to_csv(result_dir / "task_subject_means.csv", index=False)
    subject_means.groupby(["condition", "target"])[["ba", "kappa"]].agg(
        ["mean", "median", "std"]).to_csv(result_dir / "task_summary.csv")
    decision = make_decision(tasks, recalls, selection, config)
    _json(result_dir / "decision.json", decision)
    _tradeoff(tasks, recalls, selection, config, result_dir)
    _adversary_comparison(tables[1], recalls, selection, config, result_dir)
    manifest = dict(schema=1, selected=selection["selected"], seeds=config["seeds"],
                    selection_hash=_hash(selection), config_hash=_hash(config),
                    source_hash=_source_hash(), wall_seconds=time.perf_counter() - start,
                    attacker_convergence=_convergence_summary(tables[4]),
                    all_trials=len(trials), retained_T=len(train),
                    retained_E=int(((trials.session == "E") & quality_mask(trials)).sum()),
                    models=[dict(condition=c, target=t, key=setting_key(c, t))
                            for _, c, t in settings(config)])
    _json(artifact_dir / "final" / "manifest.json", manifest)
    print(json.dumps(dict(h2_accepted=decision["h2_accepted"],
                          attacker_convergence=manifest["attacker_convergence"],
                          wall_seconds=manifest["wall_seconds"])), flush=True)
    return decision


def _final_path(condition, seed, target, artifact_dir):
    artifact_dir = Path(artifact_dir)
    manifest = json.loads((artifact_dir / "final" / "manifest.json").read_text(encoding="utf-8"))
    if seed not in manifest["seeds"] or condition not in manifest["selected"]:
        raise ValueError("Condition/seed absent from final manifest")
    if target is None:
        target = manifest["selected"][condition]
    if target is None:
        raise ValueError(f"{condition}: no T-selected operating point")
    key = setting_key(condition, target)
    if not any(m["condition"] == condition and m["target"] == target for m in manifest["models"]):
        raise ValueError("Lambda absent from final manifest")
    return artifact_dir / "final" / key / f"seed{seed}"


def load_frozen(condition: str, seed: int, target=None, artifact_dir=ARTIFACTS):
    """Load an inference-only final model from trusted local joblib artifacts."""
    return joblib.load(_final_path(condition, seed, target, artifact_dir).with_suffix(".joblib"))


def load_outputs(condition: str, seed: int, target=None, artifact_dir=ARTIFACTS):
    """Return all chronological T/E outputs and metadata, including flagged rows."""
    path = _final_path(condition, seed, target, artifact_dir)
    return _load_npz(path.parent / f"{path.name}_outputs.npz")


def load_oof(condition: str, seed: int, artifact_dir=ARTIFACTS):
    """Return selected-condition T out-of-fold task logits for temperature fit."""
    folder = Path(artifact_dir) / "oof"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if seed not in manifest["seeds"] or condition not in manifest["selected"]:
        raise ValueError("Condition/seed absent from OOF manifest")
    if manifest["selected"][condition] is None:
        raise ValueError(f"{condition}: no T-selected operating point")
    return _load_npz(folder / f"{condition}_seed{seed}.npz")


def _load_covariances(development: bool):
    cfg = load_config("data")
    if development:
        # Avoid even opening E raw files/cache during development.
        cfg = dict(cfg, sessions=["T"])
    signals = load_trials(cfg)
    valid = np.isfinite(signals.X).all(axis=(1, 2))
    X = np.full((len(signals), signals.X.shape[1], signals.X.shape[1]), np.nan)
    X[valid] = covariances(signals.subset(valid)).X
    return signals.with_X(X)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["dev", "final"])
    args = parser.parse_args(argv)
    config = load_config("exp_b")
    if args.phase == "dev":
        start = time.perf_counter()
        selection = run_dev(_load_covariances(True), config)
        selection["command_wall_seconds"] = time.perf_counter() - start
        _json(RESULTS / "selection.json", selection)
        _json(ARTIFACTS / "oof" / "manifest.json",
              dict(schema=1, selected=selection["selected"], seeds=config["seeds"],
                   selection_hash=_hash(selection)))
        print(f"Full dev wall time: {selection['command_wall_seconds']:.1f}s", flush=True)
    else:
        selection = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
        run_final(_load_covariances(False), selection, config)


if __name__ == "__main__":
    main()
