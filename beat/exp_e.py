"""Exploratory deep-backbone identifier audit. Run dev before committing seal 2.

Hypothesis: reversing a co-trained head's gradient can fool that head without
removing independently decodable identity. Frozen-feature attackers refute
protection; a low training-head score alone cannot establish it.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from threadpoolctl import threadpool_limits
from torch import nn

from .attacks import fit_attackers
from .config import ROOT, load_config
from .data import Trials, load_trials, quality_mask, require_training
from .exp_b import _task_rows, evaluate_attack_scenarios
from .models import ramp, reverse_gradient
from .splits import E_HALVES
from .stats import paired_t_lower_bound, sign_flip_pvalue, unit_bootstrap_ba

RESULTS = ROOT / "results" / "exp_e"
ARTIFACTS = ROOT / "artifacts" / "exp_e"
CONFIG = ROOT / "configs" / "exp_e.yaml"


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _source_hash():
    paths = ("exp_e.py", "exp_b.py", "data.py", "models.py", "attacks.py", "stats.py", "splits.py")
    return hashlib.sha256(b"".join((ROOT / "beat" / p).read_bytes() for p in paths)).hexdigest()


def _training_hash(train):
    require_training(train)
    h = hashlib.sha256(np.ascontiguousarray(train.X).tobytes())
    h.update(train.meta.to_csv(index=False).encode())
    return h.hexdigest()


def prepare(trials, config):
    """No fitted preprocessing: retain the same A/B trials and decimate by two."""
    kept = trials.subset(quality_mask(trials))
    expected = (config["input"]["channels"],
                config["input"]["samples"] * config["input"]["decimation"])
    if kept.X.shape[1:] != expected:
        raise ValueError(f"Expected filtered input of shape (n, {expected})")
    X = kept.X[:, :, ::config["input"]["decimation"]].astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Retained input contains non-finite samples")
    return kept.with_X(X)


class Alignment:
    """Subject arithmetic-mean covariance references, never refitted at inference."""
    def __init__(self, condition, relative_floor=1e-10):
        if condition not in ("none", "ea"):
            raise ValueError(f"Unknown alignment {condition}")
        self.condition = condition
        self.relative_floor = relative_floor
        self.references = {}

    def fit(self, train):
        require_training(train)
        self.references = {}
        if self.condition == "ea":
            for subject in np.unique(train.subject):
                X = train.X[train.subject == subject].astype(np.float64)
                X = X - X.mean(axis=2, keepdims=True)
                reference = np.mean(X @ X.transpose(0, 2, 1) / (X.shape[2] - 1), axis=0)
                values, vectors = np.linalg.eigh(reference)
                if values.max() <= 0:
                    raise ValueError("Euclidean alignment requires nonzero covariance")
                values = np.maximum(values, self.relative_floor * values.max())
                self.references[int(subject)] = (vectors * values**-0.5) @ vectors.T
        return self

    def transform(self, trials):
        if self.condition == "none":
            return trials.with_X(np.asarray(trials.X, dtype=np.float32))
        X = np.empty_like(trials.X, dtype=np.float32)
        for subject in np.unique(trials.subject):
            idx = trials.subject == subject
            X[idx] = self.references[int(subject)] @ trials.X[idx]
        return trials.with_X(X)


def _same_padding(kernel):
    return nn.ZeroPad2d(((kernel - 1) // 2, kernel // 2, 0, 0))


class EEGNet(nn.Module):
    """Lawhern EEGNet-8,2; logits instead of softmax for cross entropy."""
    def __init__(self, config, identities):
        super().__init__()
        a = config["architecture"]
        f1, d, f2 = a["temporal_filters"], a["depth_multiplier"], a["pointwise_filters"]
        channels, samples = config["input"]["channels"], config["input"]["samples"]
        pool1, pool2 = a["pooling"]
        self.spatial = nn.Conv2d(f1, f1 * d, (channels, 1), groups=f1, bias=False)
        def bn(n):
            return nn.BatchNorm2d(n, eps=a["batchnorm_epsilon"], momentum=a["batchnorm_momentum"])
        self.encoder = nn.Sequential(
            _same_padding(a["temporal_kernel"]),
            nn.Conv2d(1, f1, (1, a["temporal_kernel"]), bias=False),
            bn(f1), self.spatial, bn(f1 * d), nn.ELU(),
            nn.AvgPool2d((1, pool1)), nn.Dropout(a["dropout"]),
            _same_padding(a["separable_kernel"]),
            nn.Conv2d(f1 * d, f1 * d, (1, a["separable_kernel"]), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, 1, bias=False), bn(f2), nn.ELU(),
            nn.AvgPool2d((1, pool2)), nn.Dropout(a["dropout"]), nn.Flatten())
        dimension = f2 * (samples // pool1 // pool2)
        self.task = nn.Linear(dimension, 2)
        self.identifier = nn.Sequential(nn.Linear(dimension, a["identifier_hidden"]), nn.ReLU(),
                                        nn.Linear(a["identifier_hidden"], identities))
        self.spatial_limit, self.task_limit = a["spatial_max_norm"], a["task_max_norm"]
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, X, strength=0.0):
        z = self.encoder(X[:, None])
        return z, self.task(z), self.identifier(reverse_gradient(z, strength))

    @torch.no_grad()
    def constrain(self):
        for weight, maximum in ((self.spatial.weight, self.spatial_limit),
                                (self.task.weight, self.task_limit)):
            norms = weight.flatten(1).norm(dim=1).clamp_min(1e-12)
            shape = (-1,) + (1,) * (weight.ndim - 1)
            weight.mul_((maximum / norms).clamp(max=1).reshape(shape))


@torch.inference_mode()
def infer(model, trials, batch_size):
    model.eval()
    batches = [model(torch.from_numpy(trials.X[start:start + batch_size]))
               for start in range(0, len(trials), batch_size)]
    return tuple(torch.cat([batch[i] for batch in batches]).numpy() for i in range(3))


def fit_network(train, config, seed, epochs, target, validation=None, checkpoints=()):
    """Always train the identifier, including lambda=0 (zero feature gradient)."""
    require_training(train)
    if validation is not None:
        require_training(validation)
    if epochs < 1:
        raise ValueError("Training budget must be positive")
    torch.set_num_threads(config["torch_threads"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    subjects, identity = np.unique(train.subject, return_inverse=True)
    model = EEGNet(config, len(subjects))
    c = config["training"]
    if c["optimizer"] != "adam":
        raise ValueError("Only the fixed Adam optimizer is supported")
    optimizer = torch.optim.Adam(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
    X = torch.from_numpy(np.ascontiguousarray(train.X))
    y, identity = torch.tensor(train.y, dtype=torch.long), torch.tensor(identity, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    steps = (len(train) + c["batch_size"] - 1) // c["batch_size"]
    history = []
    start_time = time.perf_counter()
    with threadpool_limits(limits=1):
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(train), generator=generator)
            for batch, start in enumerate(range(0, len(train), c["batch_size"])):
                idx = order[start:start + c["batch_size"]]
                progress = ((epoch - 1) * steps + batch) / max(epochs * steps - 1, 1)
                _, task, head = model(X[idx], ramp(target, progress))
                loss = nn.functional.cross_entropy(task, y[idx]) + nn.functional.cross_entropy(head, identity[idx])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite EEGNet loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                model.constrain()
            if validation is not None and epoch in checkpoints:
                _, logits, _ = infer(model, validation, c["batch_size"])
                history.append(dict(epoch=epoch, ba=float(np.mean(
                    [row["ba"] for row in _task_rows(validation, logits)]))))
    seconds = time.perf_counter() - start_time
    model.eval().requires_grad_(False)
    return model, subjects, history, seconds


def fit_defended(train, config, alignment, seed, epochs, target):
    require_training(train)
    reference = Alignment(alignment, config["alignment"]["eigenvalue_relative_floor"]).fit(train)
    model, subjects, _, seconds = fit_network(reference.transform(train), config, seed, epochs, target)
    return reference, model, subjects, seconds


def _parallel(config, jobs):
    workers = config["workers"]
    if not 1 <= workers <= 4 or config["torch_threads"] != 1:
        raise ValueError("Use 1-4 CPU workers with one Torch/BLAS thread each")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return Parallel(n_jobs=workers)(jobs)


def _pilot_fold(train, config, alignment, seed, run):
    require_training(train)
    fit, val = train.subset(train.run != run), train.subset(train.run == run)
    reference = Alignment(alignment, config["alignment"]["eigenvalue_relative_floor"]).fit(fit)
    candidates = config["pilot"]["epoch_candidates"]
    _, _, history, seconds = fit_network(reference.transform(fit), config, seed, max(candidates), 0,
                                         reference.transform(val), candidates)
    return [dict(alignment=alignment, seed=seed, run=run, seconds=seconds, **row) for row in history]


def _benchmark(train, config, alignment, attackers):
    """T-only timing proxy: both T halves and all T; no synthetic E labels."""
    require_training(train)
    start = time.perf_counter()
    reference, model, _, seconds = fit_defended(
        train, config, alignment, config["seeds"][0], config["training"]["budget_epochs"][alignment], 0)
    with threadpool_limits(limits=1):
        z, _, _ = infer(model, reference.transform(train), config["training"]["batch_size"])
        # Main fits have about half-session size; auxiliary uses all T.
        subsets = [np.isin(train.run, halves[0]) for halves in E_HALVES] + [np.ones(len(train), dtype=bool)]
        attack_seconds, convergence = [], []
        for mask in subsets:
            tick = time.perf_counter()
            fitted = fit_attackers(z[mask], train.subject[mask], attackers, config["seeds"][0])
            attack_seconds.append(time.perf_counter() - tick)
            convergence.append(fitted["mlp"].convergence_)
    return dict(alignment=alignment, training_seconds=seconds, attack_fit_seconds=attack_seconds,
                measured_model_seconds=time.perf_counter() - start, mlp_convergence=convergence)


def run_dev(trials, config=None, result_dir=RESULTS, artifact_dir=ARTIFACTS, config_path=None,
            attackers=None):
    """Select only epochs, using six T-run folds; persist a sealable selection."""
    require_training(trials)
    config = copy.deepcopy(config or load_config("exp_e"))
    attackers = attackers if attackers is not None else load_config("exp_b")["attackers"]
    train = prepare(trials, config)
    require_training(train)
    result_dir, artifact_dir = Path(result_dir), Path(artifact_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    bundles = _parallel(config, (delayed(_pilot_fold)(train, config, alignment, seed, run)
                                for alignment in config["alignments"]
                                for seed in config["pilot"]["seeds"]
                                for run in config["pilot"]["held_out_runs"]))
    history = pd.DataFrame([row for bundle in bundles for row in bundle])
    history.to_csv(result_dir / "pilot_history.csv", index=False)
    means = history.groupby(["alignment", "epoch"], as_index=False).ba.mean()
    for alignment in config["alignments"]:
        ranked = means[means.alignment == alignment].sort_values(["ba", "epoch"], ascending=[False, True])
        config["training"]["budget_epochs"][alignment] = int(ranked.iloc[0].epoch)
    benchmarks = _parallel(config, (delayed(_benchmark)(train, config, a, attackers)
                                    for a in config["alignments"]))
    model_count = len(config["alignments"]) * len(config["lambdas"]) * len(config["seeds"])
    total_seconds = sum(b["measured_model_seconds"] for b in benchmarks) * len(config["lambdas"]) * len(config["seeds"])
    # An explicitly labelled allowance, not an empirical upper confidence bound.
    projected = 1.25 * total_seconds / config["workers"]
    selection = dict(schema=1, budgets=config["training"]["budget_epochs"], config_hash=_hash(config),
                     attackers_hash=_hash(attackers), source_hash=_source_hash(),
                     training_hash=_training_hash(train), retained_T=len(train), session_scope=["T"],
                     model_count=model_count, benchmarks=benchmarks, workers=config["workers"],
                     projected_final_hours=projected / 3600, projection_allowance=1.25,
                     projection_caveat="T-feature attack fit proxy; E distributions, convergence and CPU contention can change runtime",
                     exceeds_four_hours=projected > 4 * 3600, wall_seconds=time.perf_counter() - start)
    if config_path is not None:
        Path(config_path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _json(result_dir / "dev_selection.json", selection)
    _json(artifact_dir / "dev_config.json", config)
    print(json.dumps(selection), flush=True)
    return selection


def _final_model(train, test, config, attackers, alignment, target, seed, artifact_dir):
    with threadpool_limits(limits=1):
        reference, model, subjects, seconds = fit_defended(
            train, config, alignment, seed, config["training"]["budget_epochs"][alignment], target)
        zt, lt, ht = infer(model, reference.transform(train), config["training"]["batch_size"])
        ze, le, he = infer(model, reference.transform(test), config["training"]["batch_size"])
        key = f"{alignment}/lambda{target:g}/seed{seed}"
        folder = Path(artifact_dir) / "final" / key
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state=model.state_dict(), subjects=subjects.tolist(), config=config,
                        alignment=alignment, references={s: torch.tensor(v) for s, v in reference.references.items()}),
                   folder / "model.pt")
        np.savez_compressed(folder / "outputs.npz", T_feature=zt, E_feature=ze, T_task=lt, E_task=le,
                            T_identifier=ht, E_identifier=he, identity_classes=subjects)
        train.meta.to_csv(folder / "T_meta.csv", index=False)
        test.meta.to_csv(folder / "E_meta.csv", index=False)
        tags = dict(alignment=alignment, target=target, seed=seed)
        tasks = _task_rows(test, le, **tags)
        pred = subjects[he.argmax(axis=1)]
        lo, hi = unit_bootstrap_ba(test.subject, pred, test.subject, test.run,
                                   config["statistics"]["bootstrap_samples"], seed,
                                   config["statistics"]["confidence"])
        head = dict(**tags, accuracy=float(accuracy_score(test.subject, pred)),
                    ba=float(balanced_accuracy_score(test.subject, pred)), ba_low=lo, ba_high=hi,
                    training_seconds=seconds)
        scores, directional, recalls, predictions, tuning = evaluate_attack_scenarios(
            zt, train, ze, test, dict(attackers=attackers, statistics=config["statistics"]), seed)
        predictions.to_csv(folder / "attack_predictions.csv.gz", index=False)
        tables = [scores, directional, recalls, pd.DataFrame(tuning)]
        for table in tables:
            for k, v in tags.items():
                table[k] = v
        return tasks, [head], *(table.to_dict("records") for table in tables)


def _paired(difference):
    return dict(mean=float(np.mean(difference)), lower_95=paired_t_lower_bound(difference),
                sign_flip_p=sign_flip_pvalue(difference))


def summarize(tasks, heads, scores, recalls, config, result_dir):
    """Seed-average per subject; maximum over attacker/scenario, never the head."""
    subjects = tasks.groupby(["alignment", "target", "subject"], as_index=False)[["ba", "kappa"]].mean()
    subjects.to_csv(result_dir / "task_subject_means.csv", index=False)
    subject_summary = subjects.groupby(["alignment", "target"])[["ba", "kappa"]].agg(["mean", "median", "std"])
    subject_summary.to_csv(result_dir / "task_summary.csv")
    attacks = scores.groupby(["alignment", "target", "scenario", "attacker"], as_index=False)[["ba", "macro_f1"]].mean()
    attacks.to_csv(result_dir / "identity_seed_means.csv", index=False)
    attack_recalls = recalls.groupby(["alignment", "target", "scenario", "attacker", "subject"]).recall.mean()
    rows, paired = [], []
    for alignment in config["alignments"]:
        base_task = subjects[(subjects.alignment == alignment) & (subjects.target == 0)].set_index("subject").ba
        baseline = attacks[(attacks.alignment == alignment) & (attacks.target == 0)].sort_values(
            ["ba", "scenario", "attacker"], ascending=[False, True, True]).iloc[0]
        base_recall = attack_recalls.loc[(alignment, 0, baseline.scenario, baseline.attacker)]
        for target in config["lambdas"]:
            task = subjects[(subjects.alignment == alignment) & (subjects.target == target)].set_index("subject")
            head = heads[(heads.alignment == alignment) & (heads.target == target)]
            subset = attacks[(attacks.alignment == alignment) & (attacks.target == target)]
            winner = subset.sort_values(["ba", "scenario", "attacker"], ascending=[False, True, True]).iloc[0]
            rows.append(dict(alignment=alignment, target=target, task_ba=float(task.ba.mean()),
                             task_kappa=float(task.kappa.mean()), identifier_accuracy=float(head.accuracy.mean()),
                             identifier_ba=float(head.ba.mean()), strongest_ba=float(winner.ba),
                             strongest_attacker=winner.attacker, strongest_scenario=winner.scenario,
                             main_strongest_ba=float(subset[subset.scenario == "main"].ba.max()),
                             auxiliary_strongest_ba=float(subset[subset.scenario == "auxiliary"].ba.max())))
            if target:
                current_recall = attack_recalls.loc[(alignment, target, winner.scenario, winner.attacker)]
                utility, protection = _paired(task.ba - base_task), _paired(base_recall - current_recall)
                paired.append(dict(alignment=alignment, target=target, utility=utility, protection=protection,
                                   utility_noninferior=utility["lower_95"] > -config["statistics"]["utility_margin"],
                                   protection_pass=(protection["mean"] >= config["statistics"]["protection_drop"]
                                                    and protection["lower_95"] > 0),
                                   withheld_at_chance=bool(baseline.ba <= config["statistics"]["chance"])))
    table = pd.DataFrame(rows)
    table.to_csv(result_dir / "lambda_summary.csv", index=False)
    for alignment, frame in table.groupby("alignment"):
        frame = frame.sort_values("target")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(frame.target, frame.identifier_ba, "o-", label="Co-trained identifier BA")
        ax.plot(frame.target, frame.strongest_ba, "s-", label="Strongest independent identity BA")
        ax.plot(frame.target, frame.task_ba, "^-", label="Task BA (subject mean)")
        ax.axhline(config["statistics"]["chance"], ls=":", color="black", label="Identity chance (1/9)")
        ax.set(xlabel="Target gradient-reversal lambda", ylabel="Balanced accuracy", ylim=(0, 1.02),
               title=f"Exploratory EEGNet-8,2: {alignment}")
        ax.set_xscale("symlog", linthresh=0.03)
        ax.set_xticks(config["lambdas"], [f"{x:g}" for x in config["lambdas"]])
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(result_dir / f"identifier_vs_independent_{alignment}.png", dpi=160)
        plt.close(fig)
    return dict(per_lambda=rows, paired_vs_lambda_zero=paired,
                interpretation="Exploratory reuse of session E; all lambdas reported; no E selection; not an ID-RemovalNet reproduction",
                strongest_definition="Maximum seed-mean identity BA across three independent attackers and main/auxiliary scenarios")


def run_final(trials, selection, config=None, result_dir=RESULTS, artifact_dir=ARTIFACTS, attackers=None):
    """Only call on real data after seal 2; synthetic end-to-end tests are allowed."""
    config = config or load_config("exp_e")
    attackers = attackers if attackers is not None else load_config("exp_b")["attackers"]
    if set(trials.session) != {"T", "E"}:
        raise ValueError("Final requires T and E")
    if (selection["config_hash"] != _hash(config) or selection["source_hash"] != _source_hash()
            or selection["attackers_hash"] != _hash(attackers)):
        raise ValueError("Code/config/attackers changed since dev; rerun T-only dev before seal 2")
    kept = prepare(trials, config)
    train, test = kept.subset(kept.session == "T"), kept.subset(kept.session == "E")
    if selection["training_hash"] != _training_hash(train):
        raise ValueError("Retained T differs from pilot")
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    bundles = _parallel(config, (delayed(_final_model)(train, test, config, attackers, alignment, target, seed, artifact_dir)
                                for alignment in config["alignments"] for target in config["lambdas"]
                                for seed in config["seeds"]))
    tables = [pd.DataFrame([row for bundle in bundles for row in bundle[i]]) for i in range(6)]
    for name, table in zip(("task_final", "identifier_final", "identity_final", "identity_directional",
                            "identity_recall", "attacker_tuning_final"), tables):
        table.to_csv(result_dir / f"{name}.csv", index=False)
    summary = summarize(*tables[:3], tables[4], config, result_dir)
    tuning = tables[5]
    mlp = tuning[tuning.attacker == "mlp"]
    summary.update(model_count=len(bundles), retained_T=len(train), retained_E=len(test),
                   wall_seconds=time.perf_counter() - tick, selection=selection,
                   mlp_fits=len(mlp), mlp_nonconverged=int((mlp.converged == False).sum()))
    _json(result_dir / "summary.json", summary)
    print(json.dumps(dict(models=len(bundles), wall_seconds=summary["wall_seconds"],
                          mlp_nonconverged=summary["mlp_nonconverged"])), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("dev", "final"))
    args = parser.parse_args(argv)
    data_config = load_config("data")
    # Even loader caches stay inside the assigned write scope.
    data_config["processed_dir"] = "artifacts/exp_e/preprocessed"
    if args.stage == "dev":
        data_config["sessions"] = ["T"]
        run_dev(load_trials(data_config), config_path=CONFIG)
    else:
        selection = json.loads((RESULTS / "dev_selection.json").read_text(encoding="utf-8"))
        run_final(load_trials(data_config), selection)


if __name__ == "__main__":
    main()
