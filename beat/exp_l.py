"""Confirmatory Lee2019 replication. ``dev`` is strictly session-1-only.

``final --seal-commit HASH`` verifies the committed inputs before opening E.
Synthetic integration calls run_final with in-memory Trials, never real E files.
All fitted defended components are imported T-guarded implementations.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed, parallel_config
from pyriemann.tangentspace import TangentSpace
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from multiprocessing import TimeoutError as WorkerTimeout

from .config import ROOT, load_config
from .data import LeakageError, Trials, require_training
from .exp_b import choose_epochs, _task_rows
from .exp_c import (command_descriptors, control_pass, decide, episode_layout,
                    feature_descriptors, fit_temperature, identify_head,
                    observation_groups, score_episodes)
from .exp_g import (FourStageDefense, first_labelled_trials, paired_recall_test,
                    THREAT_MODEL_INTERPRETATION)
from .features import covariances
from .lee2019 import load_trials
from .models import (FeatureMap, fit_defended, fit_encoder, fit_linear,
                     freeze_encoder)
from .stats import paired_t_lower_bound

RESULTS = ROOT / "results/exp_l"
ARTIFACTS = ROOT / "artifacts/exp_l"
TAGS = ["kind", "key", "mode", "tau", "k", "enrolment", "enrolled"]


class PredictionStore:
    """Disk partitions bound memory to one setting across seeds, not the full grid."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.entries = {}
        self.counter = 0

    def append(self, frame):
        for tag, part in frame.groupby(TAGS, sort=True, dropna=False):
            path = self.directory / f"{self.counter:06d}.joblib"
            joblib.dump(part, path, compress=1)
            self.entries.setdefault(tag, []).append(path)
            self.counter += 1

    def extend(self, frames):
        for frame in frames:
            self.append(frame)

    def groups(self, **query):
        for tag, paths in self.entries.items():
            tags = dict(zip(TAGS, tag))
            if all(tags[k] == v for k, v in query.items()):
                yield tags, pd.concat([joblib.load(p) for p in paths], ignore_index=True)

    def select(self, **query):
        frames = [frame for _, frame in self.groups(**query)]
        return (pd.concat(frames, ignore_index=True) if frames else
                pd.DataFrame(columns=TAGS + ["subject", "run", "trial", "prediction",
                                              "seed", "attacker", "scenario", "direction"]))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def analysis_config(cfg):
    # Approval changes no numerical selection; it is independently checked by final.
    return {k: v for k, v in cfg.items() if k != "attacker_status"}


def development_config(cfg):
    return {k: cfg[k] for k in ("data_dir", "subjects", "seeds", "dimensions", "lambdas",
                                "training", "attackers", "temperature")}


@contextmanager
def session1_only():
    """Process audit guard also catches accidental opens outside our loader."""
    state = {"active": True, "opened": [], "blocked": []}

    def audit(event, args):
        if not state["active"] or event != "open":
            return
        path = str(args[0]).replace("\\", "/").lower()
        name = path.rsplit("/", 1)[-1]
        if name.startswith("sess2") and name.endswith(".npz"):
            state["blocked"].append(path)
            raise LeakageError("dev may not open any sess2 NPZ")
        if name.startswith("sess1") and name.endswith(".npz"):
            state["opened"].append(path)

    sys.addaudithook(audit)
    try:
        yield state
    finally:
        state["active"] = False


def valid_covariances(signals):
    """Preserve invalid rows as null releases; only finite trials enter fitting."""
    good = signals.meta.finite.to_numpy(bool)
    channels = signals.X.shape[1]
    X = np.full((len(signals), channels, channels), np.nan)
    X[good] = covariances(signals.subset(good)).X
    return signals.with_X(X)


def finite_trials(trials):
    return trials.subset(trials.meta.finite.to_numpy(bool))


def phase_meta(meta):
    return meta.assign(run=np.where(meta.run.to_numpy() <= 2, 1, 2))


def settings(cfg):
    for rc in (False, True):
        suffix = "-RC" if rc else ""
        yield dict(key="P0" + suffix, condition="P0" + suffix, d=8, target=0., rc=rc)
        for d in cfg["dimensions"]:
            yield dict(key=f"P2{suffix}_d{d}", condition="P2" + suffix,
                       d=d, target=0., rc=rc)
        if not rc:
            for target in cfg["lambdas"]:
                yield dict(key=f"P3_lambda{target:g}", condition="P3",
                           d=8, target=float(target), rc=False)


def setting_seeds(key, cfg):
    return cfg["seeds"] if key in ("P0", "P2-RC_d8", "P3_lambda1") else cfg["descriptive_seeds"]


def analysis_jobs(cfg):
    """One job = one fixed collection/test split and its complete attacker set."""
    jobs = []
    def add(kind, key, seed, scenario="main", direction="offline_to_online", **extra):
        jobs.append(dict(id=len(jobs), kind=kind, key=key, seed=seed, scenario=scenario,
                         direction=direction, mode="feature", tau=-1., k=1,
                         enrolment="all", enrolled=len(cfg["subjects"]), **extra))
    splits = [("auxiliary", "T_to_E"), ("main", "offline_to_online"), ("main", "online_to_offline")]
    for setting in settings(cfg):
        key = setting["key"]
        if key != "P0-RC":
            for seed in setting_seeds(key, cfg):
                for scenario, direction in splits:
                    add("representation", key, seed, scenario, direction)
    for key in cfg["observation_conditions"]:
        for mode, tau in [("feature", -1.)] + [("command", v) for v in cfg["observation_taus"]]:
            for k in cfg["k"]:
                confirmatory = key == "P2-RC_d8" and mode == "command" and tau == .5 and k in (1, 20)
                for seed in cfg["seeds"] if confirmatory else cfg["descriptive_seeds"]:
                    for scenario, direction in splits:
                        add("observation", key, seed, scenario, direction)
                        jobs[-1].update(mode=mode, tau=tau, k=k)
    for key in cfg["enrolment_conditions"]:
        for n in cfg["enrolment_n"]:
            seeds = cfg["seeds"] if key == "G" and n == 5 else cfg["descriptive_seeds"]
            for seed in seeds:
                for direction in ("offline_to_online", "online_to_offline"):
                    add("enrolment", key, seed, direction=direction)
                    jobs[-1]["enrolment"] = str(n)
    for key in cfg["enrolled_conditions"]:
        for count in cfg["enrolled_counts"]:
            for seed in cfg["descriptive_seeds"]:
                for direction in ("offline_to_online", "online_to_offline"):
                    add("enrolled", key, seed, direction=direction)
                    jobs[-1]["enrolled"] = count
    for key in ("G", "G-raw"):
        add("benchmark", key, cfg["seeds"][0], "auxiliary", "T_to_E")
    return jobs


def job_type(job):
    if job["kind"] == "observation":
        return job["mode"] + "_groups"
    return "N_scaling" if job["kind"] == "enrolled" else job["kind"]


def budget_key(rc, d):
    return f"{'rc' if rc else 'common'}_d{d}"


def training_config(cfg, d):
    return dict(cfg["training"], bottleneck=d)


def parallel(cfg, jobs):
    workers = cfg["workers"]
    if not 1 <= workers <= 6:
        raise ValueError("Shared CPU limit: 1..6 workers")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return Parallel(n_jobs=workers)(jobs)


def _prepare_fold(train, val, rc, path):
    with threadpool_limits(limits=1):
        features = FeatureMap(rc).fit(train)
        _, X = features.transform(train)
        _, V = features.transform(val)
        joblib.dump(dict(features=features, train=train.with_X(X),
                         val=val.with_X(V)), path)


def _pilot(path, rc, d, seed, fold, cfg):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        packet = joblib.load(path)
        train, val = packet["train"], packet["val"]
        require_training(val)
        tc = training_config(cfg, d)
        epochs = tc["max_epochs"]
        checkpoints = sorted(set(range(tc["checkpoint_interval"], epochs + 1,
                                       tc["checkpoint_interval"])) | {epochs})
        states = fit_encoder(train, tc, seed, epochs, 0., checkpoints)
        rows = []
        for epoch, state in states.items():
            model = freeze_encoder(packet["features"], state, train.X.shape[1],
                                   np.unique(train.subject), tc)
            with torch.inference_mode():
                logits = model.predictor(torch.as_tensor(val.X))[1].numpy()
            ba = np.mean([row["ba"] for row in _task_rows(val, logits)])
            rows.append(dict(family=budget_key(rc, d), fold=fold, seed=seed,
                             epoch=epoch, ba=float(ba)))
        return rows


def _oof(path, setting, seed, budget, cfg):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        packet = joblib.load(path)
        train, val = packet["train"], packet["val"]
        require_training(train)
        require_training(val)
        tc = training_config(cfg, setting["d"])
        if setting["condition"].startswith("P0"):
            predictor = LogisticRegression(C=tc["linear_C"], max_iter=tc["linear_max_iter"],
                                           random_state=seed).fit(train.X, train.y)
            scores = predictor.decision_function(val.X)
            logits = np.column_stack([-scores / 2, scores / 2])
        else:
            states = fit_encoder(train, tc, seed, budget, setting["target"])
            model = freeze_encoder(packet["features"], states[budget], train.X.shape[1],
                                   np.unique(train.subject), tc)
            with torch.inference_mode():
                logits = model.predictor(torch.as_tensor(val.X))[1].numpy()
        return dict(key=setting["key"], seed=seed, logits=logits, meta=val.meta)


def attack_job_count(cfg):
    counts = {}
    for job in analysis_jobs(cfg):
        name = job_type(job)
        counts[name] = counts.get(name, 0)+1
    return dict(counts, total=sum(counts.values()))


def run_dev(signals, cfg, result_dir=RESULTS, artifact_dir=ARTIFACTS, *, timing=True):
    require_training(signals)
    start = time.perf_counter()
    out, art = Path(result_dir), Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    (art / "folds").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        train = finite_trials(valid_covariances(signals))
    folds = sorted(map(int, np.unique(train.run)))
    if folds != [1, 2, 3, 4]:
        raise ValueError("Development requires four Lee2019 blocks")
    foldpath = lambda rc, fold: art / "folds" / f"{int(rc)}_{fold}.joblib"
    parallel(cfg, (delayed(_prepare_fold)(
        train.subset(train.run != fold), train.subset(train.run == fold), rc,
        foldpath(rc, fold)) for rc in (False, True) for fold in folds))
    pilots = parallel(cfg, (delayed(_pilot)(foldpath(rc, fold), rc, d, seed, fold, cfg)
                           for rc in (False, True) for d in cfg["dimensions"]
                           for seed in cfg["seeds"] for fold in folds))
    history = pd.DataFrame([r for part in pilots for r in part])
    history.to_csv(out / "epoch_history.csv", index=False)
    budgets = choose_epochs(history)
    packets = parallel(cfg, (delayed(_oof)(
        foldpath(setting["rc"], fold), setting, seed,
        budgets[budget_key(setting["rc"], setting["d"])], cfg)
        for setting in settings(cfg) for seed in cfg["seeds"] for fold in folds))
    temperatures, taskrows = {}, []
    for setting in settings(cfg):
        key = setting["key"]
        temperatures[key] = {}
        for seed in cfg["seeds"]:
            selected = [p for p in packets if p["key"] == key and p["seed"] == seed]
            logits = np.concatenate([p["logits"] for p in selected])
            meta = pd.concat([p["meta"] for p in selected], ignore_index=True)
            oof = Trials(logits, meta)
            temperature = fit_temperature(oof, **cfg["temperature"])
            temperatures[key][str(seed)] = temperature
            taskrows.extend(_task_rows(oof, logits, key=key, seed=seed))
            joblib.dump(oof, art / f"oof_{key}_{seed}.joblib")
    pd.DataFrame(taskrows).to_csv(out / "dev_task.csv", index=False)
    selection = dict(config_hash=digest(analysis_config(cfg)),
                     development_config_hash=digest(development_config(cfg)), budgets=budgets,
                     temperatures=temperatures, subjects=sorted(map(int, np.unique(train.subject))),
                     session="T", folds=folds, n_trials=len(train))
    write_json(out / "selection.json", selection)
    projection = run_timing(signals, cfg, selection, out, art) if timing else {"status": "synthetic"}
    projection["development_seconds"] = time.perf_counter() - start
    write_json(out / "runtime_projection.json", projection)
    return selection


def phase_groups(meta, k, stride):
    """Reset grouping at phase boundaries and every excluded/missing trial."""
    meta = meta.reset_index(drop=True)
    groups = []
    for _, frame in phase_meta(meta).groupby(["subject", "session", "run"], sort=True):
        valid = frame.index[frame.finite.to_numpy(bool)].to_numpy()
        if not len(valid):
            continue
        split = np.flatnonzero(np.diff(meta.trial.to_numpy()[valid]) != 1) + 1
        for segment in np.split(valid, split):
            local, _ = observation_groups(phase_meta(meta.iloc[segment]), k, stride)
            if len(local):
                groups.extend(segment[local])
    groups = np.asarray(groups, dtype=int).reshape(-1, k)
    rows = meta.iloc[groups[:, 0]][["subject", "session", "run", "trial"]].reset_index(drop=True)
    rows["last_trial"] = meta.trial.to_numpy()[groups[:, -1]]
    return groups, rows


def grouped(messages, meta, k, mode, stride):
    groups, rows = phase_groups(meta, k, stride)
    if not len(groups):
        width = 13 if mode == "command" else 2*messages.shape[1]
        return np.empty((0, width)), rows
    descriptors, finite = (command_descriptors(messages, groups) if mode == "command"
                           else feature_descriptors(messages, groups))
    return descriptors[finite], rows.loc[finite].reset_index(drop=True)


def replay_control(meta, scores, temperature, tau, controller, head=None):
    groups, _ = phase_groups(meta, controller["episode_length"], controller["episode_length"])
    retained = groups.ravel()
    emeta = phase_meta(meta.iloc[retained]).reset_index(drop=True)
    block, _, starts = episode_layout(emeta, controller["episode_length"])
    commands, angles, blocked, elapsed = control_pass(
        scores[retained], temperature, tau, starts, controller, head=head)
    episodes = score_episodes(emeta, commands, angles, blocked, ~emeta.finite.to_numpy(bool), controller)
    bounds = emeta.assign(block=block).groupby(["subject", "session", "run", "block"]).trial.agg(
        first_trial="first", last_trial="last").reset_index()
    episodes = episodes.merge(bounds, on=["subject", "session", "run", "block"], validate="one_to_one")
    return episodes, commands, angles, elapsed


def _fit_attack(A, subjects, cfg, seed):
    """Identical B estimators; fixed SVC refit directly, without redundant CV."""
    A, subjects = np.asarray(A), np.asarray(subjects)
    _, counts = np.unique(subjects, return_counts=True)
    if len(counts) < 2 or not len(A) or not np.isfinite(A).all():
        raise ValueError("Finite observations from at least two identities are required")
    c = cfg["attackers"]
    if len(c["svm_C"]) != 1 or len(c["svm_gamma"]) != 1:
        raise ValueError("Only the approved single fixed SVM setting is supported")
    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(
            C=c["logistic_C"], max_iter=c["logistic_max_iter"], solver="lbfgs", random_state=seed)),
        "rbf_svm": make_pipeline(StandardScaler(), SVC(C=c["svm_C"][0], gamma=c["svm_gamma"][0], kernel="rbf")),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=tuple(c["mlp_hidden"]), alpha=c["mlp_alpha"],
            solver=c["mlp_solver"], max_iter=c["mlp_max_iter"],
            max_fun=c["mlp_max_fun"], random_state=seed))}
    for name, model in models.items():
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(A, subjects)
        messages = [str(w.message) for w in observed if issubclass(w.category, ConvergenceWarning)]
        for warning in observed:
            if name != "mlp" or not issubclass(warning.category, ConvergenceWarning):
                warnings.warn(warning.message, warning.category, stacklevel=2)
        if name == "mlp":
            predictor = model.steps[-1][1]
            model.convergence_ = dict(converged=not messages, iterations=int(predictor.n_iter_),
                                      loss=float(predictor.loss_), warning=" | ".join(messages))
    return models


def _predict_attackers(A, subjects, B, cfg, seed):
    with threadpool_limits(limits=1):
        models = _fit_attack(A, subjects, cfg, seed)
        return ({name: model.predict(B) for name, model in models.items()},
                {name: model.convergence_ for name, model in models.items() if name == "mlp"})


def bounded_job(function, cfg, *args):
    timeout = min(cfg["runtime"]["attack_timeout_seconds"],
                  cfg.get("_deadline", float("inf")) - time.monotonic())
    if timeout <= 0:
        raise WorkerTimeout("Sealed final runtime exhausted")
    if cfg.get("_inside_job"):
        return function(*args)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return Parallel(n_jobs=2, timeout=timeout)([delayed(function)(*args)])[0]


def paper_predictions(A, y, B):
    with threadpool_limits(limits=1):
        return SVC().fit(A, y).predict(B)


def attack(A, ameta, B, bmeta, cfg, seed, tags):
    finite_a, finite_b = np.isfinite(A).all(axis=1), np.isfinite(B).all(axis=1)
    A, ameta = A[finite_a], ameta.loc[finite_a]
    B, bmeta = B[finite_b], bmeta.loc[finite_b]
    predictions, diagnostics = [], []
    try:
        if len(np.unique(ameta.subject)) != tags["enrolled"] or not len(B):
            raise ValueError("Missing required identities or test observations")
        results, convergence = bounded_job(_predict_attackers, cfg, A, ameta.subject.to_numpy(), B, cfg, seed)
    except (ValueError, FloatingPointError, WorkerTimeout) as exc:
        diagnostics.append(dict(record="failure", **tags, seed=seed,
                                error=type(exc).__name__, reason=str(exc)))
        return pd.DataFrame(columns=TAGS + ["subject", "run", "trial", "prediction",
                                            "seed", "attacker", "scenario", "direction"]), diagnostics
    for name, prediction in results.items():
        frame = bmeta[["subject", "run", "trial"]].copy()
        frame["prediction"] = prediction
        for key, value in dict(tags, seed=seed, attacker=name).items():
            frame[key] = value
        predictions.append(frame)
        if name == "mlp":
            diagnostics.append(dict(record="convergence", **tags, seed=seed, **convergence[name]))
    return pd.concat(predictions, ignore_index=True), diagnostics


def scenarios(T, E):
    yield "auxiliary", "T_to_E", np.arange(len(T)), np.arange(len(E)), True
    yield "main", "offline_to_online", np.flatnonzero(E.run <= 2), np.flatnonzero(E.run > 2), False
    yield "main", "online_to_offline", np.flatnonzero(E.run > 2), np.flatnonzero(E.run <= 2), False


def attack_stream(tz, ez, T, E, cfg, seed, tags, *, mode=None, k=1):
    frames, diagnostics = [], []
    for scenario, direction, a, b, auxiliary in scenarios(T, E):
        if "_direction" in cfg and direction != cfg["_direction"]:
            continue
        source, smeta = (tz, T.meta) if auxiliary else (ez, E.meta)
        A, ameta, B, bmeta = source[a], smeta.iloc[a], ez[b], E.meta.iloc[b]
        if mode is not None:
            if tags["key"] == "P2-RC_d8" and mode == "command" and tags["tau"] == .5 and k in (1, 20):
                support, _ = phase_groups(bmeta, 20, 20)
                support = np.unique(support)
                B, bmeta = B[support], bmeta.iloc[support]
            A, ameta = grouped(A, ameta, k, mode, 1)
            B, bmeta = grouped(B, bmeta, k, mode, k)
        for role, meta in (("train", ameta), ("test", bmeta)):
            values = A if role == "train" else B
            meta = meta.loc[np.isfinite(values).all(axis=1)]
            for subject in cfg["subjects"]:
                for phase in (1, 2):
                    count = int(((meta.subject == subject) &
                                 (np.where(meta.run <= 2, 1, 2) == phase)).sum())
                    diagnostics.append(dict(record="group_count", **tags, seed=seed,
                                            scenario=scenario, direction=direction,
                                            role=role, subject=subject, phase=phase,
                                            session="T" if role == "train" and auxiliary else "E",
                                            usable_groups=count))
        p, d = attack(A, ameta, B, bmeta, cfg, seed,
                      dict(tags, scenario=scenario, direction=direction))
        frames.append(p)
        diagnostics.extend(d)
    return pd.concat(frames, ignore_index=True), diagnostics


def recalls(frame):
    part = frame.assign(correct=(frame.subject == frame.prediction).astype(float))
    # Direction, seed and participant have equal weight, even after exclusions.
    return part.groupby(["seed", "direction", "subject"]).correct.mean().groupby("subject").mean()


def participant_distribution(values, cfg):
    """Identical participant-index draws for every aligned summary vector."""
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ValueError("Bootstrap requires complete participant-summary vectors")
    rng = np.random.default_rng(cfg["statistics"]["seed"])
    ix = rng.integers(len(values), size=(cfg["statistics"]["bootstrap_samples"], len(values)))
    return values[ix].mean(axis=1)


def declared_subjects(tag, cfg):
    if tag["kind"] == "enrolled":
        return sorted(map(int, np.random.default_rng(cfg["subset_seed"]).permutation(
            sorted(cfg["subjects"]))[:tag["enrolled"]]))
    return sorted(cfg["subjects"])


def cell_issues(frame, subjects, seeds, directions):
    issues = []
    for seed in seeds:
        for direction in directions:
            present = frame[(frame.seed == seed) & (frame.direction == direction)].subject.unique()
            absent = sorted(set(subjects)-set(present))
            if absent:
                issues.append(dict(seed=int(seed), direction=direction, subjects=absent))
    for column, expected in (("subject", subjects), ("seed", seeds), ("direction", directions)):
        extra = sorted(set(frame[column])-set(expected))
        if extra:
            issues.append(dict(unexpected_column=column, values=[
                value.item() if isinstance(value, np.generic) else value for value in extra]))
    return issues


def summary_cells(tag, cfg, descriptive=False):
    jobs = [job for job in analysis_jobs(cfg)
            if all(job[key] == tag[key] for key in TAGS)
            and (not descriptive or job["seed"] in cfg["descriptive_seeds"])]
    attackers = (["paper_ts_svm"] if tag["kind"] == "benchmark" else
                 ["paper_ts_svm", "beat_logistic", "beat_rbf_svm", "beat_mlp"]
                 if tag["kind"] == "enrolment" and tag["key"] in ("G", "G-raw") else
                 ["logistic", "rbf_svm", "mlp"])
    for attacker in sorted(attackers):
        for scenario in sorted({job["scenario"] for job in jobs}):
            selected = [job for job in jobs if job["scenario"] == scenario]
            yield (attacker, scenario, sorted({job["seed"] for job in selected}),
                   sorted({job["direction"] for job in selected}))


def interval(frame, cfg, subjects, seeds, directions):
    if cell_issues(frame, subjects, seeds, directions):
        raise ValueError("Bootstrap requires every declared participant/seed/direction cell")
    dist = participant_distribution(recalls(frame).reindex(sorted(subjects)).to_numpy(), cfg)
    alpha = (1-cfg["statistics"]["confidence"])/2
    return [float(v) for v in np.quantile(dist, [alpha, 1-alpha])]


def paired(difference, margin=0.):
    d = np.asarray(difference, float)
    if len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("Paired inference requires >=2 finite participants")
    shifted = d - margin
    sd = shifted.std(ddof=1)
    p = (float(stats.t.sf(shifted.mean() / (sd/np.sqrt(len(d))), len(d)-1))
         if sd > 0 else (0. if shifted.mean() > 0 else 1.))
    signflip = paired_recall_test(d, "lee2019", seed=0)
    return dict(mean=float(d.mean()), lower=paired_t_lower_bound(d), p=p, n=len(d),
                sign_flip=signflip)


def holm(pvalues):
    order = sorted(pvalues, key=pvalues.get)
    adjusted, previous = {}, 0.
    for i, key in enumerate(order):
        previous = max(previous, (len(order)-i)*pvalues[key])
        adjusted[key] = min(1., previous)
    return adjusted


def candidate_vectors(frame, cfg, attackers=("logistic", "rbf_svm", "mlp"),
                      scenarios_required=("main", "auxiliary")):
    vectors, missing = {}, []
    for attacker in attackers:
        for scenario in scenarios_required:
            key = f"{attacker}/{scenario}"
            part = frame[(frame.attacker == attacker) & (frame.scenario == scenario)]
            directions = ("offline_to_online", "online_to_offline") if scenario == "main" else ("T_to_E",)
            for seed in cfg["seeds"]:
                for direction in directions:
                    present = part[(part.seed == seed) & (part.direction == direction)].subject.unique()
                    absent = sorted(set(cfg["subjects"]) - set(present))
                    if absent:
                        missing.append(dict(candidate=key, seed=seed, direction=direction, subjects=absent))
            vectors[key] = recalls(part).reindex(cfg["subjects"])
    return vectors, missing


def infer_claims(r0, r2, r3, h, u, c1, c20, g, benchmark, cfg, unavailable=None):
    """Exact plan-v3 intersection/union tests; no winner-selected inference."""
    def test(v, a=0.):
        return paired(v, a)
    J, K, N = 6, 4, len(cfg["subjects"])
    unavailable = unavailable or {}
    gate = benchmark is not None and abs(benchmark-.024) <= .05
    claims = {
        "C1": dict(p=1., observed_difference=None, effect_requirement=None, component_tests=None),
        "C2": dict(p=1., observed_difference=None, effect_requirement=None, utility=None, component_tests=None),
        "C3": dict(p=1., component_tests=None),
        "C4": dict(p=1., benchmark_gate=bool(gate), component_tests=None)}
    if not unavailable.get("C1"):
        components = {j: test(v-h) for j, v in r3.items()}
        gap = max(v.mean() for v in r3.values()) - h.mean()
        p = min(1., J*min(t["p"] for t in components.values())) if gap >= .20 else 1.
        claims["C1"].update(p=p, observed_difference=float(gap), effect_requirement=bool(gap >= .20),
                            component_tests=components)
    if not unavailable.get("C2"):
        components = {j: {l: test(v-w) for l, w in r2.items()} for j, v in r0.items()}
        gap = max(v.mean() for v in r0.values()) - max(v.mean() for v in r2.values())
        utility = test(u, -.05)
        protection = min(1., J*min(max(t["p"] for t in row.values()) for row in components.values()))
        p = max(utility["p"], protection) if gap >= .20 else 1.
        claims["C2"].update(p=p, observed_difference=float(gap), effect_requirement=bool(gap >= .20),
                            utility=utility, component_tests=components)
    if not unavailable.get("C3"):
        components = {j: dict(threshold=test(v, 2/N),
                              differences={l: test(v-w) for l, w in c1.items()})
                      for j, v in c20.items()}
        p = min(1., J*min(max(row["threshold"]["p"], max(t["p"] for t in row["differences"].values()))
                         for row in components.values()))
        claims["C3"].update(p=p, component_tests=components)
    if not unavailable.get("C4"):
        components = {j: test(v, 10/N) for j, v in g.items()}
        p = min(1., K*min(t["p"] for t in components.values())) if gate else 1.
        claims["C4"].update(p=p, component_tests=components)
    return claims


def make_decision(predictions, task, heads, reproduction, cfg):
    data, missing = {}, {}
    for key in ("P0", "P2-RC_d8", "P3_lambda1"):
        data[key], missing[key] = candidate_vectors(predictions.select(kind="representation", key=key), cfg)
    head = heads[heads.key == "P3_lambda1"]
    h = recalls(head).reindex(cfg["subjects"])
    head_missing = cell_issues(head, cfg["subjects"], cfg["seeds"],
                               ["offline_to_online", "online_to_offline"])
    means = task.groupby(["key", "subject"]).ba.mean().unstack("key").reindex(
        index=cfg["subjects"], columns=["P2-RC_d8", "P0"])
    utility = means["P2-RC_d8"] - means["P0"]
    commands = predictions.select(kind="observation", key="P2-RC_d8", mode="command", tau=.5)
    c1, missing["k1"] = candidate_vectors(commands[commands.k == 1], cfg)
    c20, missing["k20"] = candidate_vectors(commands[commands.k == 20], cfg)
    enrol = predictions.select(kind="enrolment", key="G", enrolment="5")
    g, missing["G"] = candidate_vectors(enrol, cfg,
                                       ("paper_ts_svm", "beat_logistic", "beat_rbf_svm", "beat_mlp"),
                                       ("main",))
    affected = {"C1": missing["P3_lambda1"] + head_missing,
                "C2": missing["P0"] + missing["P2-RC_d8"],
                "C3": missing["k1"] + missing["k20"], "C4": missing["G"]}
    for key in ("P0", "P2-RC_d8"):
        for seed in cfg["seeds"]:
            absent = sorted(set(cfg["subjects"]) - set(task[(task.key == key) & (task.seed == seed)].subject))
            if absent:
                affected["C2"].append(dict(task=key, seed=seed, subjects=absent))
    claims = infer_claims(data["P0"], data["P2-RC_d8"], data["P3_lambda1"],
                          h, utility, c1, c20, g, reproduction["accuracy"], cfg,
                          unavailable=affected)
    for key, cells in affected.items():
        claims[key]["missing_cells"] = cells
    adjusted = holm({key: value["p"] for key, value in claims.items()})
    for key, value in claims.items():
        value["holm_p"] = adjusted[key]
        value["decision"] = "supported" if adjusted[key] <= .05 else "not_supported"
        if value["missing_cells"]:
            value["decision"] = "not_evaluable_missing_cells"
        if key == "C4" and not value["benchmark_gate"]:
            value["decision"] = "not_evaluable_benchmark_gate"
            value["reproduction"] = reproduction
    return dict(claims=claims, chance=1/len(cfg["subjects"]),
                inference="plan-v3 cohort maxima; J=6/K=4 Bonferroni union and max conjunction t-tests",
                intervals="complete participant-summary-vector bootstrap; paired draws, conditional on fitted systems",
                multiplicity="Holm four claims; not-evaluable p=1 retains family",
                reproduction_caveat="Reduced causal 3-second epochs differ from paper preprocessing",
                threat_model=THREAT_MODEL_INTERPRETATION)


def _enrolment(GT, GE, cfg, condition):
    """G uses attacker's own tangent reference fitted only on collected labels."""
    frames, diagnostics = [], []
    for _, direction, a, b, auxiliary in scenarios(GT, GE):
        if auxiliary:
            continue
        if "_direction" in cfg and direction != cfg["_direction"]:
            continue
        for n in cfg["enrolment_n"]:
            available = finite_trials(GE.subset(a))
            if n != "all" and (available.meta.groupby("subject").size().reindex(
                    cfg["subjects"], fill_value=0) < n).any():
                diagnostics.append(dict(record="failure", key=condition, enrolment=str(n),
                                        direction=direction, reason="Insufficient retained enrollment trials"))
                continue
            train = first_labelled_trials(available, n)
            test = GE.subset(b)
            if n != "all" and (train.meta.groupby("subject").size().reindex(
                    cfg["subjects"], fill_value=0) < n).any():
                diagnostics.append(dict(record="failure", key=condition, enrolment=str(n),
                                        direction=direction, reason="Insufficient retained enrollment trials"))
                continue
            try:
                if condition in ("G", "G-raw"):
                    ts = TangentSpace(metric="riemann", tsupdate=False)
                    A, B = ts.fit_transform(train.X), ts.transform(test.X)
                else:
                    A, B = train.X, test.X
            except (ValueError, FloatingPointError) as exc:
                diagnostics.append(dict(record="failure", key=condition, enrolment=str(n),
                                        direction=direction, reason=str(exc)))
                continue
            for seed in cfg["seeds"]:
                p, d = attack(A, train.meta, B, test.meta, cfg, seed,
                              dict(kind="enrolment", key=condition, mode="feature",
                                   tau=-1., k=1, enrolment=str(n), enrolled=len(cfg["subjects"]),
                                   scenario="main", direction=direction))
                if condition in ("G", "G-raw"):
                    p["attacker"] = "beat_" + p.attacker
                    pp = test.meta[["subject", "run", "trial"]].copy()
                    try:
                        pp["prediction"] = bounded_job(paper_predictions, cfg, A, train.subject, B)
                    except (ValueError, FloatingPointError, WorkerTimeout) as exc:
                        diagnostics.append(dict(record="failure", key="G", enrolment=str(n),
                                                direction=direction, attacker="paper_ts_svm",
                                                seed=seed, reason=str(exc)))
                        frames.append(p)
                        diagnostics.extend(d)
                        continue
                    for name, value in dict(kind="enrolment", key=condition, mode="feature", tau=-1.,
                                            k=1, enrolment=str(n), enrolled=len(cfg["subjects"]),
                                            scenario="main", direction=direction, seed=seed,
                                            attacker="paper_ts_svm").items():
                        pp[name] = value
                    p = pd.concat([p, pp], ignore_index=True)
                frames.append(p)
                diagnostics.extend(d)
    return frames, diagnostics


def memory_snapshot():
    """Windows process high-water RSS and current physical memory, in MiB."""
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
            (name, ctypes.c_uint64) for name in
            ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
    class ProcessMemory(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("faults", ctypes.c_uint32)] + [
            (name, ctypes.c_size_t) for name in
            ("peak", "rss", "paged_peak", "paged", "nonpaged_peak", "nonpaged", "pagefile", "pagefile_peak")]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    process = ProcessMemory()
    process.cb = ctypes.sizeof(process)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(handle), ctypes.byref(process), process.cb):
        raise ctypes.WinError()
    return dict(available_mib=status.available/2**20, rss_mib=process.rss/2**20, peak_mib=process.peak/2**20)


def execute_job(job, cfg, art, selection, output_dir):
    """Importable, single-thread worker; no data loading or defended fitting."""
    start = time.perf_counter()
    local = dict(cfg, _inside_job=True, _direction=job["direction"])
    key, seed = job["key"], job["seed"]
    source = Path(art) / (f"covariances_{key}.joblib" if key in ("G", "G-raw")
                          else f"outputs_{key}_{seed}.joblib")
    packet = joblib.load(source, mmap_mode="r")
    T, E = Trials(packet["tz"], packet["T"]), Trials(packet["ez"], packet["E"])
    tags = {name: job[name] for name in TAGS}
    try:
        with threadpool_limits(limits=1):
            if job["kind"] == "benchmark":
                ts = TangentSpace(metric="riemann", tsupdate=False)
                A, B = ts.fit_transform(T.X), ts.transform(E.X)
                predicted = paper_predictions(A, T.subject, B)
                frame = E.meta[["subject", "run", "trial"]].copy()
                frame["prediction"] = predicted
                for name, value in dict(tags, seed=seed, scenario=job["scenario"],
                                        direction=job["direction"], attacker="paper_ts_svm").items():
                    frame[name] = value
                diagnostics = []
            elif job["kind"] == "enrolment":
                n = job["enrolment"]
                n = n if n == "all" else int(n)
                parts, diagnostics = _enrolment(T, E, dict(local, seeds=[seed], enrolment_n=[n]), key)
                frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            else:
                if job["kind"] == "enrolled":
                    ids = declared_subjects(job, cfg)
                    T, E = T.subset(np.isin(T.subject, ids)), E.subset(np.isin(E.subject, ids))
                    local["subjects"] = sorted(map(int, ids))
                tz, ez = T.X, E.X
                mode = job["mode"] if job["kind"] == "observation" else None
                if mode == "command":
                    temperature = selection["temperatures"][key][str(seed)]
                    tz = decide(packet["tl"], temperature, job["tau"])
                    ez = decide(packet["el"], temperature, job["tau"])
                frame, diagnostics = attack_stream(tz, ez, T, E, local, seed, tags, mode=mode, k=job["k"])
        failed = any(d.get("record") == "failure" for d in diagnostics)
        path = Path(output_dir) / f"job_{job['id']:04d}.joblib"
        if len(frame):
            joblib.dump(frame, path, compress=1)
        return dict(job=job, tags=tags, prediction_path=str(path) if len(frame) else None,
                    diagnostics=diagnostics, status="failed" if failed else "ok",
                    seconds=time.perf_counter()-start, **memory_snapshot())
    except (ValueError, FloatingPointError, MemoryError) as exc:
        return dict(job=job, tags=tags, prediction_path=None, status="failed",
                    diagnostics=[dict(record="failure", **tags, seed=seed, reason=str(exc))],
                    seconds=time.perf_counter()-start, **memory_snapshot())


def run_jobs(jobs, cfg, art, selection, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    memory = memory_snapshot()
    budget = int((memory["available_mib"]-cfg["memory"]["reserve_mib"]) / cfg["memory"]["per_worker_mib"])
    workers = min(cfg["workers"], max(1, budget))
    if not 1 <= cfg["workers"] <= 6:
        raise ValueError("At most six workers are authorized")
    records, free_memory = [], []
    start = time.perf_counter()
    for first in range(0, len(jobs), workers):
        free = memory_snapshot()["available_mib"]
        free_memory.append(free)
        if free < cfg["memory"]["reserve_mib"]:
            raise MemoryError("Physical memory reserve reached; no remaining cells are dropped or reselected")
        timeout = min(cfg["runtime"]["attack_timeout_seconds"],
                      cfg.get("_deadline", float("inf"))-time.monotonic())
        if timeout <= 0:
            raise WorkerTimeout("Final runtime exhausted before next batch")
        batch = jobs[first:first+workers]
        with parallel_config(backend="loky", inner_max_num_threads=1):
            try:
                completed = Parallel(n_jobs=max(2, workers), timeout=timeout, batch_size=1,
                                     pre_dispatch=workers)(
                    delayed(execute_job)(job, cfg, art, selection, output_dir) for job in batch)
            except WorkerTimeout:
                completed = [dict(job=job, tags={k: job[k] for k in TAGS}, prediction_path=None,
                                  status="failed", seconds=timeout,
                                  diagnostics=[dict(record="failure", key=job["key"],
                                                    reason="Sealed attack-batch timeout")]) for job in batch]
        records.extend(completed)
    return records, dict(workers=workers, requested_workers=cfg["workers"],
                         wall_seconds=time.perf_counter()-start,
                         minimum_free_mib=min(free_memory), initial_memory=memory)


def timing_profile(job):
    # Reverse-direction and seed variation are explicitly unmeasured; no size scaling.
    return tuple(job[k] for k in ("kind", "key", "scenario", "mode", "tau", "k", "enrolment", "enrolled"))


def run_timing(signals, cfg, selection, result_dir=RESULTS, artifact_dir=ARTIFACTS):
    require_training(signals)
    out, art = Path(result_dir), Path(artifact_dir) / "timing"
    evaluation = Trials(signals.X, signals.meta.assign(session="E"))
    local = dict(cfg, _deadline=time.monotonic()+cfg["runtime"]["final_limit_seconds"])
    _, _, _, _, preparation = prepare_artifacts(signals, evaluation, local, selection, art, standin=True)
    jobs = analysis_jobs(cfg)
    profiles = {}
    for job in jobs:
        profile = timing_profile(job)
        if profile not in profiles:
            profiles[profile] = dict(job, seed=cfg["seeds"][0])
    measured, scheduling = run_jobs(list(profiles.values()), local, art, selection, art/"predictions")
    rows = [dict(r["job"], job_type=job_type(r["job"]), seconds=r["seconds"],
                 status=r["status"], peak_mib=r.get("peak_mib"), available_mib=r.get("available_mib"))
            for r in measured]
    pd.DataFrame(rows).to_csv(out/"job_type_timings.csv", index=False)
    lookup = {timing_profile(r["job"]): r for r in measured}
    total_measured = sum(r["seconds"] for r in measured)
    parallelism = min(scheduling["workers"], total_measured/scheduling["wall_seconds"])
    breakdown = {}
    for job in jobs:
        row = breakdown.setdefault(job_type(job), dict(jobs=0, measured_profiles=0, projected_work_seconds=0.))
        row["jobs"] += 1
        row["projected_work_seconds"] += lookup[timing_profile(job)]["seconds"]
    for name, row in breakdown.items():
        row["measured_profiles"] = sum(job_type(r["job"]) == name for r in measured)
        row["projected_wall_hours"] = row["projected_work_seconds"]/parallelism/3600
    model_seconds = sum(row["seconds"]*len(setting_seeds(row["key"], cfg))
                        for row in preparation if row["stage"] == "model")
    other_seconds = sum(row["seconds"] for row in preparation if row["stage"] != "model")
    # Reports/paired statistics are cheap relative to attacks but not free: measure
    # the real bootstrap kernel at the full cohort size for every final summary cell.
    begin = time.perf_counter()
    for _ in range(attack_job_count(cfg)["total"]*3):
        participant_distribution(np.linspace(0, 1, len(cfg["subjects"])), cfg)
    report_seconds = time.perf_counter()-begin
    failures = [r["job"] for r in measured if r["status"] != "ok"]
    hours = sum(row["projected_wall_hours"] for row in breakdown.values()) + (
        model_seconds+other_seconds+report_seconds)/3600
    projection = dict(
        status="unverifiable_failed_profiles" if failures else ("over_12_hours_report_without_further_cuts" if hours > 12 else "measured_projection"),
        projected_final_hours=hours if not failures else None, breakdown=breakdown,
        job_counts=attack_job_count(cfg), measured_profiles=len(measured),
        effective_parallelism=parallelism, scheduling=scheduling,
        projected_serial_preparation_seconds=model_seconds+other_seconds,
        measured_reporting_kernel_seconds=report_seconds, preparation=preparation,
        max_worker_peak_mib=max(r.get("peak_mib", 0.) for r in measured),
        failed_profiles=failures, attacker_settings=cfg["attackers"],
        projection_basis="Measured S1 job profiles at actual row counts/dimensions; weighted by sealed job counts and observed parallel throughput",
        limits="S1 stand-ins reuse S1 outputs for auxiliary test timing only; no scientific scores. One seed/direction per profile; S2 difficulty and full report I/O unmeasured.",
        session2_opened=False)
    write_json(out/"runtime_projection.json", projection)
    return projection


def sealed_maximum(tag, rows, cfg, descriptive=False):
    missing = []
    candidates = []
    expected = list(summary_cells(tag, cfg, descriptive))
    for attacker, scenario, _, _ in expected:
        matches = [row for row in rows if row["attacker"] == attacker and row["scenario"] == scenario]
        if len(matches) != 1:
            missing.append(dict(attacker=attacker, scenario=scenario,
                                reason="missing_or_duplicate_candidate", missing_cells=[]))
        elif matches[0]["status"] != "ok" or not np.isfinite(matches[0]["ba"]):
            row = matches[0]
            missing.append(dict(attacker=attacker, scenario=scenario, reason=row["status"],
                                missing_cells=json.loads(row["missing_cells"])))
        else:
            candidates.append(matches[0])
    if not expected:
        missing.append(dict(reason="unplanned_candidate_set", missing_cells=[]))
    winner = None if missing else max(candidates, key=lambda row: row["ba"])
    return winner, missing


def summary_rows(tag, frame, cfg, descriptive=False):
    subjects = declared_subjects(tag, cfg)
    rows = []
    for attacker, scenario, seeds, directions in summary_cells(tag, cfg, descriptive):
        part = frame[(frame.attacker == attacker) & (frame.scenario == scenario)]
        issues = cell_issues(part, subjects, seeds, directions)
        row = dict(tag, attacker=attacker, scenario=scenario, ba=None, macro_f1=None,
                   ci_low=None, ci_high=None, strongest=False, identities=len(subjects),
                   seeds=",".join(map(str, seeds)), missing_cells=json.dumps(issues),
                   status="not_evaluable_incomplete_cells" if issues else "ok")
        if not issues:
            r = recalls(part).reindex(subjects)
            macro = np.mean([f1_score(s.subject, s.prediction, labels=subjects, average="macro", zero_division=0)
                             for _, s in part.groupby(["seed", "direction"])])
            ci = interval(part, cfg, subjects, seeds, directions)
            row.update(ba=float(r.mean()), macro_f1=float(macro), ci_low=ci[0], ci_high=ci[1])
        rows.append(row)
    winner, missing = sealed_maximum(tag, rows, cfg, descriptive)
    for row in rows:
        row.update(strongest_status="not_evaluable_incomplete_candidate_set" if missing else "ok",
                   strongest_ba=None if missing else winner["ba"],
                   missing_candidates=json.dumps(missing))
    if winner is not None:
        winner["strongest"] = True
    return rows


def head_summary_rows(heads, summary, cfg):
    rows = []
    for setting in settings(cfg):
        if not setting["condition"].startswith("P3"):
            continue
        key = setting["key"]
        frame = heads[(heads.key == key) & heads.seed.isin(cfg["descriptive_seeds"])]
        issues = cell_issues(frame, sorted(cfg["subjects"]), cfg["descriptive_seeds"],
                             ["offline_to_online", "online_to_offline"])
        tag = dict(kind="representation", key=key, mode="feature", tau=-1.,
                   k=1, enrolment="all", enrolled=len(cfg["subjects"]))
        independent = summary.loc[(summary.key == key) & (summary.kind == "representation")]
        winner, missing = sealed_maximum(tag, independent.to_dict("records"), cfg, descriptive=True)
        rows.append(dict(key=key, head_ba=None if issues else float(recalls(frame).mean()),
                         independent_ba=None if missing else winner["ba"],
                         status="not_evaluable_incomplete_cells" if issues else
                         ("not_evaluable_incomplete_candidate_set" if missing else "ok"),
                         missing_cells=json.dumps(issues), missing_candidates=json.dumps(missing)))
    return rows


def append_predictions(path, frame, first):
    columns = ["subject", "run", "trial", "prediction", *TAGS, "scenario", "direction", "seed", "attacker"]
    with gzip.open(path, "wt" if first else "at", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, columns=columns, index=False, header=first)


def write_tables(predictions, task, heads, controls, timings, diagnostics, cfg, out):
    summaries, descriptive, recalls_rows, directional = [], [], [], []
    first = True
    tags = dict.fromkeys(tuple(job[key] for key in TAGS) for job in analysis_jobs(cfg))
    for values in tags:
        tag = dict(zip(TAGS, values))
        frame = predictions.select(**tag)
        append_predictions(out / "identity_predictions.csv.gz", frame, first)
        first = False
        summaries.extend(summary_rows(tag, frame, cfg))
        subset = frame[frame.seed.isin(cfg["descriptive_seeds"])]
        descriptive.extend(summary_rows(tag, subset, cfg, descriptive=True))
        subjects = declared_subjects(tag, cfg)
        for attacker, scenario, seeds, directions in summary_cells(tag, cfg):
            part = frame[(frame.attacker == attacker) & (frame.scenario == scenario)]
            issues = cell_issues(part, subjects, seeds, directions)
            for seed in seeds:
                for direction in directions:
                    unit = part[(part.seed == seed) & (part.direction == direction)]
                    missing = cell_issues(unit, subjects, [seed], [direction])
                    directional.append(dict(
                        tag, attacker=attacker, scenario=scenario, seed=int(seed), direction=direction,
                        ba=None if missing else float(recalls(unit).reindex(subjects).mean()),
                        macro_f1=None if missing else float(f1_score(unit.subject, unit.prediction,
                                                                     labels=subjects, average="macro", zero_division=0)),
                        status="not_evaluable_incomplete_cells" if missing else "ok",
                        missing_cells=json.dumps(missing)))
            r = recalls(part).reindex(subjects) if not issues else pd.Series(np.nan, index=subjects)
            recalls_rows.extend(dict(tag, attacker=attacker, scenario=scenario,
                                     subject=int(s), recall=float(v),
                                     status="not_evaluable_incomplete_cells" if issues else "ok",
                                     missing_cells=json.dumps(issues)) for s, v in r.items())
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "identity_summary.csv", index=False)
    summary = pd.DataFrame(descriptive)
    summary.to_csv(out / "descriptive_summary.csv", index=False)
    pd.DataFrame(recalls_rows).to_csv(out / "participant_recall.csv", index=False)
    pd.DataFrame(directional).to_csv(out / "identity_per_seed_direction.csv", index=False)
    task.to_csv(out / "task.csv", index=False)
    task_rows = []
    for setting in settings(cfg):
        key = setting["key"]
        if key == "P0-RC":
            continue
        part = task[task.key == key].assign(direction="all")
        issues = cell_issues(part, sorted(cfg["subjects"]), setting_seeds(key, cfg), ["all"])
        values = part.groupby("subject").ba.mean().reindex(sorted(cfg["subjects"]))
        task_rows.append(dict(key=key, mean=None if issues else float(values.mean()),
                              median=None if issues else float(values.median()),
                              std=None if issues else float(values.std()),
                              status="not_evaluable_incomplete_cells" if issues else "ok",
                              missing_cells=json.dumps(issues)))
    pd.DataFrame(task_rows).to_csv(out / "task_summary.csv", index=False)
    heads.to_csv(out / "head_predictions.csv.gz", index=False)
    pd.concat(controls, ignore_index=True).to_csv(out / "control.csv", index=False)
    pd.DataFrame(timings).to_csv(out / "processing_timing.csv", index=False)
    write_json(out / "attacker_diagnostics.json", diagnostics)
    pd.DataFrame([row for row in diagnostics if row.get("record") == "group_count"]).to_csv(
        out / "usable_group_counts.csv", index=False)
    reports = {
        "enrolment": summary[summary.kind == "enrolment"],
        "enrolled_participants": summary[summary.kind == "enrolled"],
        "observation_length": summary[summary.kind == "observation"],
        "dimension_sweep": summary[(summary.kind == "representation") & summary.key.str.startswith("P2")],
        "lambda_sweep": summary[(summary.kind == "representation") & summary.key.str.startswith("P3")]}
    for name, table in reports.items():
        table.to_csv(out / f"{name}.csv", index=False)
    headrows = head_summary_rows(heads, summary, cfg)
    pd.DataFrame(headrows).to_csv(out / "head_vs_independent.csv", index=False)
    twice = summary[summary.kind == "observation"].copy()
    twice["above_twice_chance"] = twice.ba >= 2/len(cfg["subjects"])
    crossings = []
    for tag, frame in twice.groupby(["key", "mode", "tau"]):
        reached = frame[frame.above_twice_chance & frame.strongest]
        incomplete = frame[frame.status != "ok"]
        crossings.append(dict(zip(["key", "mode", "tau"], tag),
                              smallest_k=int(reached.k.min()) if len(reached) and not len(incomplete) else None,
                              status="not_evaluable_incomplete_cells" if len(incomplete) else
                              ("reached" if len(reached) else "not_reached"),
                              missing_cells=json.dumps(incomplete[["attacker", "scenario", "k", "missing_cells"]].to_dict("records"))))
    pd.DataFrame(crossings).to_csv(out / "smallest_k_twice_chance.csv", index=False)
    plot_outputs(summary, task[task.seed.isin(cfg["descriptive_seeds"])], pd.DataFrame(headrows), out)


def strongest_plot_rows(summary, kind):
    rows = summary[summary.kind == kind].drop_duplicates(TAGS).copy()
    rows["ba"] = pd.to_numeric(rows.strongest_ba)
    return rows


def plot_outputs(summary, task, heads, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def unavailable_note(ax, frame, column="strongest_status"):
        count = int((frame[column] != "ok").sum())
        if count:
            ax.text(.01, .02, f"Not evaluable: {count} candidate set(s); see CSV missing-candidate details.",
                    transform=ax.transAxes, fontsize=6)
    fig, ax = plt.subplots()
    features = strongest_plot_rows(summary, "representation")
    utility = task.groupby("key").ba.mean()
    for _, row in features[features.strongest_status == "ok"].iterrows():
        ax.scatter(utility[row.key], row.ba)
        ax.annotate(row.key, (utility[row.key], row.ba), fontsize=6)
    ax.set(xlabel="MI balanced accuracy", ylabel="Strongest identity balanced accuracy")
    unavailable_note(ax, features)
    fig.tight_layout()
    fig.savefig(out / "tradeoff.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots()
    obs = strongest_plot_rows(summary, "observation")
    obs = obs[obs.key == "P2-RC_d8"]
    for (mode, tau), frame in obs.groupby(["mode", "tau"]):
        frame = frame.sort_values("k")
        ax.plot(frame.k, frame.ba, marker="o", label=f"{mode} {tau:g}")
    ax.set(xlabel="Observation length k", ylabel="Identity balanced accuracy")
    unavailable_note(ax, obs)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "observation_length.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots()
    available_heads = heads[heads.status == "ok"]
    ax.scatter(available_heads.head_ba, available_heads.independent_ba)
    for _, row in available_heads.iterrows():
        ax.annotate(row.key, (row.head_ba, row.independent_ba), fontsize=6)
    ax.set(xlabel="Co-trained head BA", ylabel="Independent attacker BA")
    unavailable_note(ax, heads, "status")
    fig.tight_layout()
    fig.savefig(out / "head_vs_independent.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots()
    enrol = strongest_plot_rows(summary, "enrolment")
    for key, frame in enrol.groupby("key"):
        order = {str(n): i for i, n in enumerate([1, 2, 5, 10, 20, "all"])}
        frame = frame.assign(position=frame.enrolment.map(order)).sort_values("position")
        ax.plot(frame.position, frame.ba, marker="o", label=key)
    ax.set(xticks=range(6), xticklabels=["1", "2", "5", "10", "20", "all"],
           xlabel="Labelled deployment trials per identity", ylabel="Identity BA")
    unavailable_note(ax, enrol)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "enrolment.png", dpi=150)
    plt.close(fig)


def prepare_artifacts(training, evaluation, cfg, selection, artifact_dir, *, standin=False):
    """Fit defended systems serially; workers receive only small frozen outputs."""
    require_training(training)
    art = Path(artifact_dir)
    (art / "models").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    tasks, heads, controls, timings, preparation = [], [], [], [], []
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        T = valid_covariances(training)
        E = Trials(T.X, evaluation.meta) if standin else valid_covariances(evaluation)
        preparation.append(dict(stage="covariance", seconds=(time.perf_counter()-start)*(2 if standin else 1)))
        for setting in settings(cfg):
            key = setting["key"]
            seeds = [cfg["seeds"][0]] if standin else setting_seeds(key, cfg)
            for seed in seeds:
                if time.monotonic() >= cfg.get("_deadline", float("inf")):
                    raise WorkerTimeout("Final runtime exhausted before model fitting")
                model_start = time.perf_counter()
                model = fit_defended(finite_trials(T), setting["condition"],
                                     training_config(cfg, setting["d"]), seed,
                                     selection["budgets"][budget_key(setting["rc"], setting["d"])],
                                     setting["target"])
                joblib.dump(model, art / "models" / f"{key}_{seed}.joblib")
                tz, tl = model(T)
                start = time.perf_counter()
                ez, el = model(E)
                elapsed = time.perf_counter()-start
                joblib.dump(dict(T=T.meta, E=E.meta, tz=tz, tl=tl, ez=ez, el=el),
                            art / f"outputs_{key}_{seed}.joblib")
                good = E.meta.finite.to_numpy(bool)
                if key != "P0-RC":
                    tasks.extend(_task_rows(E.subset(good), el[good], key=key, seed=seed))
                if setting["condition"].startswith("P3"):
                    scores = model.adversary_logits(E.subset(good))
                    head = E.subset(good).meta[["subject", "run", "trial"]].copy()
                    head["prediction"] = model.identity_classes_[scores.argmax(axis=1)]
                    head["key"], head["seed"] = key, seed
                    head["direction"] = np.where(head.run <= 2, "online_to_offline", "offline_to_online")
                    heads.append(head)
                if key in cfg["observation_conditions"] and seed in cfg["descriptive_seeds"]:
                    temperature = selection["temperatures"][key][str(seed)]
                    receiver, error = identify_head(T.with_X(tz), tl, 1e-3)
                    for tau in cfg["taus"]:
                        episode, commands, angles, control_time = replay_control(
                            E.meta, el, temperature, tau, cfg["controller"])
                        _, fcommands, fangles, _ = replay_control(
                            E.meta, ez, temperature, tau, cfg["controller"], head=receiver)
                        if not (np.array_equal(commands, fcommands) and np.array_equal(angles, fangles)):
                            raise AssertionError("Feature/command trajectories disagree")
                        episode["key"], episode["seed"], episode["tau"] = key, seed, tau
                        controls.append(episode)
                        timings.append(dict(key=key, seed=seed, tau=tau,
                                            batch_inference_ms_per_trial=elapsed*1000/len(E),
                                            control_median_ms=float(np.median(control_time)) if len(control_time) else None,
                                            control_p95_ms=float(np.quantile(control_time, .95)) if len(control_time) else None,
                                            scored_trials=len(control_time),
                                            affine_error=error, trajectories_equal=True,
                                            note="Reduced data: filtering not timed; replay processing, not online latency"))
                preparation.append(dict(stage="model", key=key, seed=seed, seconds=time.perf_counter()-model_start))
        if time.monotonic() >= cfg.get("_deadline", float("inf")):
            raise WorkerTimeout("Final runtime exhausted before G fitting")
        start = time.perf_counter()
        defense = FourStageDefense(cfg["g_strengths"]).fit(finite_trials(training))
        joblib.dump(defense, art / "models" / "G.joblib")
        fit_seconds = time.perf_counter()-start
        start = time.perf_counter()
        GT = transformed_covariances(defense, finite_trials(training))
        GE = (Trials(GT.X, finite_trials(evaluation).meta) if standin else
              transformed_covariances(defense, finite_trials(evaluation)))
        preparation.append(dict(stage="G", seconds=fit_seconds+(time.perf_counter()-start)*(2 if standin else 1),
                                peak_mib=memory_snapshot()["peak_mib"]))
        for key, t, e in (("G", GT, GE), ("G-raw", finite_trials(T), finite_trials(E))):
            joblib.dump(dict(tz=t.X, ez=e.X, T=t.meta, E=e.meta), art/f"covariances_{key}.joblib")
    return pd.DataFrame(tasks), pd.concat(heads, ignore_index=True), controls, timings, preparation


def run_final(training, evaluation, cfg, selection, result_dir=RESULTS, artifact_dir=ARTIFACTS):
    require_training(training)
    if set(evaluation.session) != {"E"}:
        raise ValueError("Final evaluation must be session E")
    if selection["config_hash"] != digest(analysis_config(cfg)):
        raise ValueError("Configuration changed since session-1 development")
    cfg = dict(cfg, _deadline=time.monotonic()+cfg["runtime"]["final_limit_seconds"])
    out, art = Path(result_dir), Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    task, head, controls, timings, preparation = prepare_artifacts(training, evaluation, cfg, selection, art)
    exclusions = pd.concat([training.meta, evaluation.meta], ignore_index=True)
    exclusions["phase"] = np.where(exclusions.run <= 2, "offline", "online")
    exclusions.groupby(["subject", "session", "phase"]).finite.agg(total="size", retained="sum").assign(
        excluded=lambda f: f.total-f.retained).to_csv(out/"exclusions.csv")
    frames = PredictionStore(art/"prediction_partitions")
    records, scheduling = run_jobs(analysis_jobs(cfg), cfg, art, selection, frames.directory)
    reproduction = dict(accuracy=None, undefended_accuracy=None, gap=None, paper_accuracy=.024,
                        tolerance=.05, protocol="S1-fitted defense and OAS tangent reference; default unscaled RBF SVC S1 -> S2")
    diagnostics = []
    for record in records:
        diagnostics.extend(record["diagnostics"])
        if record["prediction_path"] is None:
            continue
        tag = tuple(record["tags"][key] for key in TAGS)
        frames.entries.setdefault(tag, []).append(Path(record["prediction_path"]))
        if record["job"]["kind"] == "benchmark":
            prediction = joblib.load(record["prediction_path"])
            defended = record["job"]["key"] == "G"
            accuracy = float(accuracy_score(prediction.subject, prediction.prediction))
            reproduction["accuracy" if defended else "undefended_accuracy"] = accuracy
            if defended:
                reproduction["gap"] = accuracy-.024
            prediction.to_csv(out/("reproduction_predictions.csv" if defended else "undefended_reference_predictions.csv"), index=False)
    write_json(out/"execution.json", dict(scheduling=scheduling, preparation=preparation,
                                         jobs=[{k:v for k,v in r.items() if k != "diagnostics"} for r in records]))
    write_json(out/"reproduction.json", reproduction)
    decision = make_decision(frames, task, head, reproduction, cfg)
    write_json(out/"decision.json", decision)
    write_tables(frames, task, head, controls, timings, diagnostics, cfg, out)
    return decision


def transformed_covariances(defense, trials):
    # Frozen maps are trialwise: chunk before expanding float32 EEG to float64.
    chunks = []
    for start in range(0, len(trials), 200):
        chunk = trials.subset(np.arange(start, min(start+200, len(trials))))
        chunks.append(covariances(defense.transform(chunk)).X)
    return trials.with_X(np.concatenate(chunks))


def verify_seal(commit, cfg):
    """Require the exact committed plan, selections, sources and approval."""
    if not commit or cfg["attacker_status"] != "approved":
        raise LeakageError("Seal 3 commit and Lead-approved attacker settings are required")
    paths = ["docs/analysis_plan.md", "configs/exp_l.yaml", "results/exp_l/selection.json",
             "results/exp_l/runtime_projection.json", "beat/exp_l.py", "beat/lee2019.py",
             "beat/models.py", "beat/attacks.py", "beat/features.py", "beat/exp_g.py",
             "beat/exp_b.py", "beat/exp_c.py", "beat/stats.py", "beat/data.py"]
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT)
    try:
        oid = git("rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
        git("merge-base", "--is-ancestor", oid, "HEAD")
        sealed_cfg = yaml.safe_load(git("show", f"{oid}:configs/exp_l.yaml"))
        if sealed_cfg["attacker_status"] != "approved":
            raise LeakageError("Committed config lacks Lead approval")
        if sealed_cfg != cfg:
            raise LeakageError("Supplied config differs from committed config")
        for path in paths:
            if git("show", f"{oid}:{path}") != (ROOT / path).read_bytes():
                raise LeakageError(f"Sealed input differs: {path}")
        log = git("show", f"{oid}:docs/decision_log.md").decode()
        if "seal 3" not in log.lower():
            raise LeakageError("Commit does not record seal 3 in decision_log")
    except subprocess.CalledProcessError as exc:
        raise LeakageError("Unable to verify seal 3 commit") from exc
    return oid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["dev", "timing", "final"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_l.yaml")
    parser.add_argument("--seal-commit")
    parser.add_argument("--result-dir", type=Path, default=RESULTS)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not 1 <= cfg["workers"] <= 6:
        raise ValueError("Shared CPU limit: at most 6 workers")
    if args.stage in ("dev", "timing"):
        with session1_only() as audit:
            signals = load_trials(ROOT / cfg["data_dir"], cfg["subjects"], 1)
            if args.stage == "dev":
                run_dev(signals, cfg, args.result_dir, args.artifact_dir)
            else:
                selection = json.loads((args.result_dir/"selection.json").read_text(encoding="utf-8"))
                if selection["config_hash"] != digest(analysis_config(cfg)):
                    raise ValueError("Timing configuration differs from saved selections")
                run_timing(signals, cfg, selection, args.result_dir, args.artifact_dir)
        write_json(args.result_dir / "session_access.json",
                   dict(opened_session1=sorted(set(audit["opened"])),
                        blocked_session2=audit["blocked"], session2_opened=False,
                        guard="CPython open audit hook; workers receive only in-memory T or fold artifacts"))
    else:
        if args.result_dir.resolve() != RESULTS or args.artifact_dir.resolve() != ARTIFACTS:
            raise LeakageError("Real final must use the sealed result/artifact paths")
        seal = verify_seal(args.seal_commit, cfg)
        selection = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
        if selection["config_hash"] != digest(analysis_config(cfg)):
            raise LeakageError("Selection/config mismatch before E access")
        training = load_trials(ROOT / cfg["data_dir"], cfg["subjects"], 1)
        evaluation = load_trials(ROOT / cfg["data_dir"], cfg["subjects"], 2, allow_session2=True)
        try:
            run_final(training, evaluation, cfg, selection)
        except (WorkerTimeout, ValueError, FloatingPointError, MemoryError) as exc:
            write_json(RESULTS / "final_failure.json",
                       dict(error=type(exc).__name__, reason=str(exc),
                            rule="No reselection; incomplete claims remain not evaluable",
                            claims={key: dict(p=1., decision="not_evaluable_computational_failure")
                                    for key in ("C1", "C2", "C3", "C4")}))
            raise
        write_json(RESULTS / "seal.json", dict(commit=seal))


if __name__ == "__main__":
    # Spawned workers and persisted objects must resolve an importable module.
    from beat.exp_l import main as imported_main
    imported_main()
