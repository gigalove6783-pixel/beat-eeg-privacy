"""Experiment R (EXPLORATORY): robustness addendum on the sealed Lee2019 replication L.

Plan v2 revision 4, lane R. Descriptive only; no claim of L is re-decided.

MLP sensitivity. Every confirmatory MLP attacker fit of L that stopped at its fixed budget is
refit through L's own code path (exp_l.execute_job -> attack_stream -> attack), so inputs,
splits, finite-row filtering, grouping, scaling and seeds are L's; only the attacker fit
(exp_l._predict_attackers) is replaced, inside this process, by an MLP-only fit with another
budget. Each target is also refit at L's budget and compared with L's stored fit. L's
decision function (exp_l.make_decision) is then rerun on L's stored predictions with the
refitted MLP predictions substituted.

Cross-day table. Within-day (main: session-2 offline <-> online phase) and cross-day
(auxiliary: session 1 -> session 2) identity BA of P0, P2-RC d8, P3 lambda 1 and the
co-trained head, read from L's result files.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
import warnings
from collections import Counter
from pathlib import Path, PureWindowsPath
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import exp_l
from .attacks import ATTACKERS
from .common import write_csv, write_json
from .config import ROOT

RESULTS = ROOT / "results/exp_r"
CHECKPOINTS = RESULTS / "refits"
SEALED = exp_l.RESULTS
CONDITIONS = ("P0", "P2-RC_d8", "P3_lambda1")
FIT_KEY = ("kind", "key", "mode", "tau", "k", "enrolment", "enrolled", "scenario", "direction",
           "seed")
ROW = ["subject", "run", "trial"]
SCENARIOS = {"main": "within_day", "auxiliary": "cross_day"}
LABEL = "EXPLORATORY (plan v2 lane R): descriptive addendum; the sealed decisions of L stand"


def load_cfg(path=None):
    return yaml.safe_load(Path(path or ROOT / "configs/exp_r.yaml").read_text(encoding="utf-8"))


def _read_sealed(name):
    return json.loads((SEALED / name).read_text(encoding="utf-8"))


def claim_of(job):
    """Claim of L whose sealed decision uses this job's predictions (None: descriptive job)."""
    if job["kind"] == "representation" and job["key"] in ("P0", "P2-RC_d8"):
        return "C2"
    if job["kind"] == "representation" and job["key"] == "P3_lambda1":
        return "C1"
    if ((job["kind"], job["key"], job["mode"], job["tau"]) == ("observation", "P2-RC_d8",
                                                                "command", .5)
            and job["k"] in (1, 20)):
        return "C3"
    if (job["kind"], job["key"], job["enrolment"]) == ("enrolment", "G", "5"):
        return "C4"
    return None


def fit_key(record):
    return tuple(record[name] for name in FIT_KEY)


def confirmatory_jobs(lcfg):
    return [job for job in exp_l.analysis_jobs(lcfg) if claim_of(job)]


def sealed_state():
    """L's config, selections, job records and MLP convergence records (all read-only)."""
    lcfg = yaml.safe_load((ROOT / "configs/exp_l.yaml").read_text(encoding="utf-8"))
    execution = _read_sealed("execution.json")
    if [record["job"] for record in execution["jobs"]] != exp_l.analysis_jobs(lcfg):
        raise ValueError("results/exp_l/execution.json does not match exp_l.analysis_jobs")
    convergence = {fit_key(r): r for r in _read_sealed("attacker_diagnostics.json")
                   if r.get("record") == "convergence"}
    seal = exp_l.verify_seal(_read_sealed("seal.json")["commit"], lcfg)
    return dict(cfg=lcfg, selection=_read_sealed("selection.json"), execution=execution,
                convergence=convergence, seal=seal)


def targets(state):
    """Confirmatory jobs whose sealed MLP fit did not converge, in L's job order."""
    return [job for job in confirmatory_jobs(state["cfg"])
            if not state["convergence"][fit_key(job)]["converged"]]


def mlp_only(budget):
    """Stand-in for exp_l._predict_attackers: L's MLP pipeline alone, with another budget."""
    def fit(A, subjects, B, cfg, seed):
        c = cfg["attackers"]
        model = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=tuple(c["mlp_hidden"]), alpha=c["mlp_alpha"],
            solver=c["mlp_solver"], max_iter=budget["max_iter"], max_fun=budget["max_fun"],
            random_state=seed))
        start = time.perf_counter()
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            model.fit(A, subjects)
            prediction = model.predict(B)
        messages = [str(w.message) for w in observed if issubclass(w.category, ConvergenceWarning)]
        other = [str(w.message) for w in observed
                 if not issubclass(w.category, ConvergenceWarning)]
        net = model.steps[-1][1]
        return {"mlp": prediction}, {"mlp": dict(
            converged=not messages, iterations=int(net.n_iter_), loss=float(net.loss_),
            warning=" | ".join(messages), other_warnings=" | ".join(other),
            max_iter=budget["max_iter"], max_fun=budget["max_fun"],
            seconds=time.perf_counter() - start)}
    return fit


def refit(job, state, budget):
    """One L job through L's code path with only the MLP fitted: (rows, fit record, job record)."""
    with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(exp_l, "_predict_attackers", mlp_only(budget)):
        record = exp_l.execute_job(job, state["cfg"], exp_l.ARTIFACTS, state["selection"], tmp)
        fits = [d for d in record["diagnostics"] if d.get("record") == "convergence"]
        if record["status"] != "ok" or len(fits) != 1:
            raise RuntimeError(f"Refit of L job {job['id']} failed: {record['diagnostics'][-1:]}")
        frame = joblib.load(record["prediction_path"])
    return frame, fits[0], record


def sealed_frame(job, state):
    """L's stored predictions (all attackers) of one job."""
    name = PureWindowsPath(state["execution"]["jobs"][job["id"]]["prediction_path"]).name
    return joblib.load(exp_l.ARTIFACTS / "prediction_partitions" / name)


def aligned(old, new):
    """Two prediction sets of the same job, matched per test row."""
    pair = old[ROW + ["prediction"]].merge(new[ROW + ["prediction"]], on=ROW, validate="one_to_one",
                                           suffixes=("_sealed", "_refit"))
    if not len(pair) == len(old) == len(new):
        raise ValueError("Refit and sealed predictions cover different test rows")
    return pair


def job_ba(frame):
    return float(exp_l.recalls(frame).mean())


def guard(cfg, deadline):
    """Stop at the lane's compute cap; wait (bounded) while free memory is below the reserve."""
    if time.monotonic() > deadline:
        raise TimeoutError("Lane R compute limit reached before the next fit")
    reserve, until = cfg["memory"]["reserve_mib"], time.monotonic() + cfg["memory"]["wait_seconds"]
    while (free := exp_l.memory_snapshot()["available_mib"]) < reserve:
        if time.monotonic() > until:
            raise MemoryError(f"{free:.0f} MiB available, below the {reserve} MiB reserve")
        time.sleep(15)
    return free


def sealed_mlp(job, state):
    frame = sealed_frame(job, state)
    return frame[frame.attacker == "mlp"]


def fit_targets(jobs, state, cfg, deadline):
    """Refit every target at the configured budget, then reproduce the refitted ones at L's budget.

    A refit starts only while the longest refit so far still fits before the deadline, and a
    reproduction only while runtime.reproduction_seconds does, so the cap is never overrun;
    targets left out are recorded (refit_run false, reproduced empty) with L's predictions kept.
    """
    sealed_budget = {name: state["cfg"]["attackers"][f"mlp_{name}"]
                     for name in ("max_iter", "max_fun")}
    rows, replacements, longest = [], {}, cfg["runtime"]["first_refit_seconds"]
    for number, job in enumerate(jobs, 1):
        old, sealed_fit = sealed_mlp(job, state), state["convergence"][fit_key(job)]
        row = dict(job_id=job["id"], claim=claim_of(job), **{name: job[name] for name in FIT_KEY},
                   test_rows=len(old), sealed_max_iter=sealed_budget["max_iter"],
                   sealed_converged=sealed_fit["converged"],
                   sealed_iterations=sealed_fit["iterations"], sealed_loss=sealed_fit["loss"],
                   sealed_ba=job_ba(old), refit_run=False, refit_source=None, reproduced=None)
        rows.append(row)
        saved = load_checkpoint(job, cfg["mlp_budget"])
        if saved is not None:
            row.update(saved[0], refit_source="checkpoint")
            replacements[job["id"]] = saved[1]
            print(f"[{number}/{len(jobs)}] job {job['id']}: refit reused from checkpoint",
                  flush=True)
            continue
        if deadline - time.monotonic() < longest:
            print(f"[{number}/{len(jobs)}] job {job['id']}: not refit (compute cap)", flush=True)
            continue
        free, start = guard(cfg, deadline), time.monotonic()
        frame, fit, record = refit(job, state, cfg["mlp_budget"])
        longest = max(longest, time.monotonic() - start)
        new = aligned(old, frame)
        row.update(refit_run=True, refit_source="this invocation",
                   refit_max_iter=fit["max_iter"], refit_max_fun=fit["max_fun"],
                   refit_converged=fit["converged"], refit_iterations=fit["iterations"],
                   refit_loss=fit["loss"], refit_seconds=fit["seconds"],
                   refit_warning=fit["warning"], refit_other_warnings=fit["other_warnings"],
                   refit_ba=job_ba(frame), ba_change=job_ba(frame) - row["sealed_ba"],
                   agreement=float((new.prediction_sealed == new.prediction_refit).mean()),
                   available_mib_before=free, peak_mib=record["peak_mib"])
        replacements[job["id"]] = frame
        save_checkpoint(job, cfg["mlp_budget"], row, frame)
        print(f"[{number}/{len(jobs)}] job {job['id']} {job['key']} k={job['k']} "
              f"{job['direction']} seed {job['seed']}: converged={fit['converged']} after "
              f"{fit['iterations']} it, {fit['seconds']:.1f} s, BA {row['sealed_ba']:.4f} -> "
              f"{row['refit_ba']:.4f}", flush=True)
    for job, row in zip(jobs, rows):
        if (not row["refit_run"] or row["reproduced"] is not None
                or deadline - time.monotonic() < cfg["runtime"]["reproduction_seconds"]):
            continue
        guard(cfg, deadline)
        old, sealed_fit = sealed_mlp(job, state), state["convergence"][fit_key(job)]
        repro, repro_fit, _ = refit(job, state, sealed_budget)
        same = aligned(old, repro)
        row.update(reproduced=bool((same.prediction_sealed == same.prediction_refit).all()
                                   and repro_fit["iterations"] == sealed_fit["iterations"]
                                   and repro_fit["loss"] == sealed_fit["loss"]),
                   reproduced_iterations=repro_fit["iterations"],
                   reproduced_loss=repro_fit["loss"], reproduced_seconds=repro_fit["seconds"])
        save_checkpoint(job, cfg["mlp_budget"], row)
        print(f"job {job['id']}: reproduced at L's budget = {row['reproduced']}", flush=True)
    return rows, replacements


def _checkpoint_paths(job):
    return CHECKPOINTS / f"job_{job['id']}.json", CHECKPOINTS / f"job_{job['id']}.csv.gz"


def save_checkpoint(job, budget, row, frame=None):
    """Persist a finished refit (and later its reproduction) so a rerun never repeats it."""
    meta, predictions = _checkpoint_paths(job)
    if frame is not None:
        write_csv(frame, predictions)
    write_json(meta, dict(label=LABEL, budget=budget, row=row))


def load_checkpoint(job, budget):
    """(row fields, refit predictions) of a stored refit at this budget, else None."""
    meta, predictions = _checkpoint_paths(job)
    if not meta.exists():
        return None
    saved = json.loads(meta.read_text(encoding="utf-8"))
    if saved["budget"] != budget or saved["row"]["job_id"] != job["id"]:
        return None
    return saved["row"], pd.read_csv(predictions)


class Store:
    """exp_l.PredictionStore.select over one in-memory prediction frame."""

    def __init__(self, frame):
        self.frame = frame

    def select(self, **query):
        mask = np.ones(len(self.frame), dtype=bool)
        for name, value in query.items():
            mask &= (self.frame[name] == value).to_numpy()
        return self.frame[mask].reset_index(drop=True)


def substituted(frames, replacements):
    """L's predictions with the MLP rows of each refitted job replaced by the refit."""
    parts = [pd.concat([frame[frame.attacker != "mlp"], replacements[i]], ignore_index=True)
             if i in replacements else frame for i, frame in frames.items()]
    return Store(pd.concat(parts, ignore_index=True))


def decide(store, lcfg):
    """L's sealed decision function on L's stored task, head and benchmark results."""
    return exp_l.make_decision(store, pd.read_csv(SEALED / "task.csv"),
                               pd.read_csv(SEALED / "head_predictions.csv.gz"),
                               _read_sealed("reproduction.json"), lcfg)


def component_rows(decision):
    """Every component test, named by its path in decision["claims"][claim]["component_tests"]."""
    rows = []

    def walk(claim, path, node):
        if "mean" in node:
            rows.append(dict(claim=claim, component=" | ".join(path), mean=node["mean"],
                             lower=node["lower"], p=node["p"],
                             signflip_p=node["sign_flip"]["signflip_p"]))
            return
        for name, child in node.items():
            walk(claim, [*path, name], child)

    for claim, value in decision["claims"].items():
        walk(claim, [], value["component_tests"] or {})
        if value.get("utility"):
            walk(claim, ["utility"], value["utility"])
    return rows


def claim_rows(decision):
    return [dict(claim=claim, observed_difference=value.get("observed_difference"),
                 effect_requirement=value.get("effect_requirement"), p=value["p"],
                 holm_p=value["holm_p"], decision=value["decision"])
            for claim, value in decision["claims"].items()]


def side_by_side(sealed, refit, keys):
    """Sealed and refit values in adjacent columns, numeric changes appended."""
    merged = pd.DataFrame(sealed).merge(pd.DataFrame(refit), on=keys, validate="one_to_one",
                                        suffixes=("_sealed", "_refit"))
    values = [c for c in pd.DataFrame(sealed).columns if c not in keys]
    columns = list(keys)
    for value in values:
        columns += [f"{value}_sealed", f"{value}_refit"]
        a, b = merged[f"{value}_sealed"], merged[f"{value}_refit"]
        if pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a):
            merged[f"{value}_change"] = b - a
            columns.append(f"{value}_change")
        else:
            merged[f"{value}_changed"] = [not _same(x, y) for x, y in zip(a, b)]
            columns.append(f"{value}_changed")
    return merged[columns]


def _same(x, y):
    """Equality of decision labels where missing equals missing (None, NaN or pd.NA)."""
    return (pd.isna(x) and pd.isna(y)) or (not pd.isna(x) and not pd.isna(y) and x == y)


def max_deviation(first, second):
    """Largest absolute difference between two decisions over all claim and component values."""
    a, b = pd.DataFrame(component_rows(first)), pd.DataFrame(component_rows(second))
    if not a[["claim", "component"]].equals(b[["claim", "component"]]):
        raise ValueError("Decisions have different component tests")
    ca, cb = pd.DataFrame(claim_rows(first)), pd.DataFrame(claim_rows(second))
    numbers = [(a[c], b[c]) for c in ("mean", "lower", "p", "signflip_p")]
    numbers += [(pd.to_numeric(ca[c]), pd.to_numeric(cb[c]))
                for c in ("observed_difference", "p", "holm_p")]
    return (float(max((x - y).abs().max() for x, y in numbers)),
            all(_same(x, y) for c in ("decision", "effect_requirement")
                for x, y in zip(ca[c], cb[c])))


def candidate_rows(store, lcfg):
    """Mean participant recall of every candidate vector entering C1-C3 (as in make_decision)."""
    inputs = {(key, "C1" if key == "P3_lambda1" else "C2"):
              store.select(kind="representation", key=key) for key in CONDITIONS}
    commands = store.select(kind="observation", key="P2-RC_d8", mode="command", tau=.5)
    for k in (1, 20):
        inputs[(f"P2-RC_d8 command tau=0.5 k={k}", "C3")] = commands[commands.k == k]
    rows = []
    for (name, claim), frame in inputs.items():
        vectors, missing = exp_l.candidate_vectors(frame, lcfg)
        if missing:
            raise ValueError(f"Incomplete candidate cells for {name}: {missing[:1]}")
        for candidate, vector in vectors.items():
            attacker, scenario = candidate.split("/")
            rows.append(dict(claim=claim, input=name, attacker=attacker, scenario=scenario,
                             scenario_label=SCENARIOS[scenario], ba=float(vector.mean())))
    return rows


def strongest(rows):
    """Largest candidate per claim input (the maximum the sealed gaps use)."""
    frame = pd.DataFrame(rows)
    best = frame.loc[frame.groupby("input", sort=False).ba.idxmax()]
    return {row.input: dict(candidate=f"{row.attacker}/{row.scenario}", ba=row.ba)
            for row in best.itertuples()}


def conclusion(claims, fits):
    converged, refitted = int(fits.refit_converged.eq(True).sum()), int(fits.refit_run.sum())
    checked = int(fits.reproduced.notna().sum())
    changed = claims[claims.decision_changed]
    effect = claims[claims.effect_requirement_changed]
    gaps = ", ".join(f"{row.claim} {row.observed_difference_sealed:.4f} -> "
                     f"{row.observed_difference_refit:.4f}"
                     for row in claims.itertuples() if row.claim in ("C1", "C2"))
    status = ("no Holm decision of L changes" if changed.empty else
              "Holm decisions change for " + ", ".join(
                  f"{r.claim} ({r.decision_sealed} -> {r.decision_refit})"
                  for r in changed.itertuples()))
    text = (f"Exploratory sensitivity: {refitted} of {len(fits)} target MLP fits refit, "
            f"{converged} of them converged ({int(fits.reproduced.eq(True).sum())} of {checked} "
            f"checked sealed fits reproduced exactly at L's budget). With the refitted MLP "
            f"predictions substituted, {status}; the >= 20 pp effect requirement "
            f"{'is unchanged' if effect.empty else 'changes for ' + ', '.join(effect.claim)}; "
            f"gaps {gaps}.")
    stopped = fits[fits.refit_run.eq(True) & ~fits.refit_converged.eq(True)]
    if not stopped.empty:
        named = "; ".join(f"{claim} jobs {', '.join(map(str, part.job_id))}"
                          for claim, part in stopped.groupby("claim"))
        text += (f" {len(stopped)} refits ({named}) still stopped at the larger budget without "
                 f"converging; their larger-budget predictions are substituted, so the "
                 f"sensitivity is not fully convergent for those fits.")
    for row in claims[~claims.refits_complete].itertuples():
        text += (f" {row.claim} is incomplete: {int(row.refits_run)} of {int(row.targets)} "
                 f"targets refit, the rest keep L's predictions, so its recomputed decision is "
                 f"not a full sensitivity result.")
    return text


def pilot(cfg):
    """Time one target of every type at the configured budget and project the full run."""
    start = time.monotonic()
    state = sealed_state()
    jobs = targets(state)
    kind = lambda job: (job["key"], job["mode"], job["k"], job["scenario"])
    counts, first = Counter(kind(job) for job in jobs), {}
    for job in jobs:
        first.setdefault(kind(job), job)
    chosen = list(first.values())
    rows, _ = fit_targets(chosen, state, cfg, start + cfg["runtime"]["limit_seconds"])
    per_type = [dict(type="/".join(map(str, kind(job))), count=counts[kind(job)], **row)
                for job, row in zip(chosen, rows)]
    projected = sum(r["count"] * (r.get("reproduced_seconds", 0) + r.get("refit_seconds", 0))
                    for r in per_type)
    write_json(RESULTS / "pilot.json", dict(
        label=LABEL, budget=cfg["mlp_budget"], targets=len(jobs), types=per_type,
        projected_fit_seconds=projected, pilot_wall_seconds=time.monotonic() - start,
        limit_seconds=cfg["runtime"]["limit_seconds"],
        basis="One fit per (key, mode, k, scenario) type at the configured budget plus its "
              "reproduction at L's budget, multiplied by the type's target count"))
    return projected


def run(cfg):
    start = time.monotonic()
    state = sealed_state()
    lcfg = state["cfg"]
    jobs, chosen = confirmatory_jobs(lcfg), targets(state)
    rows, replacements = fit_targets(chosen, state, cfg, start + cfg["runtime"]["limit_seconds"])
    fits = pd.DataFrame(rows)
    frames = {job["id"]: sealed_frame(job, state) for job in jobs}
    sealed_store, refit_store = Store(pd.concat(frames.values(), ignore_index=True)), substituted(
        frames, replacements)
    recorded = _read_sealed("decision.json")
    recomputed, refitted = decide(sealed_store, lcfg), decide(refit_store, lcfg)
    deviation, same_decisions = max_deviation(recorded, recomputed)
    if deviation > 1e-9 or not same_decisions:
        raise ValueError(f"Recomputed sealed decision differs from decision.json ({deviation})")
    claims = side_by_side(claim_rows(recomputed), claim_rows(refitted), ["claim"]).merge(
        fits.groupby("claim").agg(targets=("job_id", "size"), refits_run=("refit_run", "sum"),
                                  refits_converged=("refit_converged",
                                                    lambda s: int(s.eq(True).sum())))
        .reset_index(), on="claim", how="left")
    claims["refits_complete"] = claims.refits_run.fillna(0).eq(claims.targets.fillna(0))
    components = side_by_side(component_rows(recomputed), component_rows(refitted),
                              ["claim", "component"])
    sealed_candidates, refit_candidates = (candidate_rows(sealed_store, lcfg),
                                           candidate_rows(refit_store, lcfg))
    candidates = side_by_side(sealed_candidates, refit_candidates,
                              ["claim", "input", "attacker", "scenario", "scenario_label"])
    # *_refit columns (decision_refit included) are L's sealed decision function rerun on the
    # refitted predictions as an exploratory sensitivity, not a new decision on L's claims.
    for name, table in (("mlp_refits", fits), ("mlp_sensitivity_claims", claims),
                        ("mlp_sensitivity_components", components),
                        ("mlp_sensitivity_candidates", candidates)):
        write_csv(table.assign(label=LABEL)[["label", *table.columns]],
                  RESULTS / f"{name}.csv")
    previous_path = RESULTS / "mlp_sensitivity.json"
    previous = json.loads(previous_path.read_text("utf-8")) if previous_path.exists() else {}
    reused = int(fits.refit_source.eq("checkpoint").sum())
    fitted = int(fits.refit_source.eq("this invocation").sum())
    invocation_seconds = time.monotonic() - start
    write_json(RESULTS / "mlp_sensitivity.json", dict(
        label=LABEL, plan=".mixdog/plan_v2.md revision 4, lane R; acceptance A5",
        seal=dict(commit=state["seal"], verified_by="beat.exp_l.verify_seal"),
        sealed_rule=lcfg["failure_rules"]["nonconvergence"],
        sealed_budget=dict(max_iter=lcfg["attackers"]["mlp_max_iter"],
                           max_fun=lcfg["attackers"]["mlp_max_fun"]),
        refit_budget=cfg["mlp_budget"],
        targets=dict(total=len(chosen), by_claim=dict(Counter(fits.claim)),
                     confirmatory_mlp_fits=len(jobs), rule="sealed convergence record false"),
        refits=dict(run=int(fits.refit_run.sum()),
                    converged=int(fits.refit_converged.eq(True).sum()),
                    reproduction_checked=int(fits.reproduced.notna().sum()),
                    reproduced_at_sealed_budget=int(fits.reproduced.eq(True).sum()),
                    iterations_min=int(fits.refit_iterations.min()),
                    iterations_max=int(fits.refit_iterations.max()),
                    fit_seconds=float(fits.refit_seconds.sum()
                                      + fits.get("reproduced_seconds", pd.Series(dtype=float)).sum()),
                    peak_mib=float(fits.peak_mib.max()),
                    minimum_available_mib=float(fits.available_mib_before.min())),
        recomputation=dict(max_abs_deviation_from_decision_json=deviation,
                           decisions_equal=same_decisions),
        strongest=dict(sealed=strongest(sealed_candidates), refit=strongest(refit_candidates)),
        claims=claims.astype(object).where(claims.notna(), None).to_dict("records"),
        conclusion=conclusion(claims, fits),
        component_names={
            "C1": "<P3_lambda1 candidate>: independent recall - co-trained head recall > 0",
            "C2": "<P0 candidate> | <P2-RC_d8 candidate>: P0 - P2-RC_d8 recall > 0; utility: "
                  "P2-RC_d8 - P0 task BA > -0.05",
            "C3": "<k=20 candidate> | threshold: recall > 2/54; <k=20 candidate> | "
                  "differences | <k=1 candidate>: k=20 - k=1 recall > 0",
            "C4": "<G n=5 candidate>: recall > 10/54 (no C4 fit was refit)"},
        method=__doc__.split("\n\n")[2].replace("\n", " "),
        invocation_wall_seconds=invocation_seconds,
        invocation_refits=dict(reused_from_checkpoints=reused, fitted=fitted),
        first_run_wall_seconds=invocation_seconds if fitted else previous.get(
            "first_run_wall_seconds", cfg["runtime"]["recorded_first_run_wall_seconds"]),
        first_run_basis="wall seconds of the invocation that fitted the refits (later "
                        "invocations reuse its checkpoints and keep this value)"))
    cross_day(lcfg)
    print(conclusion(claims, fits), flush=True)


def cross_day(lcfg=None):
    """Within-day vs cross-day identity BA of L's confirmatory conditions and the head."""
    lcfg = lcfg or yaml.safe_load((ROOT / "configs/exp_l.yaml").read_text(encoding="utf-8"))
    summary = pd.read_csv(SEALED / "identity_summary.csv")
    rep = summary[(summary.kind == "representation") & summary.key.isin(CONDITIONS)]
    if (rep.status != "ok").any() or len(rep) != len(CONDITIONS) * len(ATTACKERS) * 2:
        raise ValueError("identity_summary.csv lacks complete confirmatory representation rows")
    task = pd.read_csv(SEALED / "task_summary.csv").set_index("key")
    heads = pd.read_csv(SEALED / "head_predictions.csv.gz")
    head = heads[heads.key == "P3_lambda1"]
    subjects, seeds = sorted(lcfg["subjects"]), lcfg["seeds"]
    head_ba = float(exp_l.recalls(head).reindex(subjects).mean())
    head_ci = exp_l.interval(head, lcfg, subjects, seeds, ["offline_to_online", "online_to_offline"])
    source = ("results/exp_l/identity_summary.csv (kind=representation, key={key}, 5 seeds; "
              "ba = mean participant recall, CI = participant bootstrap)")
    rows, best = [], {}
    for key in CONDITIONS:
        part = rep[rep.key == key]
        for attacker in ("strongest", *ATTACKERS):
            row = dict(condition=key, system="independent attacker", attacker=attacker)
            for scenario, label in SCENARIOS.items():
                cell = part[part.scenario == scenario]
                pick = (cell.loc[cell.ba.idxmax()] if attacker == "strongest"
                        else cell[cell.attacker == attacker].iloc[0])
                row.update({f"{label}_ba": pick.ba, f"{label}_ci_low": pick.ci_low,
                            f"{label}_ci_high": pick.ci_high, f"{label}_attacker": pick.attacker})
                if attacker == "strongest":
                    best[key, label] = pick.ba
            rows.append(dict(row, task_ba=task.loc[key, "mean"] if attacker == "strongest" else None,
                             source=source.format(key=key) + (
                                 "; task_ba: results/exp_l/task_summary.csv (decoder fitted on "
                                 "session 1, tested on session 2)" if attacker == "strongest"
                                 else ""),
                             note="strongest = maximum over the three attackers within the scenario"
                             if attacker == "strongest" else ""))
    rows.append(dict(condition="P3_lambda1", system="co-trained identity head", attacker="head",
                     cross_day_ba=head_ba, cross_day_ci_low=head_ci[0], cross_day_ci_high=head_ci[1],
                     source="results/exp_l/head_predictions.csv.gz (key=P3_lambda1, seeds 0-4; "
                            "beat.exp_l.recalls and beat.exp_l.interval)",
                     note="The head is trained with the model on session 1 only and scored on all "
                          "session-2 trials, so only a cross-day value exists"))
    rows.append(dict(condition="P3_lambda1", system="strongest independent - co-trained head",
                     attacker="difference",
                     within_day_ba=best["P3_lambda1", "within_day"] - head_ba,
                     cross_day_ba=best["P3_lambda1", "cross_day"] - head_ba,
                     source="rows above", note="C1 tested the maximum over both scenarios against "
                     "the head (results/exp_l/decision.json)"))
    rows.append(dict(condition="P0 - P2-RC_d8", system="strongest independent: P0 - P2-RC_d8",
                     attacker="difference",
                     within_day_ba=best["P0", "within_day"] - best["P2-RC_d8", "within_day"],
                     cross_day_ba=best["P0", "cross_day"] - best["P2-RC_d8", "cross_day"],
                     source="rows above", note="C2 tested the maximum over both scenarios "
                     "(results/exp_l/decision.json)"))
    table = pd.DataFrame(rows)
    table["within_minus_cross"] = table.within_day_ba - table.cross_day_ba
    # Every value comes from L's sealed seal-3 outputs (the auxiliary/cross-day ones are
    # pre-registered estimates without a verdict of their own); only R's MLP refits are exploratory.
    sealed = "sealed L seal-3 output (pre-registered estimate)"
    table.insert(0, "label", [
        "derived from sealed L seal-3 outputs (difference of rows above)"
        if row.attacker == "difference" else
        "computed from sealed L head predictions (results/exp_l/head_predictions.csv.gz) with "
        "beat.exp_l.recalls and beat.exp_l.interval; the CI is not stored in L's files"
        if row.attacker == "head" else
        sealed + "; within_minus_cross derived here (within_day_ba - cross_day_ba)"
        if pd.notna(row.within_minus_cross) else sealed for row in table.itertuples()])
    write_csv(table, RESULTS / "cross_day_table.csv")
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pilot", "run", "crossday"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp_r.yaml")
    args = parser.parse_args(argv)
    cfg = load_cfg(args.config)
    if cfg["status"] != "exploratory":
        raise ValueError("Experiment R is exploratory only")
    if args.stage == "pilot":
        print(f"Projected fit seconds: {pilot(cfg):.0f}", flush=True)
    elif args.stage == "run":
        run(cfg)
    else:
        cross_day()


if __name__ == "__main__":
    main()
