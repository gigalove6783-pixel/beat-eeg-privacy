"""Export compact, licence-safe JSON for the static demo in demo/.

Reads result tables (results/) and per-trial replay/attack artifacts (artifacts/exp_c/)
and writes demo/data/. Only aggregated results, decoded commands (left/right/hold),
trial labels and attacker identity predictions are written: no EEG, covariances,
features, representations or logits (BNCI2014-001 is CC BY-ND 4.0).

Experiments D and E are picked up automatically once their final tables exist under
results/exp_d/ or results/exp_e/ with the same file names as B/C (tradeoff.csv,
adversary_comparison.csv, observation_strongest.csv). Extra columns are passed through.

Usage:  .venv\\Scripts\\python scripts\\export_demo_data.py
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS, ARTIFACTS = ROOT / "results", ROOT / "artifacts"
OUT = ROOT / "demo" / "data"
EPISODE = 20
CMD = {-1: "L", 0: "H", 1: "R"}
KS = (1, 5, 10, 20)
E_HALVES = (((1, 2, 3), (4, 5, 6)), ((4, 5, 6), (1, 2, 3)))  # beat/splits.py
# Every experiment whose final tables may feed the generic charts. A missing table
# marks the experiment as pending; nothing else needs to change when it lands.
EXPERIMENTS = {
    "exp_b": "B: pooled models, identity adversary",
    "exp_c": "C: replayed arm, observation attack",
    "exp_d": "D: bottleneck-dimension sweep, stronger adversaries",
    "exp_e": "E: EEGNet",
}
GENERIC_TABLES = {"tradeoff": "tradeoff.csv", "adversary": "adversary_comparison.csv",
                  "observation": "observation_strongest.csv"}
# Columns that would carry per-trial signal content; refuse to export them.
FORBIDDEN = re.compile(r"representation|feature_\d|logit|cov|eeg|^x\d", re.I)

sources: dict[str, str] = {}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def track(path: Path) -> Path:
    sources[rel(path)] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(track(path))


def read_json(path: Path):
    return json.loads(track(path).read_text(encoding="utf-8"))


def clean(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) else round(float(value), 6)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def records(frame: pd.DataFrame, **extra) -> list[dict]:
    bad = [c for c in frame.columns if FORBIDDEN.search(str(c))]
    if bad:
        raise ValueError(f"Refusing to export per-trial signal columns: {bad}")
    return [dict(extra, **{k: clean(v) for k, v in row.items()})
            for row in frame.to_dict(orient="records")]


def src(path: str, **where) -> dict:
    return dict(file=path, **({"where": where} if where else {}))


# ---------------------------------------------------------------- generic tables

def generic():
    data = {name: [] for name in GENERIC_TABLES}
    status = {}
    for exp, title in EXPERIMENTS.items():
        found = []
        for name, file in GENERIC_TABLES.items():
            path = RESULTS / exp / file
            if path.exists():
                frame = read_csv(path)
                if name == "observation":
                    frame = frame[[c for c in frame.columns if not c.startswith("macro_f1")]]
                data[name].extend(records(frame, experiment=exp))
                found.append(rel(path))
        status[exp] = dict(title=title, status="available" if found else "pending",
                           tables=found)
    return data, status


# ---------------------------------------------------------------- headline numbers

def headline(tradeoff, adversary, observation, chance):
    t = pd.DataFrame([r for r in tradeoff if r["experiment"] == "exp_b"])
    a = pd.DataFrame([r for r in adversary if r["experiment"] == "exp_b"])
    o = pd.DataFrame([r for r in observation if r["experiment"] == "exp_c"])
    cell = lambda c: t[(t.condition == c)].iloc[0]
    ctrl = read_csv(RESULTS / "exp_c" / "control_summary.csv")
    ctrl = ctrl[ctrl["mode"] == "command"]
    thr = read_csv(RESULTS / "exp_c" / "observation_threshold.csv")
    cmd_thr = thr[thr["mode"] == "command"]
    timing = read_csv(RESULTS / "exp_c" / "timing_summary.csv")
    trans = read_csv(RESULTS / "exp_c" / "transmission_check.csv")
    p3 = a[a.condition == "P3"]
    cmd = o[o["mode"] == "command"]
    h1 = read_json(RESULTS / "exp_a" / "h1.json")
    dec = read_json(RESULTS / "exp_b" / "decision.json")
    sealed = read_json(RESULTS / "exp_c" / "sealed_gate" / "summary.json")
    rej = read_csv(RESULTS / "exp_c" / "sealed_gate" / "rejection_rates.csv")
    rej_e = rej[rej.session == "E"]
    worst = rej_e.loc[rej_e.hold_rate.idxmax()]
    lam1 = p3[p3.target == 1.0].iloc[0]
    tb, tc = "results/exp_b/tradeoff.csv", "results/exp_c/observation_strongest.csv"
    return {
        "raw_identity": dict(value=clean(cell("P0").identity_ba), source=src(tb, condition="P0")),
        "raw_rc_identity": dict(value=clean(cell("P0-RC").identity_ba),
                                source=src(tb, condition="P0-RC")),
        "bottleneck_identity": dict(value=clean(cell("P2").identity_ba),
                                    source=src(tb, condition="P2")),
        "bottleneck_rc_identity": dict(value=clean(cell("P2-RC").identity_ba),
                                       source=src(tb, condition="P2-RC")),
        "raw_task": dict(value=clean(cell("P0").task_ba), source=src(tb, condition="P0")),
        "bottleneck_rc_task": dict(value=clean(cell("P2-RC").task_ba),
                                   source=src(tb, condition="P2-RC")),
        "head_lambda1": dict(value=clean(lam1.adversary_main_ba),
                             source=src("results/exp_b/adversary_comparison.csv",
                                        condition="P3", target=1.0)),
        "independent_lambda1": dict(value=clean(lam1.independent_ba),
                                    source=src("results/exp_b/adversary_comparison.csv",
                                               condition="P3", target=1.0)),
        "independent_max": dict(value=clean(p3.independent_ba.max()),
                                source=src("results/exp_b/adversary_comparison.csv",
                                           condition="P3")),
        "selected_lambda": dict(value=clean(p3[p3.selected].target.iloc[0]),
                                source=src("results/exp_b/adversary_comparison.csv",
                                           selected=True)),
        "h2_accepted": dict(value=bool(dec["h2_accepted"]), source=src("results/exp_b/decision.json")),
        "cmd_k1_range": dict(value=[clean(cmd[cmd.k == 1].ba_mean.min()),
                                    clean(cmd[cmd.k == 1].ba_mean.max())],
                             source=src(tc, mode="command", k=1)),
        "cmd_k20_range": dict(value=[clean(cmd[cmd.k == 20].ba_mean.min()),
                                     clean(cmd[cmd.k == 20].ba_mean.max())],
                              source=src(tc, mode="command", k=20)),
        "cmd_within5": dict(value=[int((cmd_thr.smallest_k <= 5).sum()), int(len(cmd_thr))],
                            source=src("results/exp_c/observation_threshold.csv",
                                       mode="command", rule="smallest_k <= 5")),
        "chance": dict(value=clean(chance), source=src("configs/exp_c.yaml", key="observation.chance")),
        "h1": dict(value=dict(b3=clean(np.mean([v["B3"] for v in h1["per_subject"].values()])),
                              b2=clean(np.mean([v["B2"] for v in h1["per_subject"].values()])),
                              diff=clean(h1["mean_difference"]), positive=h1["n_positive"],
                              n=h1["n_subjects"], p=clean(h1["p_value"])),
                   source=src("results/exp_a/h1.json")),
        "mae_tau05": dict(value=[clean(ctrl[ctrl.tau == 0.5].mae_deg_mean.min()),
                                 clean(ctrl[ctrl.tau == 0.5].mae_deg_mean.max())],
                          source=src("results/exp_c/control_summary.csv", tau=0.5)),
        "hold_tau08": dict(value=[clean(ctrl[ctrl.tau == 0.8].hold_rate_mean.min()),
                                  clean(ctrl[ctrl.tau == 0.8].hold_rate_mean.max())],
                           source=src("results/exp_c/control_summary.csv", tau=0.8)),
        "transmission": dict(value=dict(passed=int(trans.passed.sum()), settings=int(len(trans)),
                                        mismatches=int(trans.command_mismatches.sum())),
                             source=src("results/exp_c/transmission_check.csv")),
        "timing_median_ms": dict(value=[clean(timing.total_median_ms.min()),
                                        clean(timing.total_median_ms.max())],
                                 source=src("results/exp_c/timing_summary.csv")),
        "sealed_gate_e_hold": dict(value=clean(sealed["e_hold_rate_gate"]),
                                   source=src("results/exp_c/sealed_gate/summary.json")),
        "sealed_gate_t_hold": dict(value=clean(sealed["t_hold_rate_gate"]),
                                   source=src("results/exp_c/sealed_gate/summary.json")),
        "sealed_gate_worst": dict(value=dict(subject=int(worst.subject),
                                             hold=clean(worst.hold_rate)),
                                  source=src("results/exp_c/sealed_gate/rejection_rates.csv",
                                             session="E")),
        "sealed_paths_clean": dict(value=bool(h1["sealed_paths_clean"]),
                                   source=src("results/exp_a/h1.json")),
    }


def seal():
    log = track(ROOT / "docs" / "decision_log.md").read_text(encoding="utf-8")
    match = re.search(r"\*\*E seal\*\*: commit `([0-9a-f]{40})`", log)
    if not match:
        raise ValueError("Seal commit not found in docs/decision_log.md")
    return dict(commit=match.group(1), source=src("docs/decision_log.md"))


# ---------------------------------------------------------------- replay (exp C)

def labels_by_trial():
    """Left/right labels of E trials, taken from B's saved metadata (no signal arrays)."""
    path = track(ARTIFACTS / "exp_b" / "final" / "P2" / "seed0_outputs.npz")
    with np.load(path, allow_pickle=False) as z:
        meta = pd.DataFrame({k: z[k] for k in ("subject", "session", "run", "trial", "label")})
    return meta[meta.session == "E"].set_index(["subject", "run", "trial"]).label


def simulate(commands, controller):
    angle, out = 0.0, []
    for c in commands:
        proposed = angle + controller["step_deg"] * c
        if not (c != 0 and abs(proposed) > controller["range_deg"]):
            angle = proposed
        out.append(angle)
    return np.array(out)


def replay(strongest: pd.DataFrame, controller):
    trials = pd.read_csv(track(ARTIFACTS / "exp_c" / "final" / "replay_trials.csv.gz"),
                         usecols=["subject", "run", "trial", "tau", "mode", "command", "angle",
                                  "condition", "seed"])
    preds = pd.read_csv(track(ARTIFACTS / "exp_c" / "final" / "attack_predictions.csv.gz"))
    labels = labels_by_trial()
    conditions = sorted(trials.condition.unique())
    seeds = sorted(trials.seed.unique())
    taus = sorted(trials.tau.unique())
    streams = sorted(preds.stream.unique())

    # Feature and command transmission must give identical commands (results/exp_c/
    # transmission_check.csv); export the command stream once and verify.
    key = ["condition", "seed", "tau", "subject", "run", "trial"]
    cm = trials[trials["mode"] == "command"].set_index(key).sort_index()
    fm = trials[trials["mode"] == "feature"].set_index(key).sort_index()
    if not (cm.command.to_numpy() == fm.command.to_numpy()).all():
        raise ValueError("Feature and command transmission disagree")
    cmd = trials[trials["mode"] == "command"]

    attacker = {(r.condition, r.stream, int(r.k)): (r.attacker, r.scenario)
                for r in strongest.itertuples()}
    preds = preds.set_index(["condition", "stream", "k", "attacker", "scenario"]).sort_index()
    out = {}
    for subject in sorted(cmd.subject.unique()):
        s_cmd = cmd[cmd.subject == subject]
        runs = sorted(s_cmd.run.unique())
        first = s_cmd[(s_cmd.condition == conditions[0]) & (s_cmd.seed == seeds[0])
                      & (s_cmd.tau == taus[0])]
        order = {r: first[first.run == r].trial.to_numpy() for r in runs}  # recorded order
        if any(len(order[r]) < EPISODE for r in runs):
            raise ValueError(f"Subject {subject}: run shorter than one episode")
        lab = {r: "".join("R" if labels.loc[(subject, r, t)] == 1 else "L"
                          for t in order[r][:EPISODE]) for r in runs}
        commands, attacks = {}, {}
        for (c, seed, tau), g in s_cmd.groupby(["condition", "seed", "tau"], sort=True):
            per_run = []
            for r in runs:
                rows = g[g.run == r]
                if not (rows.trial.to_numpy() == order[r]).all():
                    raise ValueError("Replay trial order differs between settings")
                seq = rows.command.to_numpy()[:EPISODE]
                if not np.allclose(simulate(seq, controller), rows.angle.to_numpy()[:EPISODE]):
                    raise ValueError("Arm simulation does not match stored angles")
                per_run.append("".join(CMD[int(v)] for v in seq))
            commands.setdefault(c, {}).setdefault(str(seed), {})[f"{tau:g}"] = per_run
        for c in conditions:
            for stream in streams:
                for k in KS:
                    name, scenario = attacker[(c, stream, k)]
                    block = preds.loc[(c, stream, k, name, scenario)]
                    block = block[block.subject == subject]
                    for seed in seeds:
                        b = block[block.seed == seed]
                        per_run = []
                        for r in runs:
                            if scenario == "main":
                                direction = 0 if r in E_HALVES[0][1] else 1
                                rows = b[(b.run == r) & (b.direction == direction)]
                            else:
                                rows = b[b.run == r]
                            n = len(order[r]) // k
                            if len(rows) != n:
                                raise ValueError(f"Expected {n} groups, got {len(rows)}")
                            # Test groups are non-overlapping and in recorded order
                            # (beat/exp_c.py observation_groups); keep those inside the episode.
                            per_run.append("".join(str(int(p)) for p in
                                                   rows.prediction.to_numpy()[:EPISODE // k]))
                        attacks.setdefault(c, {}).setdefault(str(seed), {}) \
                            .setdefault(stream, {})[str(k)] = per_run
        out[int(subject)] = dict(subject=int(subject), runs=[int(r) for r in runs],
                                 labels=[lab[r] for r in runs], commands=commands,
                                 attacks=attacks)
    meta = dict(conditions=conditions, seeds=[int(s) for s in seeds],
                taus=[float(t) for t in taus], streams=streams, k=list(KS),
                episode_length=EPISODE, controller=controller,
                attacker={f"{c}|{s}|{k}": dict(attacker=a, scenario=sc)
                          for (c, s, k), (a, sc) in attacker.items()},
                note="Episode = first 20 trials of each E run; arm and target restart at 0 deg. "
                     "Attacker guesses come from non-overlapping groups of k consecutive "
                     "outputs inside the episode; attacker and scenario are the strongest "
                     "per cell in results/exp_c/observation_strongest.csv.")
    return out, meta


# ---------------------------------------------------------------- main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "replay").mkdir(exist_ok=True)
    data, status = generic()
    config_c = yaml.safe_load(track(ROOT / "configs" / "exp_c.yaml").read_text(encoding="utf-8"))
    controller = {k: config_c["controller"][k] for k in ("step_deg", "range_deg", "episode_length",
                                                         "reach_tolerance_deg")}
    control = read_csv(RESULTS / "exp_c" / "control_summary.csv")
    control = control[[c for c in control.columns
                       if not c.endswith(("_median", "_std")) or c.startswith("mae")]]
    strongest = read_csv(RESULTS / "exp_c" / "observation_strongest.csv")
    subjects, replay_meta = replay(strongest, controller)
    for s, payload in subjects.items():
        (OUT / "replay" / f"s{s}.json").write_text(json.dumps(payload, separators=(",", ":")),
                                                   encoding="utf-8")
    results = dict(
        schema=1,
        experiments=status,
        headline=headline(data["tradeoff"], data["adversary"], data["observation"],
                          config_c["observation"]["chance"]),
        seal=seal(),
        tradeoff=dict(rows=data["tradeoff"], source="results/<exp>/tradeoff.csv"),
        adversary=dict(rows=data["adversary"], source="results/<exp>/adversary_comparison.csv"),
        observation=dict(rows=data["observation"],
                         source="results/<exp>/observation_strongest.csv"),
        control=dict(rows=records(control, experiment="exp_c"),
                     source="results/exp_c/control_summary.csv"),
        replay=dict(meta=replay_meta, files={str(s): f"data/replay/s{s}.json" for s in subjects},
                    source=["artifacts/exp_c/final/replay_trials.csv.gz",
                            "artifacts/exp_c/final/attack_predictions.csv.gz",
                            "artifacts/exp_b/final/P2/seed0_outputs.npz (labels only)"]),
        sources=dict(sorted(sources.items())),
    )
    (OUT / "results.json").write_text(json.dumps(results, separators=(",", ":")), encoding="utf-8")
    total = sum(p.stat().st_size for p in OUT.rglob("*.json"))
    print(f"Wrote {OUT} ({total / 1024:.0f} KiB); experiments: "
          + ", ".join(f"{k}={v['status']}" for k, v in status.items()))


if __name__ == "__main__":
    main()
