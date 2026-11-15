"""Cue-blind attacker beside the recorded cue-aware one (exploratory, post hoc).

Every reported linkage gave the tool each trial's task cue (`target`). This script measures an
attacker who sees only the transmitted outputs: the same frozen beat_audit source and settings,
the same input files with the `target` array withheld (and with it the cue declaration that
refers to it), the same users, folds, model seeds and windows. Without `target` the rich window
summary is the mean and log variance, and the Bayes factor counts commands in one stratum.

Tier: exploratory and post hoc (decided after the results were known); no verdict. Cells, k and
comparators are restricted through the ordinary API (one condition and one k per job); inference
budgets stay at the recorded 10,000 permutations and 2,000 bootstrap replicates.

    python scripts/audit_cue_blind.py run V1 V2          # jobs in the given order
    python scripts/audit_cue_blind.py table              # results/audit/cue_blind/summary.csv

The cue-aware values in the table are read from the recorded reports. Each job checks that the
cue-blind cell has the recorded cell's users, folds and analysed rows (equal population hash);
jobs marked `reproduce` first rerun the recorded cell with `target` present and require the
recorded AUC, interval, p and per-user values exactly. The paired difference (cue-aware minus
cue-blind) is the tool's subject bootstrap of paired partner-AUC differences (its contrast
procedure), without a verdict.
"""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import sys
import time
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The project's beat_audit, not an installed copy (the venv holds an older non-editable install).
sys.path.insert(0, str(ROOT))
OUT = "results/audit/cue_blind"
FROZEN_JOBS = "results/audit/freeze/jobs.json"
LADDER = "results/audit/ladder"
TIER = "exploratory, post hoc"
ALPHA = 0.05
DETECTED = "linkage detected"
NOT_DETECTED = "not detected: insufficient evidence, not absence"
AWARE, BLIND = "cue_aware_reproduction", "cue_blind"
COMPARED = ("auc", "interval", "p", "partner_auc", "per_user_auc")

YANG_D1 = "day 1 against days 2-3 as one block"
YANG_SAME = "trials 1-100 against 101-200 of day 2 and of day 3"
YANG_D2D3 = "day 2 against day 3"
LEE_S1S2 = "session 1 (calibration) against session 2"


def job(id_, priority, cohort, rung, condition, k, sides, source="frozen", set_=None,
        reproduce=False):
    return dict(id=id_, priority=priority, cohort=cohort, rung=rung, condition=condition, k=k,
                sides=sides, source=source, set=set_, reproduce=reproduce)


def ladder(id_, priority, cohort, set_, rung, condition, sides):
    return job(id_, priority, cohort, rung, condition, 20, sides, "ladder", set_)


JOBS = [
    job("V1", 1, "yang2025", "features_P0", "cross-day", 20, YANG_D1, reproduce=True),
    job("V2", 2, "yang2025", "commands_P2-RC_d8_tau0.60", "cross-day", 50, YANG_D1,
        reproduce=True),
    *(ladder(f"yang2025_d2d3_{rung}", 3, "yang2025", "principal", rung, "cross-day-d2-d3",
             YANG_D2D3)
      for rung in ("features_P2-RC_d8", "commands_shared_tau0.50", "commands_shared_tau0.60")),
    *(ladder(f"lee2019_s1s2_{rung}", 4, "lee2019", "calibration_day", rung,
             "cross-day-session1-session2", LEE_S1S2)
      for rung in ("features_P0", "features_P2-RC_d8", "commands_shared_tau0.50",
                   "commands_shared_tau0.80")),
    job("V1_same_day", 5, "yang2025", "features_P0", "same-day-d2-d3", 20, YANG_SAME),
    job("V2_same_day", 5, "yang2025", "commands_P2-RC_d8_tau0.60", "same-day-d2-d3", 50,
        YANG_SAME),
]
COLUMNS = [
    "tier", "condition_id", "priority", "cohort", "rung", "condition", "sides", "days",
    "relation", "k", "comparator", "recorded_strongest", "users_analysed", "population_hash",
    "aware_auc", "aware_ci_low", "aware_ci_high", "aware_p", "aware_holm_p", "aware_holm_family",
    "aware_wording", "blind_auc", "blind_ci_low", "blind_ci_high", "blind_p", "blind_wording",
    "blind_wording_in_recorded_family", "diff_aware_minus_blind", "diff_ci_low", "diff_ci_high",
    "reproduction", "cue_declaration_aware", "model_seeds", "fold_seed", "seed", "permutations",
    "bootstraps", "aware_report", "aware_report_sha256", "blind_report", "aware_input_sha256",
    "blind_input_sha256", "aware_source_sha256", "blind_source_sha256",
]


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=2)
def recorded_report(path):
    return read_json(ROOT / path)


# ------------------------------------------------------------------ withholding the cue

def npz_bytes(arrays):
    """NPZ of the arrays with fixed member metadata, so equal arrays give equal bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for key, value in arrays.items():
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system, info.external_attr = 3, 0o644 << 16
            with archive.open(info, "w", force_zip64=True) as handle:
                np.lib.format.write_array(handle, np.asarray(value), allow_pickle=False)
    return buffer.getvalue()


def withhold_target(source, target):
    """Writes source without its `target` array (every other array unchanged); returns SHA-256."""
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if "target" not in arrays:
        raise SystemExit(f"audit_cue_blind: {source} holds no target to withhold")
    del arrays["target"]
    data = npz_bytes(arrays)
    target = Path(target)
    if not target.exists() or target.read_bytes() != data:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{target.name}.{os.getpid()}.partial")
        partial.write_bytes(data)
        os.replace(partial, target)
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------ recorded specifications

def fold_assignment(users, seed):
    """The frozen batch's folds (scripts/audit_cohorts.py fold_assignment, the tool's rule)."""
    labels = sorted(str(u) for u in users)
    groups = np.array_split(np.random.default_rng(seed).permutation(labels),
                            min(5, len(labels) // 3))
    return {str(u): int(f) for f, group in enumerate(groups) for u in group}


def resolve(spec):
    """Recorded inputs, configuration, population and condition of a job's cue-aware cell."""
    if spec["source"] == "frozen":
        jobs = read_json(ROOT / FROZEN_JOBS)
        stream = next(s for s in jobs["streams"] if s["rung"] == spec["rung"])
        settings = jobs["settings"]
        condition = stream["conditions"][spec["condition"]]
        return dict(
            kind=stream["kind"], reference=f"{stream['out']}/report.json", origin="frozen",
            inputs={s: (stream["inputs"][s], stream["sha256"][s]) for s in stream["inputs"]},
            config=dict(settings, k_grid=(spec["k"],), comparators=tuple(settings["comparators"]),
                        cohort=stream["cohort"], seconds_per_output=stream["seconds_per_output"],
                        cues=("0", "1"), day_order=tuple(stream["day_order"])),
            users=[str(u) for u in stream["users"]],
            folds=fold_assignment(stream["fold_users"], settings["fold_seed"]),
            relation=condition["relation"],
            pairs=[(p["a"], p["b"]) for p in condition["pairs"]])
    base = f"{LADDER}/{spec['cohort']}/{spec['set']}"
    plan = read_json(ROOT / base / "plan.json")
    s = plan["settings"]
    condition = next(c for c in plan["conditions"] if c["name"] == spec["condition"])
    rung = next(r for r in plan["rungs"] if r["name"] == spec["rung"])
    return dict(
        kind=rung["kind"], reference=f"{base}/{spec['rung']}/report.json",
        origin=f"ladder_{spec['set']}",
        inputs={seed: (item["file"], item["sha256"])
                for seed, item in plan["inputs"][spec["rung"]].items()},
        config=dict(k_grid=(spec["k"],), fold_seed=s["fold_seed"], seed=s["seed"],
                    permutations=s["permutations"], bootstraps=s["bootstraps"],
                    threads=s["threads"], cohort=plan["cohort"],
                    comparators=tuple(s["comparators"]), gallery=s["gallery"],
                    sequential=s["sequential"], closed_set=s["closed_set"],
                    seconds_per_output=plan["seconds_per_output"], cues=None,
                    day_order=tuple(str(d) for d in plan["days"])),
        users=[str(u) for u in plan["users"]], folds=None,
        relation=condition["relation"], pairs=[(a, b) for a, b in condition["pairs"]])


def blind_config(config):
    """The recorded configuration without the cue declaration (it names `target` levels)."""
    return dict(config, cues=None)


def cell_of(report, condition, k):
    cells = [c for c in report["cells"] if c["condition"] == condition and c["k"] == k]
    if len(cells) != 1:
        raise SystemExit(f"audit_cue_blind: no single cell {condition} k={k}")
    return cells[0]


def holm_family(report, condition, comparator):
    """Size of the tool's Holm family of a cell: the condition's k grid for that comparator."""
    return sum(1 for c in report["cells"] if c["condition"] == condition and
               not (comparator.endswith("_rich") and c["k"] == 1))


# ------------------------------------------------------------------ run

def audit(spec, resolved, inputs, config, out):
    from beat_audit.api import AuditConfig, Condition, Population, Segment, audit_stream

    def side(selectors):
        return tuple(Segment(*str(s).split("/", 1)) for s in selectors)

    condition = Condition(spec["condition"], resolved["relation"],
                          tuple((side(a), side(b)) for a, b in resolved["pairs"]))
    return audit_stream({seed: str(ROOT / path) for seed, path in inputs.items()},
                        conditions=[condition], config=AuditConfig(**config),
                        population=Population(users=resolved["users"], folds=resolved["folds"]),
                        kind=resolved["kind"], out=ROOT / out)


def same_support(cell, recorded):
    """Refusal unless a rerun cell has the recorded users, folds, analysed rows and sides."""
    bad = [key for key in ("status", "analysed_users", "population_hash", "resolved_pairs")
           if cell.get(key) != recorded.get(key)]
    if set(cell["comparators"]) != set(recorded["comparators"]):
        bad.append("comparators")
    bad += [f"{name}.partner_auc users" for name, result in cell["comparators"].items()
            if name in recorded["comparators"] and
            [g["users"] for g in result["partner_auc"]] !=
            [g["users"] for g in recorded["comparators"][name]["partner_auc"]]]
    return bad


def reproduces(cell, recorded):
    return [f"{name}.{key}" for name, result in recorded["comparators"].items()
            for key in COMPARED if cell["comparators"].get(name, {}).get(key) != result[key]]


def slimmer():
    loader = importlib.util.spec_from_file_location(
        "audit_slim_reports", ROOT / "scripts" / "audit_slim_reports.py")
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def run_job(spec, reproduce):
    from beat_audit.engine import source_hashes
    started = time.perf_counter()
    resolved = resolve(spec)
    source = source_hashes()["sha256"]
    recorded = read_json(ROOT / resolved["reference"])  # up to 208 MB: keep only what is needed
    hashes, reference = recorded["hashes"], cell_of(recorded, spec["condition"], spec["k"])
    del recorded
    expected = {seed: sha for seed, (_, sha) in resolved["inputs"].items()}
    bad = [path for seed, (path, sha) in resolved["inputs"].items()
           if sha256_file(ROOT / path) != sha]
    if bad or hashes["inputs"] != expected:
        raise SystemExit(f"audit_cue_blind: {spec['id']}: inputs differ from the record {bad}")
    if hashes["source"]["sha256"] != source:
        raise SystemExit(f"audit_cue_blind: {spec['id']}: beat_audit source differs from the "
                         "recorded run's")
    blind_inputs, blind_sha = {}, {}
    for seed, (path, _) in resolved["inputs"].items():
        blind_inputs[seed] = f"{OUT}/inputs/{spec['cohort']}/{resolved['origin']}/{Path(path).name}"
        blind_sha[seed] = withhold_target(ROOT / path, ROOT / blind_inputs[seed])
    record = dict(id=spec["id"], tier=TIER, source_sha256=source,
                  reference=dict(path=resolved["reference"],
                                 sha256=sha256_file(ROOT / resolved["reference"])),
                  inputs=expected, blind_inputs={s: dict(file=blind_inputs[s], sha256=blind_sha[s])
                                                 for s in blind_inputs},
                  variants={}, reproduction="not run")
    variants = ([(AWARE, {s: p for s, (p, _) in resolved["inputs"].items()}, resolved["config"])]
                if reproduce else [])
    variants.append((BLIND, blind_inputs, blind_config(resolved["config"])))
    for variant, inputs, config in variants:
        part = time.perf_counter()
        out = f"{OUT}/{spec['id']}/{variant}"
        report = audit(spec, resolved, inputs, config, out)
        cell = cell_of(report, spec["condition"], spec["k"])
        record["variants"][variant] = dict(
            seconds=time.perf_counter() - part, tool_runtime_seconds=report["runtime_seconds"],
            report=f"{out}/report.json", slim=f"{out}/report.slim.json",
            configuration=report["hashes"]["configuration"], inputs=report["hashes"]["inputs"],
            source=report["hashes"]["source"]["sha256"], support_mismatches=same_support(
                cell, reference))
        slimmer().write_slim(ROOT / out / "report.json")
        if variant == AWARE:
            mismatches = reproduces(cell, reference)
            record["reproduction"] = "exact" if not mismatches else f"FAILED: {mismatches}"
        if record["variants"][variant]["support_mismatches"] or record["reproduction"].startswith(
                "FAILED"):
            break
    record["seconds"] = time.perf_counter() - started
    path = ROOT / OUT / spec["id"] / "run.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    failures = [v["support_mismatches"] for v in record["variants"].values()
                if v["support_mismatches"]]
    if failures or record["reproduction"].startswith("FAILED"):
        raise SystemExit(f"audit_cue_blind: {spec['id']}: refused: support {failures}, "
                         f"reproduction {record['reproduction']}")
    return record


# ------------------------------------------------------------------ table

def paired_difference(aware, blind, bootstraps, seed):
    """Cue-aware minus cue-blind mean per-user AUC with the tool's subject bootstrap of the
    paired partner-AUC differences (beat_audit.engine._contrasts' procedure); no verdict."""
    from beat_audit.inference import bootstrap
    if [g["users"] for g in aware["partner_auc"]] != [g["users"] for g in blind["partner_auc"]]:
        raise SystemExit("audit_cue_blind: paired difference needs identical users and folds")
    difference = [np.array(a["values"], dtype=float) - np.array(b["values"], dtype=float)
                  for a, b in zip(aware["partner_auc"], blind["partner_auc"])]
    result = bootstrap(difference, replicates=bootstraps, seed=seed + 1)
    return aware["auc"] - blind["auc"], result["interval"]


def wording_in_family(p, family):
    """The cue-blind wording if its k had been in the recorded Holm family of `family` tests:
    Holm's adjusted p lies between p and family * p."""
    if family * p <= ALPHA:
        return DETECTED
    if p > ALPHA:
        return NOT_DETECTED
    return "undetermined: depends on the other k of the family (not run)"


def summary_rows(spec, recorded, blind, meta):
    """One row per comparator of the job's cell: recorded cue-aware against cue-blind."""
    aware_cell = cell_of(recorded, spec["condition"], spec["k"])
    blind_cell = cell_of(blind, spec["condition"], spec["k"])
    bad = same_support(blind_cell, aware_cell)
    if bad:
        raise SystemExit(f"audit_cue_blind: {spec['id']}: not the recorded support: {bad}")
    settings = recorded["configuration"]["settings"]
    rows = []
    for name in aware_cell["comparators"]:
        a, b = aware_cell["comparators"][name], blind_cell["comparators"][name]
        family = holm_family(recorded, spec["condition"], name)
        mean, interval = paired_difference(a, b, settings["bootstraps"], settings["seed"])
        rows.append(dict(
            tier=TIER, condition_id=spec["id"], priority=spec["priority"],
            cohort=spec["cohort"], rung=spec["rung"], condition=spec["condition"],
            sides=spec["sides"], days=" ".join(aware_cell["days"]),
            relation=aware_cell["relation"], k=spec["k"], comparator=name,
            recorded_strongest=name == aware_cell["strongest"],
            users_analysed=len(aware_cell["analysed_users"]),
            population_hash=aware_cell["population_hash"],
            aware_auc=a["auc"], aware_ci_low=a["interval"][0], aware_ci_high=a["interval"][1],
            aware_p=a["p"], aware_holm_p=a["holm_p"], aware_holm_family=family,
            aware_wording=a["wording"],
            blind_auc=b["auc"], blind_ci_low=b["interval"][0], blind_ci_high=b["interval"][1],
            blind_p=b["p"], blind_wording=b["wording"],
            blind_wording_in_recorded_family=wording_in_family(b["p"], family),
            diff_aware_minus_blind=mean, diff_ci_low=interval[0], diff_ci_high=interval[1],
            cue_declaration_aware=" ".join(settings["cues"] or ["observed labels"]),
            model_seeds=" ".join(recorded["model_seeds"]), fold_seed=settings["fold_seed"],
            seed=settings["seed"], permutations=settings["permutations"],
            bootstraps=settings["bootstraps"],
            aware_source_sha256=recorded["hashes"]["source"]["sha256"],
            blind_source_sha256=blind["hashes"]["source"]["sha256"],
            aware_input_sha256=" ".join(f"{s}:{h}" for s, h in recorded["hashes"]["inputs"].items()),
            blind_input_sha256=" ".join(f"{s}:{h}" for s, h in blind["hashes"]["inputs"].items()),
            **meta))
    return rows


def write_table(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def table(_args):
    rows = []
    for spec in JOBS:
        run_path = ROOT / OUT / spec["id"] / "run.json"
        if not run_path.exists():
            continue
        record = read_json(run_path)
        if BLIND not in record["variants"] or record["reproduction"].startswith("FAILED"):
            raise SystemExit(f"audit_cue_blind: {spec['id']}: incomplete or failed run")
        reference = record["reference"]
        if sha256_file(ROOT / reference["path"]) != reference["sha256"]:
            raise SystemExit(f"audit_cue_blind: {reference['path']} changed since the run")
        blind = read_json(ROOT / record["variants"][BLIND]["report"])
        rows += summary_rows(spec, recorded_report(reference["path"]), blind, dict(
            reproduction=record["reproduction"], aware_report=reference["path"],
            aware_report_sha256=reference["sha256"],
            blind_report=record["variants"][BLIND]["slim"]))
    write_table(rows, ROOT / OUT / "summary.csv")
    print(f"{OUT}/summary.csv: {len(rows)} rows from "
          f"{len({r['condition_id'] for r in rows})} conditions")


def run(args):
    known = {spec["id"]: spec for spec in JOBS}
    unknown = sorted(set(args.ids) - set(known))
    if unknown:
        raise SystemExit(f"audit_cue_blind: unknown jobs {unknown}; choose from {list(known)}")
    for id_ in args.ids or list(known):
        spec = known[id_]
        record = run_job(spec, spec["reproduce"] or args.reproduce)
        times = ", ".join(f"{v}: {record['variants'][v]['seconds']:.0f} s"
                          for v in record["variants"])
        print(f"{id_}: {times}; reproduction {record['reproduction']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="audit jobs cue-blind (all, in priority order)")
    runner.add_argument("ids", nargs="*")
    runner.add_argument("--reproduce", action="store_true",
                        help="also rerun the recorded cue-aware cell first")
    runner.set_defaults(func=run)
    commands.add_parser("table", help="write summary.csv").set_defaults(func=table)
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
