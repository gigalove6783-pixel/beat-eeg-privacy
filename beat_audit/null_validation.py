"""Bounded, representative 51 x 3 whole-procedure null extensions.

Five feature attacks at k=20 and cue-aware commands at k=50 use three correlated
model seeds, five windows per segment and both whole-day cross-day pairs. All
12 seed/pair/direction components share each fold's matching. Both role-specific
comparison caps are active (100 potential window pairs per component).

Raw features have 210 dimensions in the full lane (8 in CI). Two cues have
population-wide effects but no user effect; rows and physical sides are
independent. Command probabilities also depend only on cue. Shared row noise
correlates model seeds without introducing an A/B identity association.

These cases test nominal-size control of each pre-named comparator, not the
familywise error of selecting the strongest comparator. Optional identification,
gallery and sequential modules are disabled: none decides linkage.
"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from .api import AuditConfig, Condition, Segment, audit_stream
from .engine import source_hashes
from .io import digest
from .resources import peak_memory_bytes
from .synthetic import stream_arrays
from .validate import binomial_tolerance


COMPARATORS = ("cosine", "lda_cosine", "pair_logistic",
               "cosine_rich", "lda_cosine_rich", "bayes_factor")


def representative_simulation(job, *, permutations=None):
    """Execute one independent synthetic audit, retaining the actual reported p.

    job = (comparator, simulation_index, lane, base_seed). Seed domains are fixed
    before outcomes are observed; timing pilots use a separate base_seed.
    """
    name, index, lane, seed = job
    started = time.perf_counter()
    commands = name == "bayes_factor"
    k = 50 if commands else 20
    settings = AuditConfig(
        k_grid=(k,), seed=seed + 29 * index, fold_seed=seed + 31 * index,
        permutations=permutations if permutations is not None else (10_000 if lane == "full" else 199),
        bootstraps=2000 if lane == "full" else 39,
        comparators=(name,) if not commands else ("cosine",),
        gallery=False, closed_set=False, sequential=False, threads=1,
        cohort="representative-null-51x3",
    )
    condition = Condition("cross-day", "cross-day", tuple(
        ((Segment("1"),), (Segment(day),)) for day in ("2", "3")))
    arrays = stream_arrays(
        51, 3, k * 5, kind="commands" if commands else "features",
        effect=0, heterogeneous=False, seed=seed + index * 1009,
    )
    cue = (arrays["order"] % 2).astype(int)
    arrays["target"] = np.where(cue == 0, "left", "right")
    dimensions = 210 if lane == "full" else 8
    rng = np.random.default_rng(seed + index * 1009 + 100_003)
    common = rng.random(len(cue)) if commands else rng.normal(size=(len(cue), dimensions))
    with tempfile.TemporaryDirectory(prefix="beat-audit-representative-") as directory:
        paths = {}
        for model_seed in range(3):
            noise = np.random.default_rng(seed + index * 1009 + model_seed * 101 + 200_003)
            if commands:
                uniform = (common + noise.uniform(0, 0.2, len(cue))) % 1
                boundaries = np.array([[0.60, 0.90], [0.20, 0.50]])[cue]
                arrays["outputs"] = (uniform[:, None] >= boundaries).sum(axis=1)
            else:
                arrays["outputs"] = (
                    common + 0.5 * noise.normal(size=common.shape) +
                    (2 * cue - 1)[:, None] * np.linspace(-0.5, 0.5, dimensions)
                )
            path = Path(directory) / f"seed-{model_seed}.npz"
            np.savez(path, **arrays)
            paths[str(model_seed)] = path
        report = audit_stream(paths, conditions=[condition], config=settings)
    cell = report["cells"][0]
    if cell["status"] != "ok":
        return {"status": "failed", "reason": cell.get("reason"),
                "seconds": time.perf_counter() - started}
    result = cell["comparators"][name]
    return {
        "status": "ok", "index": index, "p": result["p"], "reject": result["p"] <= 0.05,
        "auc": result["auc"], "interval": result["interval"], "draws": result["draws"],
        "comparison_bounds": result["comparison_bounds"],
        "inputs_sha256": report["hashes"]["inputs"],
        "configuration_sha256": report["hashes"]["configuration"],
        "population_sha256": cell["population_hash"],
        "seconds": time.perf_counter() - started, "peak_rss_bytes": peak_memory_bytes(),
    }


def run_representative_nulls(*, simulations=16, threads=4, lane="full", seed=20261114,
                             comparators=COMPARATORS, permutations=None, budget_seconds=1800):
    """Return pre-named null cases with retained failures and binomial 99% tolerances.

    Optional comparator and permutation restrictions are recorded, never inferred
    from outcomes. The decision-null CLI fixes the three deciding attacks and
    a separate seed domain; older representative records remain untouched.
    """
    if simulations < 1 or threads < 1 or threads > 4:
        raise ValueError("simulations/threads: use positive simulations and 1..4 workers")
    if lane not in ("full", "ci"):
        raise ValueError("lane: choose full or ci")
    if (not comparators or len(set(comparators)) != len(comparators) or
            not set(comparators) <= set(COMPARATORS)):
        raise ValueError("comparators: supply distinct supported null comparator names")
    permutations = permutations if permutations is not None else (10_000 if lane == "full" else 199)
    if permutations < 1:
        raise ValueError("permutations: supply a positive number of matching draws")
    started = time.perf_counter()
    configuration = {
        "lane": lane, "simulations_per_case": simulations, "threads": threads, "seed": seed,
        "users": 51, "days": 3, "model_seeds": 3, "windows_per_segment": 5,
        "segments_per_day": 2, "feature_dimensions": 210 if lane == "full" else 8,
        "feature_k": 20, "command_k": 50, "cues": ["left", "right"],
        "permutations": permutations,
        "bootstraps": 2000 if lane == "full" else 39,
        "comparators": list(comparators),
        "wall_time_budget_seconds": budget_seconds,
        "condition": "day 1 (both segments) versus day 2, and day 1 versus day 3",
        "directions": "coupled A->B and B->A; same matching across pairs and model seeds",
        "role_sampling": {"raw_pairs_per_component": 100, "components": 12,
                          "mated_cap_per_component": [16, 17],
                          "nonmated_cap_per_component": [4, 5]},
        "generative_null": "IID physical rows/sides; common row noise across model seeds; "
                           "cue effects shared across all users; no identity information",
        "size_gate": "each pre-named comparator; upper binomial 99% count bound at p=0.05",
        "scope": "whole linkage procedure including development fitting/calibration, folds, "
                 "bounded pairing, model seeds, bidirectional matching and subject bootstrap",
    }
    jobs = [(name, index, lane, seed) for name in comparators for index in range(simulations)]
    simulate = partial(representative_simulation, permutations=permutations)
    if threads == 1:
        outputs = list(map(simulate, jobs))
    else:
        with ProcessPoolExecutor(max_workers=threads) as pool:
            outputs = list(pool.map(simulate, jobs, chunksize=1))
    cases = []
    bounds = binomial_tolerance(simulations, 0.05)
    for offset, name in enumerate(comparators):
        values = outputs[offset * simulations:(offset + 1) * simulations]
        failures = [v for v in values if v["status"] != "ok"]
        count = sum(v.get("reject", False) for v in values)
        cases.append({
            "comparator": name, "kind": "commands" if name == "bayes_factor" else "features",
            "k": 50 if name == "bayes_factor" else 20,
            "status": "failed" if failures else "ok", "simulations": simulations,
            "rejections": count, "rejection_rate": count / simulations,
            "binomial_99_count_bounds": bounds,
            "binomial_99_rate_bounds": [v / simulations for v in bounds],
            "passed": not failures and count <= bounds[1], "simulations_detail": values,
        })
    worker_peak = max((v.get("peak_rss_bytes", 0) for v in outputs), default=0)
    parent_peak = peak_memory_bytes()
    return {
        "configuration": configuration, "configuration_sha256": digest(configuration),
        "source": source_hashes(), "whole_procedure": cases,
        "passed": all(c["passed"] for c in cases), "runtime_seconds": time.perf_counter() - started,
        "parent_peak_rss_bytes": parent_peak, "worker_peak_rss_bytes": worker_peak,
        "peak_rss_upper_bound_bytes": parent_peak + threads * worker_peak,
        "memory_method": "parent peak plus worker_count times maximum worker peak (upper bound)",
    }


def append_validation(path, report):
    """Preserve all original records/provenance; append an independently hashed lane."""
    path = Path(path)
    existing = json.loads(path.read_text(encoding="utf-8"))
    existing.setdefault("additional_null_validation", []).append(report)
    existing["all_recorded_lanes_passed"] = existing["passed"] and all(
        r["passed"] for r in existing["additional_null_validation"])
    path.write_text(json.dumps(existing, indent=2, allow_nan=False), encoding="utf-8")
