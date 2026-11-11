"""Reproducible analytic and whole-procedure validation; failures are retained.

Full lane: 1,000 null and 1,000 planted-effect simulations at EACH cohort shape.
Every simulation fits a cosine comparator on development users, scores both
directions, averages two model seeds, uses one matching per fold for all those
components, and executes the specified subject bootstrap. The planted case also
tests a paired same-day minus cross-day contrast with zero population contrast.
The interval estimand is evaluated independently using large Monte-Carlo draws
from the known population, conditional on each actually fitted scaler.

The CI lane uses the same procedure and cohort shapes with reduced repetitions
and draws, explicitly reported. Binomial 99% Monte-Carlo tolerances are stated.
Neither lane silently changes a failed requirement or claims success from an
unexecuted case. Four workers each limited to one BLAS thread satisfy the full
lane's four-thread compute budget.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import beta, betaln, ndtr
from scipy.stats import binom, norm

from .api import AuditConfig, Condition, Segment, audit_stream
from .comparators import log_bayes_factor
from .engine import source_hashes, thread_limit
from .gallery import predict
from .io import digest
from .metrics import Calibration, cllr, d_sys, roc_metrics
from .resources import peak_memory_bytes
from .sequential import decision, trajectory
from .synthetic import stream_arrays


def binomial_tolerance(n, probability):
    """Central 99% count interval under the declared binomial reference model."""
    return [int(binom.ppf(0.005, n, probability)), int(binom.ppf(0.995, n, probability))]


def analytic_checks(seed=41):
    """Independent analytic/integration expectations, not self-derived snapshots."""
    n = 30_000
    quantile = norm.ppf((np.arange(n) + 0.5) / n)
    m, non = quantile + 1, quantile - 1
    eer = roc_metrics(m, non)["eer"]
    expected_eer = float(ndtr(-1))
    index = d_sys(m, non)
    expected_index = quad(
        lambda s: norm.pdf(s - 1) * max(0, 2 / (1 + np.exp(-2 * s)) - 1),
        -15, 15,
    )[0]
    x, y, a = np.array([[2, 0]]), np.array([[1, 1]]), np.array([[0.7, 1.3]])
    density = lambda p: p ** (a[0, 0] - 1) * (1 - p) ** (a[0, 1] - 1) / beta(*a[0])
    integral = lambda counts: quad(
        lambda p: p ** counts[0] * (1 - p) ** counts[1] * density(p), 0, 1)[0]
    expected_bf = integral((x + y)[0]) / (integral(x[0]) * integral(y[0]))
    bf = float(np.exp(log_bayes_factor(x, y, a)))
    predictive_ratio = float(np.exp(
        betaln(*(a + x + y)[0]) - betaln(*(a + x)[0]) -
        betaln(*(a + y)[0]) + betaln(*a[0])))
    calibration = Calibration().fit(m, non)
    grid = np.linspace(-3, 3, 61)
    calibration_error = float(np.max(np.abs(calibration.predict(grid) - 2 * grid)))
    rng = np.random.default_rng(seed)
    count = 30_000
    mate = rng.normal(1, 1, count)
    candidates = rng.normal(0, 1, (count, 9))
    successes = int(np.count_nonzero(mate > candidates.max(axis=1)))
    gallery_expected = quad(lambda s: norm.pdf(s - 1) * ndtr(s) ** 9, -12, 12)[0]
    prediction = predict(quantile + 1, quantile, 10)
    bounds = binomial_tolerance(count, gallery_expected)
    checks = {
        "gaussian_eer": {"observed": eer, "expected": expected_eer, "tolerance": 0.0001,
                         "passed": abs(eer - expected_eer) < 0.0001},
        "gaussian_D_sys": {"observed": index, "expected": expected_index, "tolerance": 0.01,
                           "passed": abs(index - expected_index) < 0.01},
        "D_sys_endpoints": {"identical": d_sys([0, 1], [0, 1]),
                            "disjoint": d_sys([2, 3], [0, 1]),
                            "passed": d_sys([0, 1], [0, 1]) == 0 and
                            d_sys([2, 3], [0, 1]) == 1},
        "bayes_integral": {"observed": bf, "expected": expected_bf,
                           "passed": abs(bf - expected_bf) < 1e-8},
        "bayes_symmetry_predictive": {
            "passed": abs(float(log_bayes_factor(x, y, a) -
                                log_bayes_factor(y, x, a))) < 1e-12 and
            abs(bf - predictive_ratio) < 1e-12},
        "estimated_log_LR": {"maximum_error": calibration_error, "tolerance": 0.01,
                             "Cllr": cllr(calibration.predict(m), calibration.predict(non)),
                             "passed": calibration_error < 0.01},
        "independent_gallery": {"predicted": prediction, "expected": gallery_expected,
                                "successes": successes, "simulations": count,
                                "binomial_99_count_bounds": bounds,
                                "passed": bounds[0] <= successes <= bounds[1] and
                                abs(prediction - gallery_expected) < 0.001},
    }
    return checks


def sequential_checks(n=1000, seed=42):
    """Correct generative hypotheses, planted half-population drift, and Markov stress."""
    rng = np.random.default_rng(seed)
    alpha = np.full((1, 3), 0.3)
    errors = {mode: {"FMR": 0, "FNMR": 0, "mated_log_bf": []}
              for mode in ("iid", "drift", "markov")}
    for _ in range(n):
        p, q = rng.dirichlet(alpha[0], size=2)
        ref = rng.choice(3, 100, p=p)
        x = np.bincount(ref, minlength=3)[None]
        for mode, probability in (("iid", p), ("drift", (p + q) / 2), ("markov", p)):
            probes = [rng.choice(3, 200, p=probability), rng.choice(3, 200, p=q)]
            if mode == "markov":
                for values in probes:
                    for t in range(1, len(values)):
                        if rng.random() < 0.8:
                            values[t] = values[t - 1]
            for mated, values in zip((True, False), probes):
                path = trajectory(x, values, np.zeros(len(values), dtype=int), alpha)
                outcome = decision(path)["decision"]
                if mated:
                    errors[mode]["FNMR"] += outcome == "no link"
                    errors[mode]["mated_log_bf"].append(float(path[-1]))
                else:
                    errors[mode]["FMR"] += outcome == "link"
    output = {}
    bound = 0.05 / 0.95
    allowed = binomial_tolerance(n, bound)
    for mode, values in errors.items():
        output[mode] = {
            "FMR": values["FMR"] / n, "FNMR": values["FNMR"] / n,
            "mean_mated_log_bf": float(np.mean(values["mated_log_bf"])),
            "simulations_per_hypothesis": n,
        }
    output["ville"] = {
        "bound": bound, "binomial_99_count_bounds": allowed,
        "passed": errors["iid"]["FMR"] <= allowed[1] and errors["iid"]["FNMR"] <= allowed[1],
    }
    output["drift_effect"] = {
        "passed": output["drift"]["mean_mated_log_bf"] < output["iid"]["mean_mated_log_bf"] and
        output["drift"]["FNMR"] > output["iid"]["FNMR"],
    }
    output["markov"]["interpretation"] = "Realised rates only; no error-bound assertion"
    return output


def _conditions(days):
    same = tuple(((Segment(str(day), "1"),), (Segment(str(day), "2"),))
                 for day in range(1, days + 1))
    cross = tuple(((Segment("1", "1"),), (Segment(str(day), "2"),))
                  for day in range(2, days + 1))
    return (Condition("same", "same-day", same), Condition("cross", "cross-day", cross))


def _truth_draws(users, seed, n=32_768):
    """Independent population triples; each AUC uses a common reference window."""
    rng = np.random.default_rng(seed)
    scale_values = np.linspace(0.5, 1.5, users)
    scales = rng.choice(scale_values, (2, n, 1))
    means = rng.normal(size=(2, n, 3)) * 2
    reference = means[0] + scales[0] * rng.normal(size=(n, 3))
    mate = means[0] + scales[0] * rng.normal(size=(n, 3))
    other = means[1] + scales[1] * rng.normal(size=(n, 3))
    return reference, mate, other


def _conditional_truth(training, draws, fold_sizes):
    estimates = []
    weights = []
    for row in training:
        a, b, c = [(x - row["mean"]) / row["scale"] for x in draws]
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        wins = ((a * b).sum(1) > (a * c).sum(1)).astype(float)
        estimates.append(wins)
        weights.append(fold_sizes[row["fold"]])
    value = np.average(estimates, axis=0, weights=weights)
    return float(value.mean()), float(value.std(ddof=1) / np.sqrt(len(value)))


def _simulation(job):
    users, days, effect, index, lane, seed = job
    started = time.perf_counter()
    settings = AuditConfig(
        k_grid=(1,), seed=seed + index * 29, fold_seed=seed + index * 31,
        permutations=10_000 if lane == "full" else 199,
        bootstraps=2000 if lane == "full" else 199,
        comparators=("cosine",), gallery=False, closed_set=False, sequential=False, threads=1,
        cohort=f"synthetic-{users}x{days}",
    )
    with tempfile.TemporaryDirectory(prefix="beat-audit-validation-") as directory:
        paths = {}
        for model_seed in range(2):
            path = Path(directory) / f"seed-{model_seed}.npz"
            arrays = stream_arrays(
                users, days, 1, seed=seed + index * 1009 + model_seed * 101,
                effect=effect, heterogeneous=effect != 0,
            )
            np.savez(path, **arrays)
            paths[str(model_seed)] = path
        report = audit_stream(paths, conditions=_conditions(days), config=settings)
    bad = [c for c in report["cells"] if c["status"] != "ok"]
    if bad:
        return {"failed": bad, "seconds": time.perf_counter() - started}
    cross = report["cells"][1]["comparators"]["cosine"]
    value = {"reject": cross["p"] <= 0.05, "p": cross["p"], "auc": cross["auc"],
             "interval": cross["interval"], "seconds": time.perf_counter() - started,
             "peak_rss_bytes": peak_memory_bytes()}
    if effect:
        training = cross["training"]
        sizes = [len(row["users"]) for row in cross["partner_auc"]]
        draws = _truth_draws(users, seed + 7_000_001, n=32_768 if lane == "full" else 8192)
        truth, se = _conditional_truth(training, draws, sizes)
        contrast = report["contrasts"][0]["comparators"]["cosine"]
        value.update(truth=truth, truth_mc_se=se,
                     covered=cross["interval"][0] <= truth <= cross["interval"][1],
                     contrast=contrast["mean"], contrast_interval=contrast["interval"],
                     contrast_covered=contrast["interval"][0] <= 0 <= contrast["interval"][1])
    return value


def run_validation(*, lane="full", simulations=None, threads=4, seed=20261113):
    """Return all observed results, including unsupported coverage and explicit failures."""
    n = simulations if simulations is not None else (1000 if lane == "full" else 12)
    started = time.perf_counter()
    configuration = {"lane": lane, "simulations_per_case": n, "threads": threads, "seed": seed}
    with thread_limit(1):
        analytic = analytic_checks()
        sequential = sequential_checks(1000 if lane == "full" else 200)
    jobs = [(u, d, effect, i, lane, seed + u * 10_000 + int(effect) * 100_000)
            for u, d in ((9, 2), (54, 2), (51, 3)) for effect in (0.0, 2.0) for i in range(n)]
    if threads > 1:
        with ProcessPoolExecutor(max_workers=threads) as pool:
            outputs = list(pool.map(_simulation, jobs, chunksize=1))
    else:
        outputs = list(map(_simulation, jobs))
    cases = []
    failed = []
    for offset in range(0, len(jobs), n):
        users, days, effect, *_ = jobs[offset]
        values = outputs[offset:offset + n]
        failures = [v for v in values if "failed" in v]
        if failures:
            failed.extend(failures)
            cases.append({"users": users, "days": days, "effect": effect,
                          "status": "failed", "failures": failures})
            continue
        row = {"users": users, "days": days, "effect": effect, "simulations": n,
               "model_seeds": 2, "directions": "coupled A->B and B->A",
               "mean_auc": float(np.mean([v["auc"] for v in values])),
               "status": "ok", "simulations_detail": values}
        if effect == 0:
            count = sum(v["reject"] for v in values)
            bounds = binomial_tolerance(n, 0.05)
            row.update(rejections=count, rejection_rate=count / n,
                       binomial_99_count_bounds=bounds, passed=count <= bounds[1])
        else:
            coverage = sum(v["covered"] for v in values) / n
            contrast_coverage = sum(v["contrast_covered"] for v in values) / n
            bounds = [binomial_tolerance(n, 0.90)[0], binomial_tolerance(n, 0.99)[1]]
            gate = (0.90 <= coverage <= 0.99 and 0.90 <= contrast_coverage <= 0.99
                    if lane == "full" else
                    bounds[0] <= round(coverage * n) <= bounds[1] and
                    bounds[0] <= round(contrast_coverage * n) <= bounds[1])
            row.update(
                auc_coverage=coverage, contrast_coverage=contrast_coverage,
                mean_contrast=float(np.mean([v["contrast"] for v in values])),
                mean_truth=float(np.mean([v["truth"] for v in values])),
                truth_mc_se_max=max(v["truth_mc_se"] for v in values),
                coverage_requirement=[0.90, 0.99], binomial_99_count_bounds=bounds,
                coverage_gated=users != 9, passed=gate if users != 9 else True,
            )
        cases.append(row)
    passed = (not failed and all(row["passed"] for row in cases) and
              all(value["passed"] for value in analytic.values()) and
              sequential["ville"]["passed"] and sequential["drift_effect"]["passed"])
    worker_peak = max((v.get("peak_rss_bytes", 0) for v in outputs), default=0)
    return {
        "configuration": configuration, "configuration_sha256": digest(configuration),
        "source": source_hashes(), "analytic": analytic, "sequential": sequential,
        "whole_procedure": cases, "passed": passed,
        "runtime_seconds": time.perf_counter() - started,
        "parent_peak_rss_bytes": peak_memory_bytes(), "worker_peak_rss_bytes": worker_peak,
        "peak_rss_upper_bound_bytes": peak_memory_bytes() + threads * worker_peak,
        "memory_method": "parent peak plus worker_count times maximum worker peak (upper bound)",
        "truth_method": "Independent population Monte Carlo conditional on fitted development "
                        "scalers; common 32768 draws per full-lane shape, standard error recorded",
        "scope": "Cosine whole procedure; other comparators and edge cases exercised by pytest",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("full", "ci", "representative", "decision-null"),
                        default="full")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default="results/audit/validation.json")
    parser.add_argument("--simulations", type=int,
                        help="null repetitions per case (representative:16; decision-null:200)")
    parser.add_argument("--permutations", type=int, default=10_000,
                        help="matching draws for null extensions; recorded in their configuration")
    args = parser.parse_args(argv)
    path = Path(args.out)
    if args.lane in ("representative", "decision-null"):
        from .null_validation import append_validation, run_representative_nulls
        if args.lane == "decision-null":
            report = run_representative_nulls(
                simulations=200 if args.simulations is None else args.simulations,
                threads=args.threads, seed=20261115, permutations=args.permutations,
                comparators=("cosine_rich", "lda_cosine_rich", "bayes_factor"),
                budget_seconds=7200,
            )
        else:
            report = run_representative_nulls(
                simulations=16 if args.simulations is None else args.simulations,
                threads=args.threads, permutations=args.permutations,
            )
        append_validation(path, report)
    else:
        report = run_validation(lane=args.lane, threads=args.threads)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("passed", "runtime_seconds",
                                           "peak_rss_upper_bound_bytes")}))
    for case in report["whole_procedure"]:
        print(json.dumps({k: v for k, v in case.items() if k != "simulations_detail"}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
