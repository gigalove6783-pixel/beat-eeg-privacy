"""Audit orchestration. Physical sides and population masks remain explicit throughout."""

from contextlib import contextmanager
from dataclasses import asdict
import os
from pathlib import Path
import time

import numpy as np
from sklearn.utils.parallel import _get_threadpool_controller

from .api import Condition, Segment
from .closed_set import identify
from .comparators import Comparator, sampled_indices
from .gallery import measure, predict
from .inference import auc_cube, bootstrap, holm, matching_test, observed_partners
from .io import digest, file_hash, load_input, make_windows, masked
from .metrics import Calibration, pooled_metrics, roc_metrics
from .protocol import POST_HOC_RANKING, canonical_pairs, chronological_days, unavailable_analyses
from .sequential import decision, summarize, trajectory


@contextmanager
def thread_limit(threads):
    """Limit loaded BLAS/OpenMP pools as well as libraries loaded in this context."""
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("threads: supply a positive integer thread limit")
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    old = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = str(threads)
    try:
        with _get_threadpool_controller().limit(limits=threads):
            yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def source_hashes():
    files = {path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    return {"sha256": digest(files), "files": files}


def default_conditions(stream, day_order=None):
    days, _ = chronological_days(stream.day, day_order)
    same = []
    for day in days:
        segments = sorted({s for d, s in stream.segments if d == day})
        if len(segments) >= 2:
            midpoint = len(segments) // 2
            same.append((tuple(Segment(day, s) for s in segments[:midpoint]),
                         tuple(Segment(day, s) for s in segments[midpoint:])))
    conditions = []
    if same:
        conditions.append(Condition("same-day", "same-day", tuple(same)))
    conditions.append(Condition(
        "cross-day", "cross-day",
        tuple(((Segment(days[0]),), (Segment(day),)) for day in days[1:]),
    ))
    return conditions


def resolve_condition(condition, schema):
    if condition.relation not in ("same-day", "cross-day") or not condition.pairs:
        raise ValueError("condition: declare same-day or cross-day and at least one A/B pair")
    resolved = []
    for a, b in condition.pairs:
        sides = []
        for side in (a, b):
            selected = []
            for selector in side:
                matches = [(d, s) for d, s in schema if d == str(selector.day) and
                           (selector.segment is None or s == str(selector.segment))]
                if not matches:
                    raise ValueError("condition.segment: select day/segment labels present in input")
                selected.extend(matches)
            if not selected or len(selected) != len(set(selected)):
                raise ValueError("condition.segment: select nonempty, nonduplicated segments")
            sides.append(selected)
        a, b = sides
        da, db = {d for d, _ in a}, {d for d, _ in b}
        valid = (not set(a) & set(b) and len(da) == 1 and da == db
                 if condition.relation == "same-day" else not da & db)
        if not valid:
            raise ValueError("condition: use disjoint same-day segments or disjoint cross-day days")
        resolved.append((a, b))
    return resolved


def assign_folds(users, seed, explicit=None):
    count = min(5, len(users) // 3)
    if len(users) < 6:
        raise ValueError("user: retain at least 6 eligible users")
    if explicit is not None:
        if any(u not in explicit for u in users):
            raise ValueError("population.folds: assign every analysed user to a fold")
        mapping = {u: int(explicit[u]) for u in users}
        groups = [sorted(u for u in users if mapping[u] == f) for f in sorted(set(mapping.values()))]
        if len(groups) != count or min(map(len, groups)) < 2:
            raise ValueError("population.folds: use min(5,floor(U/3)) folds, at least 2 users each")
    else:
        shuffled = np.random.default_rng(seed).permutation(sorted(users))
        groups = [sorted(group.tolist()) for group in np.array_split(shuffled, count)]
        mapping = {u: f for f, group in enumerate(groups) for u in group}
    return mapping, groups


def _side(blocks, user, selectors):
    return np.concatenate([blocks[user, d, s] for d, s in selectors])


def _tables(comparator, sides, users, mcap, ncap, seed, *, prepared=False):
    """Two role-specific bounded score tables, including potential permuted mates."""
    score_pairs = comparator.score_transformed if prepared else comparator.score
    if all(len(side[u]) == 1 for side in sides for u in users):
        a, b = [np.concatenate([side[u] for u in users]) for side in sides]
        score = np.empty(len(users) ** 2)
        for start in range(0, len(score), 4096):
            flat = np.arange(start, min(start + 4096, len(score)))
            score[flat] = score_pairs(a[flat // len(users)], b[flat % len(users)])
        table = score.reshape(len(users), len(users), 1)
        return table, table
    rng = np.random.default_rng(seed)
    tables = []
    for cap in (mcap, ncap):
        table = [[None for _ in users] for _ in users]
        left, right, destinations, lengths = [], [], [], []

        def flush():
            if not lengths:
                return
            scores = score_pairs(np.concatenate(left), np.concatenate(right))
            parts = np.split(scores, np.cumsum(lengths)[:-1])
            for (i, j), part in zip(destinations, parts):
                table[i][j] = part
            left.clear()
            right.clear()
            destinations.clear()
            lengths.clear()

        for ui, u in enumerate(users):
            for vi, v in enumerate(users):
                a, b = sides[0][u], sides[1][v]
                i, j = sampled_indices(len(a), len(b), cap, rng)
                if sum(lengths) + len(i) > 4096:
                    flush()
                left.append(a[i])
                right.append(b[j])
                destinations.append((ui, vi))
                lengths.append(len(i))
        flush()
        tables.append(table)
    return tables


def _pooled(tables, users):
    """Each user totals one in each class; partners carry equal nonmated weights."""
    mtable, ntable = tables
    if isinstance(mtable, np.ndarray):
        n = len(users)
        return (np.diag(mtable[:, :, 0]), ntable[:, :, 0][~np.eye(n, dtype=bool)],
                np.ones(n), np.full(n * (n - 1), 1 / (n - 1)))
    m, n, wm, wn = [], [], [], []
    for i in range(len(users)):
        own = mtable[i][i]
        m.extend(own)
        wm.extend(np.repeat(1 / len(own), len(own)))
        for j in range(len(users)):
            if i == j:
                continue
            other = ntable[i][j]
            n.extend(other)
            wn.extend(np.repeat(1 / ((len(users) - 1) * len(other)), len(other)))
    return tuple(np.asarray(x) for x in (m, n, wm, wn))


def _raw(stream, user, selectors):
    own = np.flatnonzero(stream.user == user)
    # Selector order establishes day order; order is only within a user/day.
    parts = []
    for day, segment in selectors:
        rows = own[(stream.day[own] == day) & (stream.segment[own] == segment)]
        parts.append(rows[np.argsort(stream.order[rows], kind="stable")])
    return np.concatenate(parts)


def _sequential_records(stream, users, selectors, comparator, cues, component_count):
    records = []
    cue_index = {cue: i for i, cue in enumerate(cues)}
    for u in users:
        ref = _raw(stream, u, selectors[0])[:100]
        x = np.zeros_like(comparator.alpha)
        np.add.at(x, ([cue_index[t] for t in stream.target[ref]], stream.outputs[ref]), 1)
        for v in users:
            probe = _raw(stream, v, selectors[1])
            path = trajectory(x, stream.outputs[probe],
                              [cue_index[t] for t in stream.target[probe]], comparator.alpha)
            records.append({
                **decision(path), "mated": u == v, "user": u,
                "weight": 1 / (component_count * (1 if u == v else len(users) - 1)),
            })
    return records


def _gallery_rows(comparator, sides, users, cap, seed):
    """Bounded genuine galleries: one shared probe per row, not independent pair scores."""
    rng = np.random.default_rng(seed)
    result = []
    for u in users:
        candidates = [u] + [v for v in users if v != u]
        probe = sides[1][u][rng.integers(len(sides[1][u]), size=cap)]
        columns = []
        for v in candidates:
            reference = sides[0][v][rng.integers(len(sides[0][v]), size=cap)]
            columns.append(comparator.score(reference, probe))
        result.append(np.column_stack(columns))
    return result


def _cell_population(streams, users, folds, required, k):
    masks = {}
    for seed, stream in streams.items():
        indices = []
        for user in users:
            for selector in required:
                rows = _raw(stream, user, [selector])
                indices.extend(stream.rows[rows[:len(rows) // k * k]].tolist())
        masks[seed] = sorted(indices)
    value = {"users": sorted(users), "folds": folds, "masks": masks}
    return value, digest(value)


def _analyse(streams, blocks, users, groups, resolved, name, config, cues, k, *, timing_folds=None):
    components = len(streams) * len(resolved) * 2
    if components > 50:
        raise ValueError("condition: at most 50 seed/pair/direction components fit comparison bounds")
    cubes, matrices, user_values = [], [], {}
    pooled = {key: [] for key in ("m", "n", "wm", "wn", "lm", "ln")}
    operating, training, sequential, closed, gallery_fold = [], [], [], {}, []
    selected_groups = [group for f, group in enumerate(groups)
                       if timing_folds is None or f in timing_folds]
    target_count = sum(map(len, selected_groups))
    for f, targets in enumerate(groups):
        if timing_folds is not None and f not in timing_folds:
            continue
        development = [u for u in users if u not in targets]
        forward_sum = np.zeros((len(targets),) * 3)
        reverse_sum = np.zeros_like(forward_sum)
        seed_pools = []
        gallery_measured = {n: [] for n in range(2, len(targets) + 1)}
        for s, (seed_name, stream) in enumerate(streams.items()):
            fold_pool = {key: [] for key in ("m", "n", "wm", "wn")}
            seed_pools.append(fold_pool)
            block = blocks[seed_name]
            required = sorted({key for pair in resolved for side in pair for key in side})
            x = np.concatenate([block[u, d, seg] for u in development for d, seg in required])
            labels = np.concatenate([
                np.repeat(u, len(block[u, d, seg])) for u in development for d, seg in required
            ])
            attack_seed = config.seed + f * 1009 + s * 101
            development_sides = [
                [{u: _side(block, u, side) for u in development} for side in pair]
                for pair in resolved
            ] if name == "pair_logistic" else None
            comparator = Comparator(name, attack_seed).fit(x, labels, side_pairs=development_sides)
            training.append({"fold": f, "model_seed": seed_name, **comparator.info})
            for p, pair in enumerate(resolved):
                side_values = [{u: _side(block, u, side) for u in users} for side in pair]
                prepared = name.endswith("_rich")
                score_sides = [
                    {u: comparator.transform(values) for u, values in side.items()}
                    for side in side_values
                ] if prepared else side_values
                for direction in range(2):
                    selectors = pair if direction == 0 else pair[::-1]
                    sides = side_values if direction == 0 else side_values[::-1]
                    scoring = score_sides if direction == 0 else score_sides[::-1]
                    part = (s * len(resolved) + p) * 2 + direction
                    mcap = 200 // components + (part < 200 % components)
                    ncap = 50 // components + (part < 50 % components)
                    draw_seed = attack_seed + p * 37 + direction * 17
                    tables = _tables(comparator, scoring, targets, mcap, ncap, draw_seed,
                                     prepared=prepared)
                    cube = auc_cube(*tables)
                    if direction == 0:
                        forward_sum += cube / (len(streams) * len(resolved))
                    else:
                        reverse_sum += cube / (len(streams) * len(resolved))
                    dev_tables = _tables(comparator, scoring, development, mcap, ncap, draw_seed,
                                         prepared=prepared)
                    dm, dn, dwm, dwn = _pooled(dev_tables, development)
                    calibration = Calibration().fit(dm, dn, dwm, dwn)
                    threshold = roc_metrics(dm, dn, dwm, dwn)["threshold"]
                    m, n, wm, wn = _pooled(tables, targets)
                    wm, wn = wm / components, wn / components
                    values = (m, n, wm, wn, calibration.predict(m), calibration.predict(n))
                    for key, value in zip(pooled, values):
                        pooled[key].append(value)
                    for key, value in zip(fold_pool, (m, n, wm, wn)):
                        fold_pool[key].append(value)
                    operating.append({
                        "fold": f, "model_seed": seed_name, "pair": p,
                        "direction": "A->B" if direction == 0 else "B->A",
                        "development_eer_threshold": threshold,
                        "FMR": float(np.average(n >= threshold, weights=wn)),
                        "FNMR": float(np.average(m < threshold, weights=wm)),
                        "user_weight": len(targets) / (target_count * components),
                    })
                    if name.endswith("_rich"):
                        operating[-1]["calibration_parameters"] = {
                            "mean": float(calibration.mean), "scale": float(calibration.scale),
                            "coef": calibration.model.coef_.tolist(),
                            "intercept": calibration.model.intercept_.tolist(),
                        }
                    if config.sequential and stream.kind == "commands":
                        sequential.extend(_sequential_records(
                            stream, targets, selectors, comparator, cues, components))
                    if config.closed_set and not name.endswith("_rich"):
                        reference = np.concatenate([sides[0][u] for u in targets])
                        probe = np.concatenate([sides[1][u] for u in targets])
                        ru = np.concatenate([np.repeat(u, len(sides[0][u])) for u in targets])
                        pu = np.concatenate([np.repeat(u, len(sides[1][u])) for u in targets])
                        enrolled = identify(
                            reference, ru, probe, pu, kind=stream.kind,
                            alpha=getattr(comparator, "alpha", None), seed=attack_seed,
                        )
                        for attack, accuracy in enrolled.items():
                            for u, value in accuracy.items():
                                closed.setdefault(attack, {}).setdefault(u, []).append(value)
                    if config.gallery:
                        rows = _gallery_rows(comparator, sides, targets, ncap, draw_seed)
                        for n_gallery in gallery_measured:
                            gallery_measured[n_gallery].extend(
                                measure(row, n_gallery) for row in rows)
        cubes.append((forward_sum, reverse_sum))
        matrix = observed_partners(forward_sum, reverse_sum)
        matrices.append(matrix)
        user_values.update({u: float(np.nanmean(matrix[i])) for i, u in enumerate(targets)})
        if config.gallery:
            fps = [{key: np.concatenate(value) for key, value in pool.items()}
                   for pool in seed_pools]
            small = []
            for n_gallery, values in gallery_measured.items():
                prediction = float(np.mean([
                    predict(fp["m"], fp["n"], n_gallery, fp["wm"], fp["wn"]) for fp in fps
                ]))
                measured = float(np.mean(values))
                small.append({"N": n_gallery, "predicted": prediction, "measured": measured})
            supported = all(abs(r["predicted"] - r["measured"]) <= 0.05 for r in small)
            gallery_fold.append({"fold": f, "user_weight": len(targets) / target_count,
                                 "check": small, "supported": supported, "_pools": fps})
    result = matching_test(cubes, draws=config.permutations, seed=config.seed)
    result.update(bootstrap(matrices, replicates=config.bootstraps, seed=config.seed + 1))
    result.update(per_user_auc=user_values, training=training)
    result["partner_auc"] = [
        {"users": group, "values": [[None if np.isnan(v) else float(v) for v in row]
                                   for row in matrix]} for group, matrix in zip(selected_groups, matrices)
    ]
    scores = {key: np.concatenate(value) for key, value in pooled.items()}
    result["metrics"] = pooled_metrics(
        scores["m"], scores["n"], scores["wm"], scores["wn"], scores["lm"], scores["ln"])
    result["operating_points"] = operating
    result["metrics"]["development_threshold_FMR"] = sum(
        row["FMR"] * row["user_weight"] for row in operating)
    result["metrics"]["development_threshold_FNMR"] = sum(
        row["FNMR"] * row["user_weight"] for row in operating)
    edges = np.linspace(min(scores["m"].min(), scores["n"].min()) - 1e-9,
                        max(scores["m"].max(), scores["n"].max()) + 1e-9, 51)
    result["histogram"] = {
        "edges": edges.tolist(),
        "mated": np.histogram(scores["m"], edges, weights=scores["wm"] / target_count)[0].tolist(),
        "nonmated": np.histogram(scores["n"], edges, weights=scores["wn"] / target_count)[0].tolist(),
    }
    if sequential:
        result["sequential"] = summarize(sequential)
    if closed:
        result["closed_set"] = {
            attack: {"accuracy": float(np.mean([np.mean(v) for v in values.values()])),
                     "per_user": {u: float(np.mean(v)) for u, v in values.items()}}
            for attack, values in closed.items()
        }
    if gallery_fold:
        supported = all(row["supported"] for row in gallery_fold)
        larger = []
        maximum = min(len(pool["n"]) // 10 for row in gallery_fold for pool in row["_pools"])
        if supported:
            candidates = sorted({max(map(len, groups)) + 1, 20, 50, 100, maximum})
            for n in candidates:
                if n <= max(map(len, groups)) or n > maximum:
                    continue
                larger.append({
                    "N": n,
                    "predicted": sum(row["user_weight"] * float(np.mean([
                        predict(pool["m"], pool["n"], n, pool["wm"], pool["wn"])
                        for pool in row["_pools"]])) for row in gallery_fold),
                })
        for row in gallery_fold:
            del row["_pools"]
        result["gallery"] = {
            "folds": gallery_fold,
            "interpretation": ("Sensitivity scenario under independence; not population risk"
                               if supported else "independence model not supported"),
            "larger_N": larger,
        }
    result["comparison_bounds"] = {"mated_per_user": 200, "nonmated_per_partner": 50,
                                    "components": components, "training_pairs": 20_000}
    if result["auc"] < 0.5:
        result["warning"] = "mean AUC below 0.5"
    return result


def _validate_config(config):
    if not config.k_grid or any(not isinstance(k, int) or k < 1 for k in config.k_grid):
        raise ValueError("k: supply a nonempty grid of positive integer window sizes")
    if len(set(config.k_grid)) != len(config.k_grid):
        raise ValueError("k: remove duplicate window lengths")
    if config.permutations < 1 or config.bootstraps < 1:
        raise ValueError("inference: use positive permutation and bootstrap counts")
    if not config.comparators or not set(config.comparators) <= {
            "cosine", "lda_cosine", "pair_logistic", "cosine_rich", "lda_cosine_rich"}:
        raise ValueError("comparators: choose cosine, lda_cosine, pair_logistic, "
                         "cosine_rich or lda_cosine_rich")
    if config.seconds_per_output is not None and config.seconds_per_output <= 0:
        raise ValueError("seconds-per-output: supply a positive sampling period")


def run_audit(inputs, *, conditions, config, population, kind, out, timing_folds=None):
    started = time.perf_counter()
    _validate_config(config)
    if timing_folds is not None and (
            not timing_folds or len(set(timing_folds)) != len(timing_folds) or
            any(not isinstance(f, int) or f < 0 for f in timing_folds)):
        raise ValueError("folds: supply distinct zero-based fold positions for the timing pilot")
    paths = {"0": inputs} if isinstance(inputs, (str, Path)) else {str(k): v for k, v in inputs.items()}
    if not paths:
        raise ValueError("inputs: supply at least one model-seed input file")
    streams = {key: load_input(path, kind=kind) for key, path in sorted(paths.items())}
    first = next(iter(streams.values()))
    for stream in streams.values():
        if (stream.users != first.users or stream.segments != first.segments or
                stream.kind != first.kind or stream.alphabet != first.alphabet or
                set(stream.target) != set(first.target)):
            raise ValueError("inputs: model seeds must share users, segments, kind, C and cues")
    day_order, day_order_source = chronological_days(
        first.day, getattr(config, "day_order", None), required=conditions is None)
    conditions = (list(conditions) if conditions is not None
                  else default_conditions(first, day_order))
    if not conditions or len({c.name for c in conditions}) != len(conditions):
        raise ValueError("conditions: supply uniquely named conditions")
    resolved = {c.name: resolve_condition(c, first.segments) for c in conditions}
    users = sorted(str(u) for u in population.users) if population is not None else first.users
    if len(set(users)) != len(users):
        raise ValueError("population.users: remove duplicate user labels")
    if population is not None:
        for key, mask in population.masks.items():
            if key not in streams:
                raise ValueError("population.masks: keys must match input model-seed names")
            streams[key] = masked(streams[key], mask)
    cues = list(config.cues) if config.cues is not None else sorted(set(first.target))
    if len(set(cues)) != len(cues) or not set(first.target) <= set(cues):
        raise ValueError("cues: declare each observed cue exactly once; absent extra levels are allowed")
    configuration = {
        "settings": config.to_dict(), "conditions": [asdict(c) for c in conditions],
        "kind": first.kind, "day_order": day_order, "day_order_source": day_order_source,
    }
    report = {
        "schema_version": 1, "cohort": config.cohort,
        "days": sorted(set(first.day)), "kind": first.kind,
        "day_order": day_order, "day_order_source": day_order_source,
        "model_seeds": list(streams), "configuration": configuration,
        "hashes": {"source": source_hashes(), "configuration": digest(configuration),
                   "inputs": {key: file_hash(path) for key, path in paths.items()}},
        "cells": [], "contrasts": [],
        "warnings": ["Below 20 users: estimates are unstable"] if len(users) < 20 else [],
        "interpretation": (
            "Unseen by the attacker's comparator, not necessarily unseen by the BCI. "
            "No anonymity, safety or legal conclusion. Transmitted outputs only."
        ),
    }
    if timing_folds is not None:
        report.update(purpose="timing_only", timing_folds=list(timing_folds))
    for relation in ("same-day", "cross-day"):
        if not any(c.relation == relation for c in conditions):
            report["warnings"].append(f"{relation}: not evaluated; supply explicit side conditions")
    if config.permutations != 10_000 or config.bootstraps != 2000:
        report["warnings"].append("Nonstandard inference budget; not the full §3 procedure")
    with thread_limit(config.threads):
        for k in config.k_grid:
            blocks = {seed: make_windows(stream, k, cues) for seed, stream in streams.items()}
            rich_blocks = {
                seed: make_windows(stream, k, cues, summary="rich")
                for seed, stream in streams.items()
            } if (first.kind == "features" and k > 1 and
                  any(name.endswith("_rich") for name in config.comparators)) else None
            for condition in conditions:
                required = sorted({key for pair in resolved[condition.name]
                                   for side in pair for key in side})
                eligible = [u for u in users if all(
                    (u, d, s) in block and len(block[u, d, s]) > 0
                    for block in blocks.values() for d, s in required
                )]
                excluded = [u for u in users if u not in eligible]
                cell = {"condition": condition.name, "relation": condition.relation, "k": k,
                        "days": sorted({d for d, _ in required}), "excluded_users": excluded,
                        "analysed_users": eligible, "status": "not_evaluable", "comparators": {},
                        "resolved_pairs": canonical_pairs(resolved[condition.name])}
                pop, phash = _cell_population(streams, eligible, {}, required, k)
                cell.update(population=pop, population_hash=phash)
                report["cells"].append(cell)
                if len(excluded) > 0.1 * len(users) or len(eligible) < 6:
                    cell["reason"] = (
                        f"k={k}, user: {len(excluded)}/{len(users)} excluded; supply a window "
                        "in every required segment for >=90% of users and at least 6 users, "
                        "or reduce k"
                    )
                    continue
                try:
                    folds, groups = assign_folds(
                        eligible, config.fold_seed, population.folds if population else None)
                    if timing_folds is not None and max(timing_folds) >= len(groups):
                        raise ValueError("folds: choose positions below the number of eligible folds")
                    pop, phash = _cell_population(streams, eligible, folds, required, k)
                    cell.update(population=pop, population_hash=phash)
                    names = ("bayes_factor",) if first.kind == "commands" else config.comparators
                    if k == 1 and first.kind == "features":
                        cell["skipped_comparators"] = {
                            name: "k=1: rich summary equals mean; redundant attack skipped"
                            for name in names if name.endswith("_rich")
                        }
                        names = tuple(name for name in names if not name.endswith("_rich"))
                    if not names:
                        cell["reason"] = "comparators: k=1 rich attacks are redundant; include mean attacks"
                        continue
                    for name in names:
                        part_started = time.perf_counter()
                        cell["comparators"][name] = _analyse(
                            streams, rich_blocks if name.endswith("_rich") else blocks,
                            eligible, groups, resolved[condition.name], name, config, cues, k,
                            timing_folds=timing_folds,
                        )
                        if name.endswith("_rich"):
                            cell["comparators"][name]["window_summary"] = {
                                "name": "rich", "cue_means": first.target_given,
                                "log_variance": k >= 5, "variance_floor": 1e-12, "variance_ddof": 0,
                            }
                        if timing_folds is not None:
                            cell["comparators"][name]["timing_seconds"] = time.perf_counter() - part_started
                    cell["status"] = "ok"
                    cell["strongest"] = max(names, key=lambda name: cell["comparators"][name]["auc"])
                    cell["strongest_interpretation"] = POST_HOC_RANKING
                except Exception as exc:
                    cell.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        for condition in conditions:
            cells = [c for c in report["cells"] if c["condition"] == condition.name]
            names = sorted({name for c in cells for name in c["comparators"]})
            for name in names:
                tested = [c for c in cells if not (name.endswith("_rich") and c["k"] == 1)]
                adjusted = holm([
                    c["comparators"][name]["p"] if c["status"] == "ok" else 1 for c in tested])
                for cell, p in zip(tested, adjusted):
                    if cell["status"] != "ok":
                        continue
                    value = cell["comparators"][name]
                    value["holm_p"] = p
                    value["wording"] = ("linkage detected" if p <= 0.05 else
                                        "not detected: insufficient evidence, not absence")
            for cell in cells:
                if cell["status"] == "ok":
                    cell["wording"] = ("linkage detected" if any(
                        a["holm_p"] <= 0.05 for a in cell["comparators"].values()) else
                        "not detected: insufficient evidence, not absence")
        if timing_folds is None:
            _contrasts(report, config)
    report["smallest_detected_k"] = {
        condition.name: min((c["k"] for c in report["cells"] if c["condition"] == condition.name
                            and c.get("wording") == "linkage detected"), default=None)
        for condition in conditions
    }
    report["runtime_seconds"] = time.perf_counter() - started
    if timing_folds is not None:
        from .resources import peak_memory_bytes
        report["peak_rss_bytes"] = peak_memory_bytes()
    report["unavailable_analyses"] = unavailable_analyses(
        report["cells"], report["days"], users)
    if out is not None:
        from .report import write_report
        write_report(report, out)
    return report


def _contrasts(report, config):
    for same in report["cells"]:
        if same["relation"] != "same-day" or same["status"] != "ok":
            continue
        for cross in report["cells"]:
            if (cross["relation"] != "cross-day" or cross["status"] != "ok" or
                    same["k"] != cross["k"]):
                continue
            row = {"same_day": same["condition"], "cross_day": cross["condition"], "k": same["k"]}
            report["contrasts"].append(row)
            if (same["analysed_users"] != cross["analysed_users"] or
                    same["population"]["folds"] != cross["population"]["folds"]):
                row.update(status="not_evaluable",
                           reason="population: supply identical eligible users and folds for pairing")
                continue
            row.update(status="ok", comparators={})
            for name in set(same["comparators"]) & set(cross["comparators"]):
                a, b = same["comparators"][name], cross["comparators"][name]
                difference = [
                    np.array(x["values"], dtype=float) - np.array(y["values"], dtype=float)
                    for x, y in zip(a["partner_auc"], b["partner_auc"])
                ]
                row["comparators"][name] = {
                    "mean": a["auc"] - b["auc"],
                    **bootstrap(difference, replicates=config.bootstraps, seed=config.seed + 1),
                    "interpretation": "Paired estimate only; no deciding test",
                }
