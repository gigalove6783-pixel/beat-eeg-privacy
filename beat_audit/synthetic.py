"""Synthetic transmitted outputs only; no EEG/cohort artifacts are read."""

import numpy as np


def stream_arrays(users=12, days=2, per_segment=10, *, kind="features", seed=7,
                  effect=2.0, drift=0.0, heterogeneous=True):
    """Independent segment noise, planted identity means, and optional day drift.

    Feature dimension is 3. Commands have a declared 3-symbol alphabet. For
    features, heterogeneous scales are deterministic user quantiles, permuted by
    the simulation seed; neither their population mean nor labels reveal day.
    """
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(users, 3))
    scales = rng.permutation(np.linspace(0.5, 1.5, users)) if heterogeneous else np.ones(users)
    probabilities = rng.dirichlet(np.full(3, 0.4), size=users)
    independent = rng.dirichlet(np.full(3, 0.4), size=(days, users))
    rows = {key: [] for key in ("outputs", "user", "day", "segment", "order")}
    for u in range(users):
        for d in range(days):
            for segment in range(2):
                if kind == "commands":
                    p = probabilities[u] if effect else np.repeat(1 / 3, 3)
                    if d:
                        p = (1 - drift) * p + drift * independent[d, u]
                    outputs = rng.choice(3, per_segment, p=p)
                else:
                    mean = effect * means[u]
                    if d:
                        mean = (1 - drift) * mean + drift * effect * rng.normal(size=3)
                    outputs = mean + scales[u] * rng.normal(size=(per_segment, 3))
                rows["outputs"].extend(outputs)
                rows["user"].extend([f"u{u:03}"] * per_segment)
                rows["day"].extend([str(d + 1)] * per_segment)
                rows["segment"].extend([str(segment + 1)] * per_segment)
                rows["order"].extend(range(segment * per_segment, (segment + 1) * per_segment))
    result = {key: np.asarray(value) for key, value in rows.items()}
    if kind == "commands":
        result["C"] = np.array(3)
    return result
