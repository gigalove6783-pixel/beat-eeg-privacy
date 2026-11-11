"""Development-only comparators, bounded pair construction, and command evidence."""

import numpy as np
from scipy.special import gammaln
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .io import digest


def log_beta(a):
    return gammaln(a).sum(axis=-1) - gammaln(np.sum(a, axis=-1))


def log_bayes_factor(x, y, alpha):
    """Natural-log BF; last axes are (cue,command). Empty cues contribute zero."""
    x, y, a = np.asarray(x), np.asarray(y), np.asarray(alpha)
    return (log_beta(a + x + y) + log_beta(a) -
            log_beta(a + x) - log_beta(a + y)).sum(axis=-1)


def fit_prior(windows):
    """Positive per-cue prior; one shared concentration selected on the 41-point grid."""
    pooled = windows.sum(axis=0)
    c = windows.shape[-1]
    pi = (pooled + 0.5) / (pooled.sum(axis=-1, keepdims=True) + 0.5 * c)
    grid = np.logspace(-1, 3, 41)
    objective = np.array([
        (log_beta(windows + a * pi) - log_beta(a * pi)).sum() for a in grid
    ])
    index = int(np.argmax(objective))
    return grid[index] * pi, {
        "alpha0": float(grid[index]), "grid_endpoint": index in (0, 40),
        "pi": pi.tolist(),
    }


def sampled_indices(n, m, limit, rng):
    """O(limit) memory even for huge Cartesian products."""
    size = n * m
    flat = np.arange(size) if size <= limit else rng.choice(size, limit, replace=False)
    return flat // m, flat % m


def training_pairs(side_pairs, users, seed, limit=20_000):
    """Balanced development pairs drawn only from the condition's physical sides."""
    rng = np.random.default_rng(seed)
    window_count = sum(len(side[u]) for pair in side_pairs for side in pair for u in users)
    n = min(limit, max(2, window_count * 8))
    y = np.arange(n) % 2
    first_user = rng.integers(len(users), size=n)
    second_user = (first_user + rng.integers(1, len(users), size=n)) % len(users)
    second_user[y == 1] = first_user[y == 1]
    pairs = rng.integers(len(side_pairs), size=n)
    directions = rng.integers(2, size=n)
    left, right = [], []
    for u, v, p, direction in zip(first_user, second_user, pairs, directions):
        a, b = side_pairs[p][direction][users[u]], side_pairs[p][1 - direction][users[v]]
        left.append(a[rng.integers(len(a))])
        right.append(b[rng.integers(len(b))])
    return np.stack(left), np.stack(right), y


class Comparator:
    """Each instance is fitted once on development users; scoring never fits."""

    def __init__(self, name, seed=0):
        self.name, self.seed = name, seed
        self.info = {}

    def fit(self, x, users, *, side_pairs=None):
        if self.name == "bayes_factor":
            self.alpha, self.info = fit_prior(x)
            return self
        self.scaler = StandardScaler().fit(x)
        self.info = {"mean": self.scaler.mean_.tolist(), "scale": self.scaler.scale_.tolist()}
        z = self.scaler.transform(x)
        self.projection = None
        base = self.name.removesuffix("_rich")
        if base == "lda_cosine":
            # Ridge whitening keeps low-rank development sets well defined.
            means = np.stack([z[users == u].mean(0) for u in sorted(set(users))])
            centered = np.concatenate([
                z[users == u] - means[j] for j, u in enumerate(sorted(set(users)))
            ])
            within = centered.T @ centered / max(1, len(centered) - len(means))
            eigen, vectors = np.linalg.eigh(within)
            ridge = max(float(np.trace(within) / within.shape[0]), 1.0) * 1e-6
            whitening = vectors / np.sqrt(np.maximum(eigen, ridge))
            _, _, directions = np.linalg.svd(means @ whitening, full_matrices=False)
            rank = min(len(means) - 1, z.shape[1])
            self.projection = whitening @ directions[:rank].T
            self.info.update(within_covariance_ridge=ridge, lda_dimensions=rank)
        elif base == "pair_logistic":
            if side_pairs is None:
                raise ValueError("training pairs: supply the condition's physical A/B window sides")
            left, right, y = training_pairs(side_pairs, sorted(set(users)), self.seed)
            difference = np.abs(self.scaler.transform(left) - self.scaler.transform(right))
            self.model = LogisticRegression(C=1.0, max_iter=1000, random_state=self.seed)
            self.model.fit(difference, y)
            self.info.update(training_pairs=len(y))
        elif base != "cosine":
            raise ValueError(f"comparators: unknown {self.name}; choose a supported feature attack")
        if self.name.endswith("_rich"):
            self.info["fitted_parameters_sha256"] = digest({
                "mean": self.scaler.mean_.tolist(), "scale": self.scaler.scale_.tolist(),
                "projection": None if self.projection is None else self.projection.tolist(),
            })
        return self

    def score(self, x, y):
        if self.name == "bayes_factor":
            return log_bayes_factor(x, y, self.alpha)
        a, b = self.scaler.transform(x), self.scaler.transform(y)
        if self.name == "pair_logistic":
            return self.model.decision_function(np.abs(a - b))
        if self.projection is not None:
            a, b = a @ self.projection, b @ self.projection
        denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        return np.divide(np.sum(a * b, axis=1), denominator,
                         out=np.zeros(len(a)), where=denominator > 0)
