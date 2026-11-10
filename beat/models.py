"""T-guarded defended models and the immutable inference object used by B/C."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn

from .data import Trials, require_training
from .features import CommonTangentSpace, RecenteredTangentSpace


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.strength * grad, None


def reverse_gradient(x, strength):
    return _Reverse.apply(x, strength)


def ramp(target: float, progress: float) -> float:
    return float(target * (2 / (1 + np.exp(-10 * progress)) - 1))


class Encoder(nn.Module):
    def __init__(self, dimension, identities, config):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dimension, config["encoder_hidden"]), nn.ReLU(),
            nn.Linear(config["encoder_hidden"], config["bottleneck"]))
        self.task = nn.Linear(config["bottleneck"], 2)
        self.adversary = nn.Sequential(
            nn.Linear(config["bottleneck"], config["identity_hidden"]), nn.ReLU(),
            nn.Linear(config["identity_hidden"], identities))

    def forward(self, x, strength=0.0):
        representation = self.encoder(x)
        return (representation, self.task(representation),
                self.adversary(reverse_gradient(representation, strength)))


class FeatureMap:
    def __init__(self, recentered: bool):
        self.reference = RecenteredTangentSpace() if recentered else CommonTangentSpace()
        self.scale = StandardScaler()

    def fit(self, train: Trials):
        require_training(train)
        self.reference.fit(train)
        self.scale.fit(self.reference.transform(train).X)
        return self

    def transform(self, trials: Trials):
        tangent = self.reference.transform(trials).X
        return tangent, self.scale.transform(tangent).astype(np.float32)


def initialized_encoder(dimension, identities, config, seed):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return Encoder(dimension, identities, config)


def fit_encoder(train: Trials, config: dict, seed: int, epochs: int, target: float,
                checkpoints=()):
    """Train already-standardized T features, returning checkpoint state dicts.

    For a fixed seed all lambdas start with the same full network and minibatch
    order. P2 prefixes do not depend on max_epochs, so a pilot checkpoint is
    identical to training P2 for just that chosen budget.
    """
    require_training(train)
    if epochs < 1:
        raise ValueError("epochs must be positive")
    _, identity = np.unique(train.subject, return_inverse=True)
    model = initialized_encoder(train.X.shape[1], len(np.unique(identity)), config, seed)
    if config["optimizer"] != "adam":
        raise ValueError("Only the prespecified adam optimizer is supported")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    X = torch.as_tensor(train.X, dtype=torch.float32)
    y = torch.as_tensor(train.y, dtype=torch.long)
    identity = torch.as_tensor(identity, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    steps = (len(train) + config["batch_size"] - 1) // config["batch_size"]
    snapshots = {}
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator)
        for batch, start in enumerate(range(0, len(train), config["batch_size"])):
            idx = order[start:start + config["batch_size"]]
            progress = ((epoch - 1) * steps + batch) / max(epochs * steps - 1, 1)
            _, logits, identities = model(X[idx], ramp(target, progress))
            loss = nn.functional.cross_entropy(logits, y[idx])
            if target:
                loss = loss + nn.functional.cross_entropy(identities, identity[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch in checkpoints or epoch == epochs:
            snapshots[epoch] = copy.deepcopy(model.state_dict())
    return snapshots


@dataclass
class FrozenModel:
    """Inference only. Input: covariance Trials. Output: (representation, logits).

    P0 exposes unstandardized tangent vectors, with standardization internal to
    its classifier. P2/P3 expose the 8-d bottleneck. Logit columns are left (0),
    right (1). Rows with non-finite covariance entries remain NaN, preserving
    chronological alignment for C's hold rule. Artifact flags never drop rows.
    """
    features: FeatureMap
    predictor: object
    neural: bool
    identity_classes_: np.ndarray | None = None

    def __post_init__(self):
        if self.neural:
            self.predictor.eval()
            self.predictor.requires_grad_(False)

    def transform(self, trials: Trials):
        if trials.X.ndim != 3 or trials.X.shape[1] != trials.X.shape[2]:
            raise ValueError("FrozenModel expects covariance Trials, not EEG signals")
        valid = np.isfinite(trials.X).all(axis=(1, 2))
        n_channels = trials.X.shape[1]
        dimension = (self.predictor.task.in_features if self.neural
                     else n_channels * (n_channels + 1) // 2)
        representations = np.full((len(trials), dimension), np.nan)
        logits = np.full((len(trials), 2), np.nan)
        if valid.any():
            tangent, X = self.features.transform(trials.subset(valid))
            if self.neural:
                with torch.inference_mode():
                    z, scores, _ = self.predictor(torch.as_tensor(X))
                representations[valid], logits[valid] = z.numpy(), scores.numpy()
            else:
                scores = self.predictor.decision_function(X)
                representations[valid] = tangent
                logits[valid] = np.column_stack([-scores / 2, scores / 2])
        return representations, logits

    __call__ = transform

    def adversary_logits(self, trials: Trials):
        """Diagnostic identity-head scores; never independent protection evidence.

        Columns follow identity_classes_. The ordinary call remains the two-item
        (representation, task logits) API used by Experiment C.
        """
        if not self.neural:
            raise ValueError("Linear P0 models have no adversary head")
        representation, _ = self(trials)
        valid = np.isfinite(representation).all(axis=1)
        scores = np.full((len(trials), len(self.identity_classes_)), np.nan)
        if valid.any():
            with torch.inference_mode():
                scores[valid] = self.predictor.adversary(
                    torch.as_tensor(representation[valid], dtype=torch.float32)).numpy()
        return scores


def freeze_encoder(features, state, dimension, subjects, config):
    model = Encoder(dimension, len(subjects), config)
    model.load_state_dict(state)
    return FrozenModel(features, model, True, np.asarray(subjects).copy())


def fit_linear(train: Trials, features: FeatureMap, config: dict, seed: int):
    require_training(train)
    _, X = features.transform(train)
    predictor = LogisticRegression(C=config["linear_C"], max_iter=config["linear_max_iter"],
                                   solver="lbfgs", random_state=seed).fit(X, train.y)
    return FrozenModel(features, predictor, False)


def fit_defended(train: Trials, condition: str, config: dict, seed: int,
                 epochs: int, target: float = 0.0) -> FrozenModel:
    """Fit a whole defended pipeline; session E is rejected before preprocessing."""
    require_training(train)
    if condition not in ("P0", "P0-RC", "P2", "P2-RC", "P3", "P3-RC"):
        raise ValueError(f"Unknown condition {condition}")
    features = FeatureMap(condition.endswith("-RC")).fit(train)
    if condition.startswith("P0"):
        return fit_linear(train, features, config, seed)
    _, X = features.transform(train)
    state = fit_encoder(train.with_X(X), config, seed, epochs,
                        target if condition.startswith("P3") else 0.0)[epochs]
    return freeze_encoder(features, state, X.shape[1], np.unique(train.subject), config)
