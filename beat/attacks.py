"""Independent closed-set identity attackers, also used by Experiment C.

Only collected representations and identity labels enter fit. No T-only guard
belongs here: E-output collection is an explicitly permitted attack scenario.
SVM scaling and both hyperparameters are refit inside its training-only CV.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ATTACKERS = ("logistic", "rbf_svm", "mlp")


def fit_attackers(X, subject, config: dict, seed: int = 0) -> dict:
    """Fit three fresh attackers; all tuning sees only the supplied training X.

    Returns sklearn estimators keyed by ATTACKERS; each supports predict.
    Insufficient identities/examples for the specified inner CV is an error,
    never a fallback to tuning on the outer test split.

    The MLP pipeline also exposes convergence_ (converged, iterations, loss,
    warning). Its convergence warnings are recorded there instead of flooding
    stderr; experiment runners persist each fit and print the aggregate count.
    """
    X, subject = np.asarray(X), np.asarray(subject)
    _, counts = np.unique(subject, return_counts=True)
    if len(counts) < 2 or counts.min() < config["inner_folds"]:
        raise ValueError("Each identity needs at least inner_folds training examples")
    if X.ndim != 2 or len(X) != len(subject) or not np.isfinite(X).all():
        raise ValueError("Attack inputs must be finite aligned feature rows")

    def pipeline(classifier):
        return Pipeline([("scale", StandardScaler()), ("classifier", classifier)])

    estimators = {
        "logistic": pipeline(LogisticRegression(
            C=config["logistic_C"], max_iter=config["logistic_max_iter"],
            solver="lbfgs", random_state=seed)),
        "rbf_svm": GridSearchCV(
            pipeline(SVC(kernel="rbf")),
            {"classifier__C": config["svm_C"], "classifier__gamma": config["svm_gamma"]},
            scoring="balanced_accuracy",
            cv=StratifiedKFold(config["inner_folds"], shuffle=True, random_state=seed),
            n_jobs=1, refit=True, error_score="raise"),
        "mlp": pipeline(MLPClassifier(
            hidden_layer_sizes=tuple(config["mlp_hidden"]), alpha=config["mlp_alpha"],
            solver=config["mlp_solver"], max_iter=config["mlp_max_iter"],
            max_fun=config["mlp_max_fun"], random_state=seed)),
    }
    for name, estimator in estimators.items():
        if name != "mlp":
            estimator.fit(X, subject)
            continue
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(X, subject)
        messages = [str(w.message) for w in observed if issubclass(w.category, ConvergenceWarning)]
        for warning in observed:
            if not issubclass(warning.category, ConvergenceWarning):
                warnings.warn(warning.message, warning.category, stacklevel=2)
        estimator.convergence_ = dict(
            converged=not messages, iterations=int(estimator["classifier"].n_iter_),
            loss=float(estimator["classifier"].loss_), warning=" | ".join(messages))
    return estimators
