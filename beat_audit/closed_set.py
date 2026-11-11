"""Enrolled identification, separate from user-disjoint linkage inference."""

import numpy as np
from scipy.special import gammaln
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def enrolled_log_likelihood(windows, enrollment, alpha):
    """[window,user] Dirichlet-categorical predictive, omitting user-constant terms.

    Tables are [window,cue,command] and [user,cue,command]; alpha broadcasts to
    [cue,command]. This routine is independent of beat.exp_k.bayes_scores.
    """
    windows, enrollment = np.asarray(windows), np.asarray(enrollment)
    a = enrollment + alpha
    result = np.empty((len(windows), len(enrollment)))
    for start in range(0, len(windows), 256):
        part = windows[start:start + 256, None]
        result[start:start + 256] = (
            (gammaln(a[None] + part) - gammaln(a)[None]).sum(axis=(2, 3)) +
            (gammaln(a.sum(-1))[None] -
             gammaln(a.sum(-1)[None] + part.sum(-1))).sum(-1)
        )
    return result


def identify(reference, reference_users, probe, probe_users, *, kind, alpha=None, seed=0):
    """Reference-side enrollment only; equal per-user accuracy on probe windows.

    Feature defaults: standardisation, C=1 logistic / RBF-SVM (gamma=scale),
    and MLP (100 hidden units, 500 iterations). No hyperparameter selection.
    """
    users = sorted(set(reference_users))
    if kind == "commands":
        counts = np.stack([reference[reference_users == u].sum(0) for u in users])
        score = enrolled_log_likelihood(probe, counts, alpha)
        maximum = score.max(axis=1, keepdims=True)
        ties = score == maximum
        truth = np.array([users.index(u) for u in probe_users])
        accuracy = ties[np.arange(len(probe)), truth] / ties.sum(axis=1)
        models = {"dirichlet_categorical": accuracy}
    else:
        models = {}
        for name, model in (
            ("logistic", LogisticRegression(C=1, max_iter=1000, random_state=seed)),
            ("rbf_svm", SVC(C=1, gamma="scale")),
            ("mlp", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=seed)),
        ):
            pipeline = make_pipeline(StandardScaler(), model)
            pipeline.fit(reference, reference_users)
            models[name] = (pipeline.predict(probe) == probe_users).astype(float)
    return {name: {u: float(value[probe_users == u].mean()) for u in users}
            for name, value in models.items()}
