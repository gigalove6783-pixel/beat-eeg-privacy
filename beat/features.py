"""Covariance and tangent-space features shared by experiments A, B and C."""
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.geometry.base import invsqrtm
from pyriemann.geometry.mean import mean_riemann
from pyriemann.geometry.tangentspace import tangent_space

from .data import Trials, require_training


def covariances(trials: Trials, estimator: str = "oas") -> Trials:
    """Shrinkage spatial covariance per trial; X must be (n, channels, times)."""
    return trials.with_X(Covariances(estimator=estimator).transform(trials.X.astype(np.float64)))


class CommonTangentSpace:
    """Tangent space at the Riemannian mean of the training covariances, reference fixed."""

    def fit(self, train: Trials):
        require_training(train)
        self.ts_ = TangentSpace(metric="riemann", tsupdate=False).fit(train.X)
        return self

    def transform(self, trials: Trials) -> Trials:
        return trials.with_X(self.ts_.transform(trials.X))


class RecenteredTangentSpace:
    """Per-subject Riemannian recentering to the identity, then tangent space at the identity.

    The reference of subject s is the Riemannian mean of s's training covariances
    (Zanini et al., IEEE TBME 2018). Subjects absent from training raise KeyError.
    """

    def fit(self, train: Trials):
        require_training(train)
        self.whiteners_ = {
            s: invsqrtm(mean_riemann(train.X[train.subject == s])) for s in np.unique(train.subject)
        }
        return self

    def transform(self, trials: Trials) -> Trials:
        n = trials.X.shape[1]
        out = np.empty((len(trials), n * (n + 1) // 2))
        for s in np.unique(trials.subject):
            idx = trials.subject == s
            w = self.whiteners_[s]
            out[idx] = tangent_space(w @ trials.X[idx] @ w, np.eye(n), metric="riemann")
        return trials.with_X(out)
