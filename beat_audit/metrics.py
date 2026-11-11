"""Descriptive weighted score metrics and development-only estimated log LR."""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve


def normalized(weights, n):
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    return w / w.sum()


def roc_metrics(mated, nonmated, wm=None, wn=None):
    wm, wn = normalized(wm, len(mated)), normalized(wn, len(nonmated))
    fpr, tpr, thresholds = roc_curve(
        np.r_[np.ones(len(mated)), np.zeros(len(nonmated))],
        np.r_[mated, nonmated], sample_weight=np.r_[wm, wn], drop_intermediate=False,
    )
    delta = fpr + tpr - 1
    high = int(np.searchsorted(delta, 0))
    low = max(0, high - 1)
    fraction = 0 if delta[high] == delta[low] else -delta[low] / (delta[high] - delta[low])
    eer = fpr[low] + fraction * (fpr[high] - fpr[low])
    # A deployable threshold must be an actual deterministic score cut, not the
    # randomised interpolated EER used as the descriptive scalar.
    j = int(np.argmin(np.abs(delta)))
    threshold = thresholds[j]
    if not np.isfinite(threshold):
        threshold = np.nextafter(max(np.max(mated), np.max(nonmated)), np.inf)
    return {
        "eer": float(eer),
        "tpr_at_fpr_01": float(np.interp(0.01, fpr, tpr)),
        "tpr_at_fpr_10": float(np.interp(0.10, fpr, tpr)),
        "threshold": float(threshold),
        "fpr": fpr.tolist(), "fnmr": (1 - tpr).tolist(),
    }


class Calibration:
    """Equal-class-total logistic calibration; its logit is an estimated log LR."""

    def fit(self, mated, nonmated, wm=None, wn=None):
        x = np.r_[mated, nonmated]
        self.mean, self.scale = x.mean(), max(float(x.std()), 1e-12)
        weights = np.r_[normalized(wm, len(mated)), normalized(wn, len(nonmated))]
        self.model = LogisticRegression(C=1e6, max_iter=1000, tol=1e-9)
        self.model.fit(((x - self.mean) / self.scale)[:, None],
                       np.r_[np.ones(len(mated)), np.zeros(len(nonmated))],
                       sample_weight=weights * len(x) / 2)
        return self

    def predict(self, score):
        return self.model.decision_function(
            ((np.asarray(score) - self.mean) / self.scale)[:, None]
        )


def cllr(mated_lr, nonmated_lr, wm=None, wn=None):
    return float(0.5 * (
        np.dot(normalized(wm, len(mated_lr)), np.logaddexp(0, -np.asarray(mated_lr))) +
        np.dot(normalized(wn, len(nonmated_lr)), np.logaddexp(0, np.asarray(nonmated_lr)))
    ) / np.log(2))


def cllr_min(mated, nonmated, wm=None, wn=None):
    x = np.r_[mated, nonmated]
    labels = np.r_[np.ones(len(mated)), np.zeros(len(nonmated))]
    weights = np.r_[normalized(wm, len(mated)), normalized(wn, len(nonmated))]
    probability = IsotonicRegression(out_of_bounds="clip").fit_transform(
        x, labels, sample_weight=weights,
    )
    # Only observed-class loss is evaluated; zero-probability unobserved classes
    # are not clipped into artificial calibration loss.
    p = probability[:len(mated)]
    q = probability[len(mated):]
    return float(-0.5 * (np.dot(weights[:len(mated)], np.log2(p)) +
                        np.dot(weights[len(mated):], np.log2(1 - q))))


def d_sys(mated, nonmated, wm=None, wn=None):
    x, y = np.asarray(mated), np.asarray(nonmated)
    unique = np.unique(np.r_[x, y])
    if len(unique) <= 50:
        pm = np.bincount(np.searchsorted(unique, x), weights=normalized(wm, len(x)),
                         minlength=len(unique))
        pn = np.bincount(np.searchsorted(unique, y), weights=normalized(wn, len(y)),
                         minlength=len(unique))
    else:
        edges = np.linspace(unique[0], unique[-1], 51)
        pm = np.histogram(x, bins=edges, weights=normalized(wm, len(x)))[0]
        pn = np.histogram(y, bins=edges, weights=normalized(wn, len(y)))[0]
    local = np.divide(pm - pn, pm + pn, out=np.zeros_like(pm), where=(pm + pn) > 0)
    return float(np.dot(pm, np.maximum(0, local)))


def pooled_metrics(mated, nonmated, wm, wn, lm, ln):
    result = roc_metrics(mated, nonmated, wm, wn)
    result.update(D_sys=d_sys(mated, nonmated, wm, wn),
                  Cllr=cllr(lm, ln, wm, wn),
                  Cllr_min=cllr_min(mated, nonmated, wm, wn),
                  calibration="estimated log likelihood ratio (natural log); transfer-dependent")
    return result
