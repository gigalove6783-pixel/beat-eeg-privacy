import numpy as np
import pandas as pd
import pytest

from beat.config import ROOT, load_config
from beat.data import LeakageError, Trials, load_trials, quality_mask, require_training
from beat.features import CommonTangentSpace, RecenteredTangentSpace, covariances
from beat.splits import t_run_folds
from beat.stats import sign_flip_pvalue


def synthetic_trials(session: str, n_per_subject: int = 12, n_channels: int = 4) -> Trials:
    rng = np.random.default_rng(0)
    rows, X = [], []
    for subject in (1, 2):
        for i in range(n_per_subject):
            rows.append(dict(subject=subject, session=session, run=i % 6 + 1, trial=i + 1,
                             label=i % 2, artifact=False, finite=True))
            X.append(rng.standard_normal((n_channels, 100)))
    return covariances(Trials(np.stack(X), pd.DataFrame(rows)))


@pytest.mark.parametrize("model", [CommonTangentSpace, RecenteredTangentSpace])
def test_fit_rejects_session_e(model):
    with pytest.raises(LeakageError):
        model().fit(synthetic_trials("E"))


def test_require_training_rejects_mixed_sessions():
    mixed = Trials(np.zeros(2), pd.DataFrame({"session": ["T", "E"]}))
    with pytest.raises(LeakageError):
        require_training(mixed)


def test_t_run_folds_reject_session_e():
    with pytest.raises(LeakageError):
        next(t_run_folds(synthetic_trials("E")))


def test_recentering_maps_each_training_mean_to_origin():
    train = synthetic_trials("T")
    features = RecenteredTangentSpace().fit(train).transform(train).X
    for s in (1, 2):
        # The Riemannian mean recentred to the identity has a tangent vector at the origin;
        # the arithmetic mean of tangent vectors is close to it.
        assert np.abs(features[train.subject == s].mean(axis=0)).max() < 0.05


def test_sign_flip_exact_extremes():
    assert sign_flip_pvalue(np.ones(9)) == pytest.approx(1 / 512)
    assert sign_flip_pvalue(-np.ones(9)) == 1.0


@pytest.mark.skipif(not (ROOT / load_config("data")["raw_dir"]).exists(), reason="raw data not downloaded")
def test_loaded_trial_counts_and_shape():
    trials = load_trials()
    assert trials.X.shape[1:] == (22, 750)
    counts = trials.meta.groupby(["subject", "session", "run"]).size()
    assert len(counts) == 9 * 2 * 6 and (counts == 24).all()
    assert (trials.meta.groupby(["subject", "session"])["label"].mean() == 0.5).all()
    assert quality_mask(trials).sum() > 0.85 * len(trials)
