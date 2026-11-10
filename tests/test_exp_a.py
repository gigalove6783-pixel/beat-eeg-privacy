import json

import numpy as np
import pandas as pd
import pytest

from beat.config import load_config
from beat.data import LeakageError, Trials
from beat.exp_a import DECODERS, band_power, make_decoder, run_experiment, subject_values, write_outputs
from beat.features import CommonTangentSpace, covariances
from beat.stats import sign_flip_pvalue

SFREQ = 250
CFG = load_config("exp_a")


def synthetic_signals(subjects=(1, 2), sessions=("T", "E"), n_channels=6, per_run=8, seed=0) -> Trials:
    """Left/right trials whose 10 Hz power is higher over channel 0 (left) or channel 1 (right)."""
    rng = np.random.default_rng(seed)
    t = np.arange(750) / SFREQ
    rows, X = [], []
    for subject in subjects:
        for session in sessions:
            for run in range(1, 7):
                for trial in range(1, per_run + 1):
                    label = trial % 2
                    x = rng.standard_normal((n_channels, t.size))
                    x[label] += 1.5 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
                    X.append(x)
                    rows.append(dict(subject=subject, session=session, run=run, trial=trial, label=label,
                                     artifact=False, finite=True))
    return Trials(np.stack(X).astype(np.float32), pd.DataFrame(rows))


@pytest.mark.parametrize("name", DECODERS)
@pytest.mark.parametrize("sessions", [("E",), ("T", "E")])
def test_decoder_fit_rejects_session_e(name, sessions):
    with pytest.raises(LeakageError):
        make_decoder(name, CFG, SFREQ).fit(synthetic_signals(subjects=(1,), sessions=sessions))


def test_feature_dimensions_match_plan():
    trials = synthetic_signals(subjects=(1,), sessions=("T",), n_channels=22, per_run=4)
    assert band_power(trials, CFG["b1_band_power"], SFREQ).shape == (len(trials), 44)
    cov = covariances(trials, CFG["b3_tangent_space"]["covariance_estimator"])
    assert CommonTangentSpace().fit(cov).transform(cov).X.shape == (len(trials), 253)
    assert make_decoder("B2", CFG, SFREQ).fit(trials).filters_.shape == (6, 22)


@pytest.fixture(scope="module")
def e_scores():
    trials = synthetic_signals()
    # Excluded trials must vanish from both calibration and test sets.
    excluded = ((trials.meta.session == "E") & (trials.meta.trial == 1)) | ((trials.meta.session == "T") & (trials.meta.trial == 2))
    trials.meta.loc[excluded & (trials.meta.run % 2 == 0), "artifact"] = True
    trials.meta.loc[excluded & (trials.meta.run % 2 == 1), "finite"] = False
    return run_experiment(trials, CFG, SFREQ)


def test_e_scoring_covers_every_combination_on_retained_e_trials(e_scores):
    counts = e_scores.groupby(["subject", "decoder", "n_runs"]).size().unstack()
    assert (counts[[1, 2, 3, 6]] == [6, 15, 20, 1]).all().all()
    assert len(counts) == 2 * len(DECODERS)
    assert (e_scores.test_session == "E").all()
    assert (e_scores.n_test == 6 * (8 - 1)).all()
    assert (e_scores.n_train == e_scores.n_runs * (8 - 1)).all()
    # Same calibration trials for every decoder.
    assert (e_scores.groupby(["subject", "runs"]).n_train.nunique() == 1).all()


def test_decoders_learn_synthetic_signal(e_scores):
    full = e_scores[e_scores.n_runs == 6]
    assert (full.ba > 0.8).all() and (full.kappa > 0.6).all()


def test_t_only_never_scores_session_e():
    scores = run_experiment(synthetic_signals(subjects=(1,), sessions=("T",)), CFG, SFREQ, t_only=True)
    assert (scores.test_session == "T").all()
    assert set(scores.n_runs) == {1, 2, 3}
    assert (scores.n_test == (6 - scores.n_runs) * 8).all()


def test_h1_and_outputs(e_scores, tmp_path):
    h1 = write_outputs(e_scores, CFG, tmp_path)
    per_subject = subject_values(e_scores)
    one = per_subject[per_subject.n_runs == 1].pivot(index="subject", columns="decoder", values="ba")
    diffs = (one["B3"] - one["B2"]).to_numpy()
    assert h1["p_value"] == pytest.approx(sign_flip_pvalue(diffs, "greater"))
    assert h1["mean_difference"] == pytest.approx(diffs.mean())
    assert h1["n_sign_patterns"] == 4 and h1["test_session"] == "E"
    assert json.loads((tmp_path / "h1.json").read_text())["p_value"] == h1["p_value"]
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert len(summary) == len(DECODERS) * 4 and (summary.n_subjects == 2).all()
    assert len(pd.read_csv(tmp_path / "scores.csv")) == len(e_scores)
    assert (tmp_path / "calibration_curve.png").stat().st_size > 0


def test_run_is_deterministic():
    trials = synthetic_signals(subjects=(1,))
    a = run_experiment(trials, CFG, SFREQ)
    b = run_experiment(trials, CFG, SFREQ)
    pd.testing.assert_frame_equal(a, b)
