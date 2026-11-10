"""Experiment C contracts. The final path runs only on synthetic T/E fixtures."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from beat.config import load_config
from beat.data import LeakageError, Trials, bandpass_sos, causal_bandpass
from beat.exp_c import (HOLD, LEFT, RIGHT, FiniteGate, RejectionRule, _attack_job, _sorted,
                        bootstrap_seed_mean, command_descriptors, control_pass, decide,
                        feature_descriptors, fit_temperature, gate_from_dict, identify_head,
                        make_gate, observation_groups, replay_condition, run_dev, run_final,
                        score_episodes, summarize_observation, trial_covariances, variant)

PER_RUN = 10


def synthetic_recordings(sessions=("T", "E"), subjects=9):
    """Raw continuous runs plus the cached trials epoched from them as beat.data does."""
    sos = bandpass_sos(load_config("data"))
    lead, gap, length = 40, 30, 64
    rows, X, raw_runs = [], [], {}
    for session in sessions:
        rng = np.random.default_rng(1 if session == "T" else 2)
        for subject in range(1, subjects + 1):
            for run in range(1, 7):
                labels = rng.permutation(np.arange(PER_RUN) % 2)
                raw = rng.normal(size=(lead + PER_RUN * (length + gap), 4))
                raw *= np.array([1.0, 1.0, np.exp(subject / 6), np.exp(-subject / 7)])
                starts = {}
                for trial, label in enumerate(labels, start=1):
                    a = starts[trial] = lead + (trial - 1) * (length + gap)
                    raw[a:a + length, int(label)] *= 1.2
                    finite = not (subject == 2 and run == 3 and trial == 4)
                    if not finite:
                        raw[a + 5, 1] = np.nan
                    rows.append(dict(subject=subject, session=session, run=run,
                                     trial=trial, label=int(label),
                                     artifact=bool(rng.random() < 0.05), finite=finite))
                filtered = causal_bandpass(sos, np.where(np.isfinite(raw), raw, 0.0))
                X.extend(filtered[a:a + length].T for a in starts.values())
                raw_runs[(subject, session, run)] = (raw, starts)
    return Trials(np.asarray(X, dtype=np.float32), pd.DataFrame(rows)), raw_runs


def synthetic_signals(sessions=("T", "E"), subjects=9):
    return synthetic_recordings(sessions, subjects)[0]


class FakeModel:
    """Affine task head on a 4-d representation, NaN rows for non-finite covariances."""

    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.rotation = np.linalg.qr(rng.normal(size=(4, 4)))[0]
        base = np.array([[2.0, -2.0], [-2.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
        self.weights, self.bias = self.rotation.T @ base, np.array([0.1, -0.1])

    def __call__(self, covs):
        valid = np.isfinite(covs.X).all(axis=(1, 2))
        rep, logits = np.full((len(covs), 4), np.nan), np.full((len(covs), 2), np.nan)
        rep[valid] = np.log(np.diagonal(covs.X[valid], axis1=1, axis2=2)) @ self.rotation
        logits[valid] = rep[valid] @ self.weights + self.bias
        return rep, logits


class FakeSource:
    def __init__(self, signals, estimator="oas"):
        self.covs = trial_covariances(_sorted(signals), estimator)
        self.models = {"P0": FakeModel(0), "P2": FakeModel(1)}

    def _mapping(self, mask, condition):
        covs = self.covs.subset(mask)
        rep, logits = self.models[condition](covs)
        out = {c: covs.meta[c].to_numpy() for c in covs.meta.columns}
        out["session"] = out["session"].astype("U1")
        return dict(out, representation=rep, logits=logits)

    def oof_manifest(self):
        return dict(schema=1, selected={"P0": 0.0, "P2": 0.0, "P3": None}, seeds=[0])

    def oof(self, condition, seed):
        meta = self.covs.meta
        out = self._mapping((meta.session == "T") & ~meta.artifact & meta.finite, condition)
        out.pop("representation")
        return out

    def frozen(self, condition, seed):
        return self.models[condition]

    def outputs(self, condition, seed):
        return self._mapping(np.ones(len(self.covs), dtype=bool), condition)


def small_config():
    cfg = copy.deepcopy(load_config("exp_c"))
    cfg.update(conditions=["P0", "P2", "P3"], taus=[0.5, 0.7], workers=1)
    cfg["controller"].update(episode_length=5)
    cfg["observation"].update(k=[1, 5])
    cfg["statistics"].update(bootstrap_samples=20)
    return cfg


def small_attackers():
    cfg = copy.deepcopy(load_config("exp_b")["attackers"])
    cfg.update(svm_C=[1.0], svm_gamma=["scale"], inner_folds=2, mlp_max_iter=200)
    return cfg


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("exp_c")
    (signals, raw_runs), config = synthetic_recordings(), small_config()
    data_config = load_config("data")
    source = FakeSource(signals)
    t = signals.subset(signals.session == "T")
    calibration = run_dev(t, config, source, root / "results", root / "artifacts", data_config,
                          raw_runs)
    summary = run_final(signals, calibration, config, source, small_attackers(),
                        root / "results", root / "artifacts", data_config, raw_runs)
    return dict(root=root, signals=signals, raw_runs=raw_runs, config=config, source=source,
                calibration=calibration, summary=summary, data_config=data_config)


def test_t_only_fit_steps_reject_e(tmp_path):
    e = synthetic_signals(("E",), subjects=3)
    covs = trial_covariances(e, "oas")
    with pytest.raises(LeakageError):
        fit_temperature(e.with_X(np.zeros((len(e), 2))), [0.05, 20])
    with pytest.raises(LeakageError):
        RejectionRule(3.0).fit(covs)
    with pytest.raises(LeakageError):
        FiniteGate().fit(covs)
    with pytest.raises(LeakageError):
        identify_head(e.with_X(np.ones((len(e), 4))), np.ones((len(e), 2)), 1e-3)
    with pytest.raises(LeakageError):
        run_dev(e, small_config(), FakeSource(e), tmp_path / "r", tmp_path / "a")


def test_temperature_recovers_generating_scale():
    rng = np.random.default_rng(0)
    margin = rng.normal(0, 4, 20000)
    labels = (rng.random(20000) < 1 / (1 + np.exp(-margin / 2.5))).astype(int)
    meta = pd.DataFrame(dict(subject=1, session="T", run=1, trial=1, label=labels,
                             artifact=False, finite=True))
    logits = np.column_stack([-margin / 2, margin / 2])
    assert fit_temperature(Trials(logits, meta), [0.05, 20]) == pytest.approx(2.5, rel=0.08)


def test_decide_holds_below_tau_and_on_gate():
    logits = np.array([[0.0, 0.1], [0.0, 3.0], [3.0, 0.0], [0.0, 0.0], [np.nan, 1.0],
                       [0.0, 3.0]])
    gate = np.array([True, True, True, True, True, False])
    assert decide(logits, 1.0, 0.5, gate).tolist() == [RIGHT, RIGHT, LEFT, LEFT, HOLD, HOLD]
    assert decide(logits, 1.0, 0.6, gate).tolist() == [HOLD, RIGHT, LEFT, HOLD, HOLD, HOLD]
    # A larger temperature flattens probabilities and turns weak moves into holds.
    assert decide(logits, 10.0, 0.6, gate).tolist() == [HOLD, HOLD, HOLD, HOLD, HOLD, HOLD]


def test_gate_options():
    signals = synthetic_signals(("T",), subjects=2)
    covs = trial_covariances(_sorted(signals), "oas")
    finite = np.isfinite(covs.X).all(axis=(1, 2))
    assert not finite.all()
    drifted = covs.with_X(np.where(finite[:, None, None], covs.X * 50.0, covs.X))
    config = load_config("exp_c")
    assert config["rejection"]["gate"] == "none"
    none = gate_from_dict(make_gate(config["rejection"]).fit(covs).to_dict())
    sealed_config = variant("sealed_gate", config)[0]
    sealed = gate_from_dict(make_gate(sealed_config["rejection"]).fit(covs).to_dict())
    assert isinstance(none, FiniteGate) and isinstance(sealed, RejectionRule)
    # `none` holds only non-finite trials, however far a trial drifts.
    np.testing.assert_array_equal(none.accept(drifted), finite)
    assert not sealed.accept(drifted).any()
    assert sealed.accept(covs).sum() > 0.8 * finite.sum()
    with pytest.raises(ValueError, match="Unknown gate"):
        make_gate(dict(gate="other", z=3.0))


def test_attack_cell_not_evaluable_when_identity_lacks_training_groups():
    ordered = _sorted(synthetic_signals())
    meta_t = ordered.meta[ordered.session == "T"].reset_index(drop=True)
    meta_e = ordered.meta[ordered.session == "E"].reset_index(drop=True)
    rng = np.random.default_rng(0)

    def stream(meta):
        return meta.subject.to_numpy()[:, None] * np.ones(3) + rng.normal(size=(len(meta), 3))

    stream_t, stream_e = stream(meta_t), stream(meta_e)
    stream_e[(meta_e.subject == 5) & (meta_e.run <= 3)] = np.nan  # all held: no groups
    tags = dict(condition="P0", seed=0, stream="feature")
    predictions, rows = _attack_job(tags, "feature", 1, stream_t, stream_e, meta_t, meta_e,
                                    small_attackers(), 0, 1, 1)
    rows = pd.DataFrame(rows)
    blocked = rows[~rows.evaluable]
    # Direction 0 trains on E runs 1-3, where subject 5 has no group.
    assert set(zip(blocked.scenario, blocked.direction)) == {("main", 0)}
    assert set(blocked.attacker) == {"logistic", "rbf_svm", "mlp"}
    assert (blocked.short_identities == "5:0").all() and blocked.ba.isna().all()
    assert rows[rows.evaluable].ba.notna().all()
    streams = {"feature": dict(mode="feature", tau=None)}
    config = small_config()
    _, seed_mean, strongest, threshold, _ = summarize_observation(rows, predictions, streams,
                                                                 config)
    assert seed_mean[seed_mean.scenario == "main"].ba_mean.isna().all()
    best = strongest.iloc[0]
    assert best.scenario == "auxiliary" and best.not_evaluable_cells == 3
    assert np.isfinite([best.ba_mean, best.ba_low, best.ba_high]).all()


def test_arm_and_episode_metrics_by_hand():
    controller = dict(step_deg=5.0, range_deg=10.0, episode_length=4, reach_tolerance_deg=10.0)
    code = {RIGHT: [0.0, 5.0], LEFT: [5.0, 0.0], HOLD: [0.0, 0.0]}
    commands = [RIGHT, RIGHT, RIGHT, LEFT, LEFT, HOLD, RIGHT, RIGHT, LEFT, LEFT]
    labels = [1, 1, 0, 0, 1, 1, 1, 1, 0, 0]
    meta = pd.DataFrame(dict(subject=1, session="E", run=1, trial=range(1, 11), label=labels,
                             artifact=False, finite=True))
    starts = np.arange(10) % 4 == 0
    got, angles, blocked, ms = control_pass(np.array([code[c] for c in commands]), 1.0, 0.6,
                                            starts, controller)
    assert got.tolist() == commands
    assert angles.tolist() == [5, 10, 10, 5, -5, -5, 0, 5, -5, -10]
    assert blocked.tolist() == [False, False, True] + [False] * 7
    assert (ms >= 0).all()
    episodes = score_episodes(meta, got, angles, blocked, np.zeros(10, bool), controller)
    assert len(episodes) == 2  # trailing 2-trial block is replayed but not scored
    first, second = episodes.iloc[0], episodes.iloc[1]
    # The blocked wrong-direction attempt (trial 3) counts as out-of-range, not as a move.
    assert (first.mae_deg, first.end_error_deg, first.wrong_direction_moves,
            first.out_of_range_commands, first.hold_rate, first.effective_moves,
            first.unnecessary_movement_deg, first.reached) == (2.5, 5, 0, 1, 0, 3, 0, True)
    assert (second.mae_deg, second.end_error_deg, second.wrong_direction_moves,
            second.out_of_range_commands, second.hold_rate, second.effective_moves,
            second.unnecessary_movement_deg, second.reached) == (10, 5, 1, 0, 0.25, 3, 5, True)


def test_command_descriptors_by_hand():
    groups = np.arange(5)[None]
    X, keep = command_descriptors(np.array([RIGHT, RIGHT, HOLD, LEFT, LEFT]), groups)
    expected = [0.2, 0.0, 5 / 3, 2, 0.25, 0, 0, 0.25, 0, 0, 0, 0.25, 0.25]
    np.testing.assert_allclose(X[0], expected)
    assert keep.all()


def test_observation_groups_slide_for_training_and_tile_for_testing():
    meta = pd.DataFrame(dict(subject=[1] * 20, session="E", run=[1] * 10 + [2] * 10,
                             trial=list(range(1, 11)) * 2))
    train, train_info = observation_groups(meta, 5, 1)
    test, test_info = observation_groups(meta, 5, 5)
    assert train.shape == (12, 5) and test.shape == (4, 5)
    assert test.tolist() == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14],
                             [15, 16, 17, 18, 19]]
    assert (meta.run.to_numpy()[train] == train_info.run.to_numpy()[:, None]).all()
    messages = np.arange(40, dtype=float).reshape(20, 2)
    messages[1] = np.nan
    X, keep = feature_descriptors(messages, test[:1])
    np.testing.assert_allclose(X[0], np.r_[messages[[0, 2, 3, 4]].mean(0),
                                           messages[[0, 2, 3, 4]].std(0)])
    messages[:5] = np.nan
    assert not feature_descriptors(messages, test[:1])[1][0]


def test_bootstrap_interval_contains_point():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(dict(subject=np.repeat(np.arange(1, 4), 30),
                              run=np.tile(np.repeat(np.arange(1, 4), 10), 3), direction=0))
    frame["prediction"] = np.where(rng.random(90) < 0.6, frame.subject, 1)
    point = np.mean([(g.prediction == s).mean() for s, g in frame.groupby("subject")])
    low, high = bootstrap_seed_mean([frame, frame], 500, 0, 0.95)
    assert low <= point <= high and low < high


def test_bootstrap_interval_ignores_duplicated_seeds():
    # Seeds predict the same recorded runs, so identical seeds add no information.
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(dict(subject=np.repeat(np.arange(1, 4), 30),
                              run=np.tile(np.repeat(np.arange(1, 4), 10), 3), direction=0))
    frame["prediction"] = np.where(rng.random(90) < 0.5, frame.subject, 2)
    single = bootstrap_seed_mean([frame], 500, 0, 0.95)
    assert bootstrap_seed_mean([frame] * 5, 500, 0, 0.95) == pytest.approx(single, abs=1e-12)


def test_labels_never_change_commands(pipeline):
    signals, config, source = pipeline["signals"], pipeline["config"], pipeline["source"]
    calibration, data_config = pipeline["calibration"], pipeline["data_config"]
    rule = gate_from_dict(calibration["rejection"])
    ordered = _sorted(signals)
    t, e = ordered.subset(ordered.session == "T"), ordered.subset(ordered.session == "E")
    t_gate = rule.accept(trial_covariances(t, "oas"))
    shuffled_meta = e.meta.copy()
    shuffled_meta["label"] = np.random.default_rng(5).permutation(shuffled_meta.label)
    runs = [replay_condition(source.frozen("P0", 0), source.outputs("P0", 0), trials,
                             pipeline["raw_runs"], t.meta, t_gate, rule, calibration["temperatures"]["P0"]["0"],
                             bandpass_sos(data_config), config, "oas")
            for trials in (e, Trials(e.X, shuffled_meta))]
    columns = ["subject", "run", "trial", "tau", "mode", "command", "angle", "gated"]
    pd.testing.assert_frame_equal(runs[0]["trials"][columns], runs[1]["trials"][columns])


def test_final_produces_every_output(pipeline):
    results, artifacts = pipeline["root"] / "results", pipeline["root"] / "artifacts"
    for name in ("control_episodes.csv", "control_metrics.csv", "control_summary.csv",
                 "timing_summary.csv", "timing_by_seed.csv", "transmission_check.csv",
                 "replay_consistency.csv", "observation_directional.csv",
                 "observation_identity.csv", "observation_seed_mean.csv",
                 "observation_strongest.csv", "observation_threshold.csv",
                 "observation_feature_exclusions.csv", "observation_not_evaluable.csv",
                 "rejection_rates.csv", "observation_length.png", "summary.json",
                 "dev_temperatures.csv",
                 "dev_rejection.csv", "dev_replay_metrics.csv", "dev_timing.csv"):
        assert (results / name).exists(), name
    for name in ("replay_trials.csv.gz", "attack_predictions.csv.gz", "manifest.json"):
        assert (artifacts / "final" / name).exists(), name
    summary = json.loads((results / "summary.json").read_text())
    assert summary["no_operating_point"] == ["P3"]
    assert summary["transmission_check_passed"] and summary["transmission_command_mismatches"] == 0
    assert summary["replay_matches_saved"]
    assert summary["stream_matches_cache"]
    # Primary gate `none`: only the one non-finite trial per session is held.
    assert summary["gate"] == "none" and summary["not_evaluable_attack_cells"] == 0
    rejection = pd.read_csv(results / "rejection_rates.csv")
    assert set(rejection.session) == {"T", "E"} and (rejection.rejected_finite == 0).all()
    assert (rejection.nonfinite.sum(), (rejection.hold_rate * rejection.trials).sum()) == (2, 2)
    dev = json.loads((results / "dev_summary.json").read_text())
    assert dev["stream_matches_cache"] and dev["stream_trials"] == 9 * 6 * PER_RUN
    exclusions = pd.read_csv(results / "observation_feature_exclusions.csv")
    assert len(exclusions) == 2 * 2 * 2 and (exclusions.excluded_groups_mean >= 0).all()
    metrics = pd.read_csv(results / "control_metrics.csv")
    assert len(metrics) == 9 * 2 * 1 * 2 * 2
    keys = ["condition", "seed", "tau", "subject"]
    command = metrics[metrics["mode"] == "command"].sort_values(keys).reset_index(drop=True)
    feature = metrics[metrics["mode"] == "feature"].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(command.drop(columns="mode"), feature.drop(columns="mode"))
    tau5 = metrics[(metrics.tau == 0.5) & (metrics["mode"] == "command")]
    tau7 = metrics[(metrics.tau == 0.7) & (metrics["mode"] == "command")]
    assert tau7.hold_rate.mean() > tau5.hold_rate.mean()
    strongest = pd.read_csv(results / "observation_strongest.csv")
    assert len(strongest) == 2 * 3 * 2
    assert set(strongest.scenario) <= {"main", "auxiliary"}
    assert (strongest.ba_low <= strongest.ba_mean + 1e-12).all()
    assert (strongest.ba_mean <= strongest.ba_high + 1e-12).all()
    feature_rows = strongest[strongest.stream == "feature"]
    assert (feature_rows.ba_mean > 2 / 9).all()  # synthetic subject gains leak identity
    threshold = pd.read_csv(results / "observation_threshold.csv")
    assert len(threshold) == 2 * 3
    assert (threshold[threshold.stream == "feature"].smallest_k == 1).all()
    directional = pd.read_csv(results / "observation_directional.csv")
    assert set(directional.attacker) == {"logistic", "rbf_svm", "mlp"}
    assert set(directional[directional.scenario == "main"].direction) == {0, 1}
    timing = pd.read_csv(results / "timing_summary.csv")
    assert (timing.total_p95_ms >= timing.total_median_ms).all()
    assert (timing.inference_median_ms > 0).all() and (timing.filter_median_ms > 0).all()


def test_sealed_gate_variant_runs_end_to_end(pipeline, tmp_path):
    config = variant("sealed_gate", pipeline["config"])[0]
    signals, source = pipeline["signals"], pipeline["source"]
    args = dict(result_dir=tmp_path / "r", artifact_dir=tmp_path / "a",
                data_config=pipeline["data_config"], raw_runs=pipeline["raw_runs"])
    calibration = run_dev(signals.subset(signals.session == "T"), config, source, **args)
    assert calibration["rejection"]["gate"] == "riemann_distance"
    with pytest.raises(ValueError, match="config changed"):
        run_final(signals, calibration, pipeline["config"], source, small_attackers(), **args)
    summary = run_final(signals, calibration, config, source, small_attackers(), **args)
    assert summary["gate"] == "riemann_distance" and summary["transmission_check_passed"]
    rejection = pd.read_csv(tmp_path / "r" / "rejection_rates.csv")
    assert rejection.threshold.notna().all() and len(rejection) == 2 * 9
    assert (tmp_path / "r" / "observation_not_evaluable.csv").exists()


def test_final_guards(pipeline, tmp_path):
    signals, calibration = pipeline["signals"], pipeline["calibration"]
    args = dict(source=pipeline["source"], attackers=small_attackers(),
                result_dir=tmp_path / "r", artifact_dir=tmp_path / "a",
                data_config=pipeline["data_config"])
    with pytest.raises(ValueError, match="both T and E"):
        run_final(signals.subset(signals.session == "T"), calibration, pipeline["config"], **args)
    changed = copy.deepcopy(pipeline["config"])
    changed["taus"] = [0.5]
    with pytest.raises(ValueError, match="config changed"):
        run_final(signals, calibration, changed, **args)
