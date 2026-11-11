"""scripts/audit_ladder.py on synthetic stand-ins (docs/plan_v3.md §4 "S4 protocol", §5 A10): the
per-user decoder's fitting boundary, the common trial support joined on trial keys for 2a-like
exports, and one population hash for every rung of a cohort's ladder (checked by the script and
by the tool's own `compare`) and one comparison digest (checked by the script)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
KEY = ["user", "source_session", "source_run", "source_trial"]


@pytest.fixture(scope="module")
def ladder():
    spec = importlib.util.spec_from_file_location("audit_ladder",
                                                  ROOT / "scripts" / "audit_ladder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decoder_data():
    """Three users: 40 calibration-day trials (day 1), 30 on each of days 2 and 3."""
    rng = np.random.default_rng(0)
    user, day, y = [], [], []
    for u in (1, 2, 3):
        for d, n in ((1, 40), (2, 30), (3, 30)):
            user += [u] * n
            day += [d] * n
            y += rng.permutation(np.tile([0, 1], n // 2)).tolist()
    user, day, y = map(np.array, (user, day, y))
    X = rng.normal(size=(len(y), 4)) + 0.8 * y[:, None] + 0.5 * user[:, None]
    return user, day, X, y


def test_no_later_day_trial_reaches_the_per_user_scaler_or_model(ladder, monkeypatch):
    user, day, X, y = decoder_data()
    later = day > 1
    seen = []
    fit = StandardScaler.fit

    def spy(self, data, *args, **kwargs):
        seen.append(np.asarray(data).copy())
        return fit(self, data, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", spy)
    margins, fits = ladder.per_user_margins(user, day, X, y, 1)
    monkeypatch.undo()
    # every scaler (one decoder and five cross-fitted folds per user) saw calibration rows only
    assert len(seen) == 3 * 6
    calibration, later_rows = {tuple(r) for r in X[~later]}, {tuple(r) for r in X[later]}
    assert all({tuple(r) for r in s} <= calibration and not {tuple(r) for r in s} & later_rows
               for s in seen)
    assert [f["calibration_trials"] for f in fits] == [40, 40, 40]
    # later-day margins are those of a decoder fitted on the user's calibration day alone
    for u in (1, 2, 3):
        cal, own = (user == u) & ~later, (user == u) & later
        scaler = StandardScaler().fit(X[cal])
        model = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(X[cal]), y[cal])
        np.testing.assert_allclose(margins[own], model.decision_function(scaler.transform(X[own])))
    # rewriting every later-day feature and label changes neither the decoders nor the
    # calibration-day (cross-fitted) margins; later-day margins move only with their features
    X2, y2 = X.copy(), y.copy()
    X2[later] = 50 + 100 * np.random.default_rng(1).normal(size=X2[later].shape)
    y2[later] = 1 - y2[later]
    margins2, fits2 = ladder.per_user_margins(user, day, X2, y2, 1)
    assert fits2 == fits
    np.testing.assert_array_equal(margins2[~later], margins[~later])
    y3 = y.copy()
    y3[later] = 1 - y3[later]
    np.testing.assert_array_equal(ladder.per_user_margins(user, day, X, y3, 1)[0], margins)


def test_commands_follow_the_studys_maximum_probability_rule(ladder):
    margins = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])  # max p = expit(|margin|): .95 .62 .5 .62 .95
    assert ladder.commands(margins, 0.5).tolist() == [1, 1, 1, 2, 2]  # every trial moves
    assert ladder.commands(margins, 0.6).tolist() == [1, 1, 0, 2, 2]
    assert ladder.commands(margins, 0.8).tolist() == [1, 0, 0, 0, 2]


def export(ladder, folder, name, frame, outputs, rng=None):
    """An export and its trial-key sidecar; order renumbered within user and day after filtering,
    rows optionally stored in another order."""
    frame = frame.sort_values(["user", "day", "source_run", "source_trial"]).reset_index(drop=True)
    frame = frame.assign(order=frame.groupby(["user", "day"]).cumcount())
    values = outputs(frame)
    if rng is not None:
        shuffled = rng.permutation(len(frame))
        frame, values = frame.iloc[shuffled].reset_index(drop=True), values[shuffled]
    arrays = dict(outputs=values, **{c: frame[c].to_numpy(np.int64)
                                     for c in ("user", "day", "segment", "order", "target")})
    path = Path(folder) / f"{name}.npz"
    ladder.save_npz(path, arrays)
    keys = frame.assign(row=np.arange(len(frame)))[["row", "user", "day", "order"] + KEY[1:]]
    keys.to_csv(path.with_suffix(".trials.csv"), index=False)
    return ladder.load_export(path, path.with_suffix(".trials.csv"))


def test_the_common_trial_support_joins_2a_like_exports_on_trial_keys(ladder, tmp_path):
    """B-like features drop artifact trials on both sessions; K-like commands drop them on the
    calibration session only and keep every deployment trial (2a: 113 such trials)."""
    rng = np.random.default_rng(1)
    trials = pd.DataFrame([(u, s, r, t) for u in (1, 2, 3) for s in ("T", "E")
                           for r in range(1, 7) for t in range(1, 9)], columns=KEY)
    trials = trials.assign(day=trials.source_session.map({"T": 1, "E": 2}),
                           segment=np.where(trials.source_run <= 3, 1, 2),
                           target=(trials.source_run + trials.source_trial) % 2)
    artifact = rng.random(len(trials)) < 0.2
    code = lambda f: (f.user * 10000 + f.source_run * 100 + f.source_trial).to_numpy()
    features, farr = export(ladder, tmp_path, "features", trials[~artifact],
                            lambda f: np.column_stack([code(f), -code(f)]).astype(float))
    kept = ~artifact | (trials.day == 2).to_numpy()
    commands, carr = export(ladder, tmp_path, "commands", trials[kept], lambda f: code(f) % 3, rng)
    extra = int((artifact & (trials.day == 2).to_numpy()).sum())
    assert extra > 0 and len(commands) == len(features) + extra
    # the stores disagree on (user, day, order): a positional join would pair different trials
    naive = features.merge(commands, on=["user", "day", "order"], suffixes=("", "_c"))
    assert (naive.source_trial != naive.source_trial_c).any()
    support = ladder.common_support({"features": features, "commands": commands}, [1, 2])
    assert len(support) == len(features)
    assert not support.merge(trials[artifact], on=KEY).shape[0]
    expected = code(support)
    np.testing.assert_array_equal(farr["outputs"][ladder.aligned_rows(features, support), 0],
                                  expected)
    np.testing.assert_array_equal(carr["outputs"][ladder.aligned_rows(commands, support)],
                                  expected % 3)
    assert (support.groupby(["user", "day"]).order.diff().dropna() == 1).all()
    # the written inputs share one layout and keep the key of every row
    layouts = []
    for name, frame, arrays, kind in (("f", features, farr, "features"),
                                      ("c", commands, carr, "commands")):
        path = ladder.write_input(tmp_path / "ladder", name, 0, support,
                                  arrays["outputs"][ladder.aligned_rows(frame, support)], kind)
        layouts.append(ladder.load_export(path, path.with_name(path.stem + ".trials.csv")))
    for column in ("user", "day", "segment", "order", "target"):
        np.testing.assert_array_equal(layouts[0][1][column], layouts[1][1][column])
    pd.testing.assert_frame_equal(layouts[0][0][KEY], layouts[1][0][KEY])
    np.testing.assert_array_equal(layouts[1][1]["outputs"], code(layouts[1][0]) % 3)
    # a shared trial whose segment or cue differs between stores, or another time order, stops it
    both = commands.merge(features[KEY], on=KEY, how="left", indicator=True)
    shared = both[both._merge == "both"].sort_values(["user", "day", "order"]).index
    for field, flip in (("segment", lambda v: 3 - v), ("target", lambda v: 1 - v)):
        bad = commands.copy()
        bad.loc[shared[0], field] = flip(bad.loc[shared[0], field])
        with pytest.raises(ValueError, match=f"{field} differs"):
            ladder.common_support({"features": features, "commands": bad}, [1, 2])
    swapped = commands.copy()
    swapped.loc[shared[:2], "order"] = swapped.loc[shared[1::-1], "order"].to_numpy()
    with pytest.raises(ValueError, match="time order differs"):
        ladder.common_support({"features": features, "commands": swapped}, [1, 2])


SETTINGS = dict(k_grid=[4, 8], fold_seed=1, seed=2, permutations=19, bootstraps=19, threads=1,
                comparators=["cosine"], gallery=False, sequential=False, closed_set=False)
CONDITIONS = [("same-day-d2", "same-day", [(["2/1"], ["2/2"])], "segments 1 and 2 of day 2")]


def synthetic_ladder(ladder, folder, drop=None, swap=None):
    """Nine users, a calibration day and a later day with two 17-trial segments each; four rungs
    written on one support (optionally one rung loses one later-day trial, or has the segment
    labels of the later day swapped)."""
    rng = np.random.default_rng(3)
    support = pd.DataFrame([(u, f"S{d}", s, t, d, s) for u in range(1, 10) for d in (1, 2)
                            for s in (1, 2) for t in range(17)], columns=KEY + ["day", "segment"])
    support = support.assign(target=support.source_trial % 2,
                             order=support.groupby(["user", "day"]).cumcount())
    user, day, target = (support[c].to_numpy() for c in ("user", "day", "target"))
    p0 = rng.normal(size=(len(support), 4)) + 0.7 * user[:, None] + target[:, None]
    p2 = p0[:, :2] @ np.array([[1.0, 0.5], [0.2, 1.0]])
    margins, _ = ladder.per_user_margins(user, day, p0, target, 1)
    outputs = {"features_P0": ("features", p0, None),
               "features_P2": ("features", p2, None),
               "commands_shared": ("commands", ladder.commands(p2[:, 0] - p2[:, 1], 0.6),
                                   "features_P2"),
               "commands_per_user": ("commands", ladder.commands(margins, 0.6), "features_P0")}
    plan = dict(cohort="synthetic", days=[1, 2], users=[str(u) for u in range(1, 10)],
                settings=SETTINGS, seconds_per_output=None, inputs={}, rungs=[],
                conditions=[dict(name=n, relation=r, pairs=p, sides=s) for n, r, p, s in CONDITIONS],
                support=dict(sha256_of_keys=ladder.layout_sha(support)))
    layouts = []
    for name, (kind, values, derived) in outputs.items():
        keep = np.ones(len(support), bool)
        if name == drop:
            keep[np.flatnonzero((user == 1) & (day == 2))[0]] = False
        written = support
        if name == swap:
            written = support.assign(segment=np.where(day == 2, 3 - support.segment, support.segment))
        plan["rungs"].append(dict(name=name, kind=kind, derived_from=derived))
        plan["inputs"][name] = {}
        for seed in ("0", "1"):
            path = ladder.write_input(folder, name, seed, written[keep], values[keep], kind)
            layouts.append(ladder.load_npz(path))
            plan["inputs"][name][seed] = dict(file=str(path), sha256=ladder.sha256_file(path))
    return plan, layouts


def test_every_rung_of_a_cohort_carries_one_population_hash(ladder, tmp_path):
    plan, layouts = synthetic_ladder(ladder, tmp_path / "inputs")
    eligible = ladder.eligibility(layouts, CONDITIONS, SETTINGS["k_grid"])
    assert eligible["same-day-d2"]["4"] == plan["users"]
    reports, paths = {}, {}
    for rung in plan["rungs"]:
        paths[rung["name"]] = tmp_path / rung["name"] / "report.json"
        reports[rung["name"]] = ladder.audit_rung(plan, rung, tmp_path / rung["name"])
    hashes = ladder.check_populations(reports, eligible)
    assert sorted(hashes) == [("same-day-d2", 4), ("same-day-d2", 8)]
    for (condition, k), value in hashes.items():
        assert {c["population_hash"] for r in reports.values() for c in r["cells"]
                if (c["condition"], c["k"]) == (condition, k)} == {value}
    # the tool's own compare accepts the matched reports and applies the data-processing guard
    utility = {name: dict(task_ba=0.7, hold_rate=0.1, correct_move_rate=0.6, wrong_move_rate=0.3)
               for name in reports}
    derived = {r["name"]: r["derived_from"] for r in plan["rungs"] if r["derived_from"]}
    rows, command = ladder.run_compare(paths, utility, derived, tmp_path / "compare.csv")
    requested = rows.condition == "same-day-d2"
    assert set(rows[requested].population_hash) == set(hashes.values())
    assert set(rows[~requested].condition) == {"cross-day"}  # the tool's coverage rows
    assert set(rows[~requested].status) == {"not_evaluable"}
    guards = ladder.guard_outcomes(rows, derived, ["same-day-d2"])
    assert len(guards) == 2 * len(SETTINGS["k_grid"])
    assert set(guards.outcome) <= {"flagged", "not flagged", "not evaluable"}
    digests = ladder.check_comparison(plan, reports)
    assert list(digests) == list(reports) and len(set(digests.values())) == 1
    # the reviewer's case: one rung with the later day's segment labels swapped keeps the tool's
    # population hash (users, folds, row indices) but is refused through the comparison digest
    swapped, _ = synthetic_ladder(ladder, tmp_path / "swapped", swap="commands_shared")
    rung = next(r for r in swapped["rungs"] if r["name"] == "commands_shared")
    matched = reports["commands_shared"]
    reports["commands_shared"] = ladder.audit_rung(swapped, rung, tmp_path / "swapped_shared")
    assert ladder.check_populations(reports, eligible) == hashes
    plan_swapped = dict(plan, inputs=dict(plan["inputs"],
                                          commands_shared=swapped["inputs"]["commands_shared"]))
    with pytest.raises(ValueError, match="commands_shared: comparison digest differs"):
        ladder.check_comparison(plan_swapped, reports)
    reports["commands_shared"] = matched
    # one rung on a support without one later-day trial: refused by the script and by compare
    other, _ = synthetic_ladder(ladder, tmp_path / "other", drop="commands_shared")
    rung = next(r for r in other["rungs"] if r["name"] == "commands_shared")
    paths["commands_shared"] = tmp_path / "other_shared" / "report.json"
    reports["commands_shared"] = ladder.audit_rung(other, rung, tmp_path / "other_shared")
    with pytest.raises(ValueError, match="population hash differs"):
        ladder.check_populations(reports)
    with pytest.raises(SystemExit, match="compare refused.*population_hash"):
        ladder.run_compare(paths, utility, derived, tmp_path / "refused.csv")
