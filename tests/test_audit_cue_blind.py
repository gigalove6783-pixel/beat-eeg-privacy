"""Cue-blind attacker (scripts/audit_cue_blind.py; exploratory, post hoc) on synthetic stand-ins:
withholding `target` removes the cue from what the tool sees and changes nothing else, and the
summary rows pair a recorded cue-aware cell with its cue-blind rerun on the same support."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from beat_audit.api import AuditConfig, Condition, Segment, audit_stream
from beat_audit.io import load_input, make_windows

ROOT = Path(__file__).resolve().parents[1]
USERS, DAYS, SEGMENTS, TRIALS = 12, ("1", "2"), ("1", "2"), 20
CROSS = Condition("cross-day", "cross-day", (((Segment("1"),), (Segment("2"),)),))
SPEC = dict(id="synthetic", priority=1, cohort="synthetic", rung="features", condition="cross-day",
            k=5, sides="day 1 against day 2")


@pytest.fixture(scope="module")
def cue_blind():
    spec = importlib.util.spec_from_file_location("audit_cue_blind",
                                                  ROOT / "scripts" / "audit_cue_blind.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stand_in(kind="features"):
    """Users whose outputs differ mainly in their cue-specific means (identity in the cue)."""
    rng = np.random.default_rng(7)
    rows = []
    for u in range(USERS):
        offsets = rng.normal(0, 1.5, (2, 4))
        for day in DAYS:
            for segment in SEGMENTS:
                target = rng.integers(0, 2, TRIALS)
                rows += [(f"u{u:02d}", day, segment, t, offsets[t] + rng.normal(0, 1, 4))
                         for t in target]
    user, day, segment, target, x = map(np.array, zip(*rows))
    outputs = x if kind == "features" else (x[:, 0] > 0).astype(np.int64) + 1
    data = dict(outputs=outputs, user=user, day=day, segment=segment,
                order=np.arange(len(user), dtype=float), target=target.astype(str))
    if kind == "commands":
        data["C"] = np.array(3)
    return data


def write(path, data):
    np.savez(path, **data)
    return path


def test_withholding_removes_only_target_and_the_tool_sees_no_cue(cue_blind, tmp_path):
    for kind in ("features", "commands"):
        aware = write(tmp_path / f"{kind}.npz", stand_in(kind))
        blind = tmp_path / "blind" / f"{kind}.npz"
        sha = cue_blind.withhold_target(aware, blind)
        data = blind.read_bytes()
        assert cue_blind.withhold_target(aware, blind) == sha
        assert blind.read_bytes() == data == cue_blind.npz_bytes(
            {k: v for k, v in np.load(aware).items() if k != "target"})
        with np.load(aware) as a, np.load(blind, allow_pickle=False) as b:
            assert set(b.files) == set(a.files) - {"target"}
            for key in b.files:
                assert b[key].dtype == a[key].dtype and np.array_equal(b[key], a[key])
        seen, withheld = load_input(aware), load_input(blind)
        assert seen.target_given and not withheld.target_given
        assert set(withheld.target) == {"*"}
        for name in ("outputs", "user", "day", "segment", "order", "rows"):
            assert np.array_equal(getattr(seen, name), getattr(withheld, name))
        if kind == "features":   # rich summary: mean, two cue means, log variance -> mean, log var
            full = make_windows(seen, 5, ["0", "1"], summary="rich")
            mean_var = make_windows(withheld, 5, ["*"], summary="rich")
            plain = make_windows(withheld, 5, ["*"])
            for key, value in full.items():
                assert value.shape[1] == 16 and mean_var[key].shape[1] == 8
                assert np.array_equal(mean_var[key][:, :4], plain[key])
                assert np.array_equal(mean_var[key][:, :4], value[:, :4])
        else:                    # Bayes factor counts: two cue strata -> one stratum, same totals
            full = make_windows(seen, 5, ["0", "1"])
            pooled = make_windows(withheld, 5, ["*"])
            for key, value in full.items():
                assert pooled[key].shape[1] == 1
                assert np.array_equal(pooled[key][:, 0], value.sum(axis=1))
    bare = write(tmp_path / "bare.npz", {k: v for k, v in stand_in().items() if k != "target"})
    with pytest.raises(SystemExit, match="no target to withhold"):
        cue_blind.withhold_target(bare, tmp_path / "blind" / "bare.npz")


def reports(cue_blind, tmp_path):
    aware = write(tmp_path / "aware.npz", stand_in())
    blind = tmp_path / "blind.npz"
    cue_blind.withhold_target(aware, blind)
    settings = dict(fold_seed=3, seed=4, permutations=2000, bootstraps=300, threads=1,
                    cohort="synthetic", comparators=("cosine", "cosine_rich"), gallery=False,
                    sequential=False, closed_set=False)
    recorded = audit_stream(str(aware), conditions=[CROSS], kind="features", out=None,
                            config=AuditConfig(k_grid=(5, 10), cues=("0", "1"), **settings))
    rerun = audit_stream(str(blind), conditions=[CROSS], kind="features", out=None,
                         config=AuditConfig(**cue_blind.blind_config(
                             dict(k_grid=(5,), cues=("0", "1"), **settings))))
    return json.loads(json.dumps(recorded)), json.loads(json.dumps(rerun))


def test_summary_rows_pair_the_recorded_cell_with_its_cue_blind_rerun(cue_blind, tmp_path):
    recorded, rerun = reports(cue_blind, tmp_path)
    meta = dict(reproduction="exact", aware_report="aware/report.json", aware_report_sha256="a",
                blind_report="blind/report.slim.json")
    rows = cue_blind.summary_rows(SPEC, recorded, rerun, meta)
    assert [r["comparator"] for r in rows] == ["cosine", "cosine_rich"]
    aware_cell = cue_blind.cell_of(recorded, "cross-day", 5)
    blind_cell = cue_blind.cell_of(rerun, "cross-day", 5)
    for row in rows:
        a = aware_cell["comparators"][row["comparator"]]
        b = blind_cell["comparators"][row["comparator"]]
        assert set(row) == set(cue_blind.COLUMNS)
        assert (row["aware_auc"], row["blind_auc"]) == (a["auc"], b["auc"])
        assert row["diff_aware_minus_blind"] == a["auc"] - b["auc"]
        assert row["diff_ci_low"] <= row["diff_ci_high"]
        assert row["aware_holm_family"] == 2 and row["aware_holm_p"] == a["holm_p"]
        assert row["users_analysed"] == USERS
        assert row["population_hash"] == blind_cell["population_hash"]
        assert row["tier"] == "exploratory, post hoc"
    cosine, rich = rows
    # the mean summary never reads target: the cue-blind attacker is the cue-aware one exactly
    assert cosine["blind_auc"] == cosine["aware_auc"]
    assert (cosine["diff_aware_minus_blind"], cosine["diff_ci_low"], cosine["diff_ci_high"]) == (
        0, 0, 0)
    # the rich summary loses its cue means; on this stand-in identity sits in them
    assert rich["diff_aware_minus_blind"] > 0 and rich["diff_ci_low"] > 0
    path = tmp_path / "summary.csv"
    cue_blind.write_table(rows, path)
    data = path.read_bytes()
    assert b"\r" not in data and data.splitlines()[0].decode() == ",".join(cue_blind.COLUMNS)
    cue_blind.write_table(cue_blind.summary_rows(SPEC, recorded, rerun, meta), path)
    assert path.read_bytes() == data


@pytest.mark.parametrize("field", ["population_hash", "analysed_users", "resolved_pairs"])
def test_a_rerun_on_another_support_is_refused(cue_blind, tmp_path, field):
    recorded, rerun = reports(cue_blind, tmp_path)
    cell = cue_blind.cell_of(rerun, "cross-day", 5)
    cell[field] = {"population_hash": "0" * 64, "analysed_users": cell[field][1:],
                   "resolved_pairs": [pair[::-1] for pair in cell[field]]}[field]
    with pytest.raises(SystemExit, match=f"not the recorded support: \\['{field}'\\]"):
        cue_blind.summary_rows(SPEC, recorded, rerun, {})


def test_the_cue_blind_wording_is_placed_in_the_recorded_holm_family(cue_blind):
    assert cue_blind.wording_in_family(0.0001, 5) == "linkage detected"
    assert cue_blind.wording_in_family(0.2, 5) == "not detected: insufficient evidence, not absence"
    assert cue_blind.wording_in_family(0.03, 5).startswith("undetermined")
