"""A1 input errors, windows, explicit populations and import isolation."""

import subprocess
import sys

import numpy as np
import pytest

from beat_audit import AuditConfig, Population, audit_stream
from beat_audit.cli import main
from beat_audit.io import from_arrays, make_windows
from beat_audit.synthetic import stream_arrays


def test_import_does_not_load_sealed_experiments_or_torch(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c",
         "import beat_audit,sys; assert 'torch' not in sys.modules; "
         "assert not any(n.startswith('beat.exp_') for n in sys.modules)"],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("field,change,fix", [
    ("day", lambda a: a.pop("day"), "add"),
    ("day", lambda a: a.update(day=np.repeat("1", len(a["day"]))), "two"),
    ("user", lambda a: a.update(user=np.repeat("one", len(a["user"]))), "6"),
    ("outputs", lambda a: a.update(outputs=a["outputs"] + 0.5), "integers"),
])
def test_a1_invalid_input_names_field_and_fix(tmp_path, capsys, field, change, fix):
    arrays = stream_arrays(kind="commands")
    change(arrays)
    path = tmp_path / "invalid.npz"
    np.savez(path, **arrays)
    assert main(["run", str(path), "--out", str(tmp_path / "out"), "--k", "1"]) == 2
    error = capsys.readouterr().err
    assert field in error and fix in error


def test_no_window_error_and_over_ten_percent_exclusions(tmp_path, capsys):
    arrays = stream_arrays(kind="commands", users=9, per_segment=5)
    path = tmp_path / "stream.npz"
    np.savez(path, **arrays)
    assert main(["run", str(path), "--out", str(tmp_path / "none"), "--k", "100"]) == 2
    assert "k=100" in capsys.readouterr().err
    mask = ~((arrays["user"] == "u000") & (arrays["day"] == "1") & (arrays["segment"] == "1"))
    config = AuditConfig(k_grid=(5,), bootstraps=19, permutations=19,
                         gallery=False, closed_set=False, sequential=False)
    report = audit_stream(path, config=config, population=Population(
        tuple(sorted(set(arrays["user"]))), masks={"0": mask}))
    assert all(c["status"] == "not_evaluable" for c in report["cells"])
    assert all(c["excluded_users"] == ["u000"] for c in report["cells"])


def test_single_missing_user_is_listed_below_exclusion_limit(tmp_path):
    arrays = stream_arrays(kind="commands", users=12, per_segment=5)
    path = tmp_path / "stream.npz"
    np.savez(path, **arrays)
    mask = arrays["user"] != "u000"
    config = AuditConfig(k_grid=(5,), bootstraps=19, permutations=19,
                         gallery=False, closed_set=False, sequential=False)
    report = audit_stream(path, config=config, population=Population(
        tuple(sorted(set(arrays["user"]))), masks={"0": mask}))
    assert all(c["status"] == "ok" for c in report["cells"])
    assert all(c["excluded_users"] == ["u000"] for c in report["cells"])


def test_windows_follow_order_and_never_cross_segments():
    arrays = stream_arrays(kind="commands", users=6, per_segment=5)
    arrays["outputs"] = np.arange(len(arrays["outputs"])) % 3
    data = from_arrays(arrays)
    windows = make_windows(data, 3, ["*"])
    assert all(value.shape == (1, 1, 3) for value in windows.values())
    assert all(value.sum() == 3 for value in windows.values())
    assert np.array_equal(windows["u000", "1", "1"][0, 0], [1, 1, 1])
    arrays.pop("segment")
    reversed_order = np.arange(len(arrays["outputs"]))[::-1]
    arrays["order"] = reversed_order
    data = from_arrays(arrays)
    selected = (data.user == "u000") & (data.day == "1")
    assert np.array_equal(data.segment[selected], ["B"] * 5 + ["A"] * 5)


def test_rejects_pickle_and_attacker_scores(tmp_path):
    path = tmp_path / "unsafe.npz"
    np.savez(path, outputs=np.array([{"x": 1}], dtype=object))
    with pytest.raises(ValueError, match="pickle"):
        from beat_audit.io import load_input
        load_input(path)
    arrays = stream_arrays(kind="commands")
    arrays["attacker_scores"] = np.zeros(len(arrays["outputs"]))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="attacker scores"):
        load_input(path)


def test_positive_alphabet_and_absent_cues():
    arrays = stream_arrays(kind="commands", users=6, per_segment=3)
    arrays["outputs"][:] = 0
    arrays["C"] = np.array(4)
    data = from_arrays(arrays)
    windows = make_windows(data, 3, ["*", "absent"])
    first = next(iter(windows.values()))
    assert first.shape == (1, 2, 4)
    assert first[0, 1].sum() == 0
    assert first[0, 0, 0] == 3
