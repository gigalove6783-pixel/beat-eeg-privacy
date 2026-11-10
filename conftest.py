"""Attach `data` / `slow` markers at collection time.

The test files are covered by preregistration seals and cannot be edited, so markers are assigned
here by node id or test name. Select with e.g. `pytest -m "not data and not slow"`.
"""
import sys

import pytest

# Read the BNCI2014-001 raw files from data/; the dataset is not redistributable.
DATA_TESTS = {
    "tests/test_foundation.py::test_loaded_trial_counts_and_shape",
}

# beat.exp_l.memory_snapshot calls the Windows memory API (ctypes.windll). The module is sealed
# (seal 3), so its end-to-end test runs on Windows only; CI covers it on its Windows job.
WINDOWS_ONLY = {
    "tests/test_exp_l.py::test_synthetic_final_all_outputs_and_pickling",
}


def _is_slow(item):
    # Full synthetic dev+final round trips: test_synthetic_*final* in each experiment's tests.
    name = getattr(item, "originalname", item.name)
    return name.startswith("test_synthetic") and "final" in name


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.nodeid.split("[")[0] in DATA_TESTS:
            item.add_marker(pytest.mark.data)
        if _is_slow(item):
            item.add_marker(pytest.mark.slow)
        if item.nodeid.split("[")[0] in WINDOWS_ONLY and sys.platform != "win32":
            item.add_marker(pytest.mark.skip(
                reason="sealed beat.exp_l uses the Windows memory API; covered by the Windows CI job"))
