"""Attach `data` / `slow` markers at collection time.

The test files are covered by preregistration seals and cannot be edited, so markers are assigned
here by node id or test name. Select with e.g. `pytest -m "not data and not slow"`.
"""
import pytest

# Read the BNCI2014-001 raw files from data/; the dataset is not redistributable.
DATA_TESTS = {
    "tests/test_foundation.py::test_loaded_trial_counts_and_shape",
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
