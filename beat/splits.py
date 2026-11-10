"""Run-level evaluation splits."""
from itertools import combinations

from .data import Trials, require_training

N_RUNS = 6

# Post-deployment attack scenario: train on one half of the E runs, test on the other, both directions.
E_HALVES = (((1, 2, 3), (4, 5, 6)), ((4, 5, 6), (1, 2, 3)))


def t_run_folds(trials: Trials):
    """Leave-one-run-out folds over session T: yields (held_out_run, train, validation)."""
    require_training(trials)
    for run in range(1, N_RUNS + 1):
        held = trials.run == run
        yield run, trials.subset(~held), trials.subset(held)


def calibration_sets(n_runs: int) -> list[tuple[int, ...]]:
    """Every combination of n_runs T runs used as calibration data."""
    return list(combinations(range(1, N_RUNS + 1), n_runs))
