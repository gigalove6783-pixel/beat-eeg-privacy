"""Lee2019 plan-v3 blocks; session 2 requires explicit post-seal authorization."""
from pathlib import Path

import numpy as np
import pandas as pd

from .data import LeakageError, Trials


def load_trials(directory, subjects=range(1, 55), session=1, *, allow_session2=False):
    if session not in (1, 2):
        raise ValueError("Session must be 1 or 2")
    if session == 2 and not allow_session2:
        raise LeakageError("Session 2 is sealed")
    arrays, frames = [], []
    for subject in subjects:
        if not 1 <= subject <= 54:
            raise ValueError("Lee2019 subject must be 1..54")
        with np.load(Path(directory) / f"sess{session}_subj{subject:02d}.npz",
                     allow_pickle=False) as data:
            X = data["X"]
            phase, trial = data["phase"].astype(str), data["trial"]
            if X.shape != (200, 20, 750):
                raise ValueError("Expected 200 x 20 x 750 reduced Lee2019 signals")
            if not np.isin(phase, ["train", "test"]).all():
                raise ValueError("Unknown phase")
            for name in ("train", "test"):
                if not np.array_equal(trial[phase == name], np.arange(1, 101)):
                    raise ValueError("Phase trials must preserve recorded order 1..100")
            if not np.isin(data["label"], [0, 1]).all():
                raise ValueError("Expected binary MI labels")
            finite = data["finite"].astype(bool)
            if not np.isfinite(X[finite]).all():
                raise ValueError("Rows marked retained contain non-finite signals")
            block = np.where(phase == "train", 1, 3) + (trial > 50)
            arrays.append(X)
            frames.append(pd.DataFrame(dict(
                subject=subject, session="T" if session == 1 else "E",
                run=block, trial=trial, label=data["label"],
                artifact=False, finite=finite, phase=phase)))
    return Trials(np.concatenate(arrays), pd.concat(frames, ignore_index=True))
