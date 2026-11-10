"""Loading, causal filtering and epoching of BNCI2014-001 left/right motor-imagery trials."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, sosfilt, sosfilt_zi

from .config import ROOT, load_config

META_COLUMNS = ["subject", "session", "run", "trial", "label", "artifact", "finite"]


class LeakageError(RuntimeError):
    """Raised when a fit step receives session-E trials."""


@dataclass(frozen=True)
class Trials:
    """Per-trial array plus metadata; X may hold signals, covariances or features.

    meta columns: subject (1-9), session ("T"/"E"), run (1-6 within session),
    trial (1-48, chronological position within the run, counting all four MI
    classes), label (0 left, 1 right), artifact (official flag), finite.
    """

    X: np.ndarray
    meta: pd.DataFrame

    def __post_init__(self):
        if len(self.X) != len(self.meta):
            raise ValueError("X and meta lengths differ")

    def __len__(self) -> int:
        return len(self.meta)

    def subset(self, mask) -> Trials:
        mask = np.asarray(mask)
        idx = np.flatnonzero(mask) if mask.dtype == bool else mask
        return Trials(self.X[idx], self.meta.iloc[idx].reset_index(drop=True))

    def with_X(self, X) -> Trials:
        return Trials(np.asarray(X), self.meta)

    @property
    def y(self) -> np.ndarray:
        return self.meta["label"].to_numpy()

    @property
    def subject(self) -> np.ndarray:
        return self.meta["subject"].to_numpy()

    @property
    def session(self) -> np.ndarray:
        return self.meta["session"].to_numpy()

    @property
    def run(self) -> np.ndarray:
        return self.meta["run"].to_numpy()


def require_training(trials: Trials) -> None:
    """Every fit entry point calls this so that session E can never shape a model."""
    sessions = set(trials.meta["session"].unique())
    if sessions != {"T"}:
        raise LeakageError(f"fit step received sessions {sorted(sessions)}; only T is allowed")


def quality_mask(trials: Trials) -> np.ndarray:
    """Trials kept in experiments A and B: no official artifact flag and finite samples."""
    return (~trials.meta["artifact"].to_numpy()) & trials.meta["finite"].to_numpy()


def causal_bandpass(sos: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Forward-only filtering of a continuous recording (samples x channels)."""
    zi = sosfilt_zi(sos)[:, :, None] * signal[0][None, None, :]
    out, _ = sosfilt(sos, signal, axis=0, zi=zi)
    return out


def bandpass_sos(cfg: dict) -> np.ndarray:
    return butter(cfg["filter_order"], cfg["bandpass_hz"], btype="bandpass", fs=cfg["sfreq"], output="sos")


def _mi_runs(path):
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    # The first runs of each session are EOG/rest recordings without trials.
    return [run for run in np.atleast_1d(mat["data"]) if np.atleast_1d(run.trial).size]


def _epoch_session(path, subject: int, session: str, cfg: dict, sos: np.ndarray):
    fs = cfg["sfreq"]
    cue = cfg["cue_after_trial_start_s"]
    start_off = int(round((cue + cfg["window_after_cue_s"][0]) * fs))
    stop_off = int(round((cue + cfg["window_after_cue_s"][1]) * fs))
    guard = int(round(cfg["nonfinite_guard_s"] * fs))
    label_map = {int(k): v for k, v in cfg["labels"].items()}
    signals, rows = [], []
    for run_no, run in enumerate(_mi_runs(path), start=1):
        raw = np.asarray(run.X, dtype=np.float64)[:, : cfg["n_eeg_channels"]]
        finite_raw = np.isfinite(raw)
        bad_sample = ~finite_raw.all(axis=1)
        filtered = causal_bandpass(sos, np.where(finite_raw, raw, 0.0))
        onsets = np.atleast_1d(run.trial).astype(int) - 1  # MATLAB indices are 1-based
        codes = np.atleast_1d(run.y).astype(int)
        artifacts = np.atleast_1d(run.artifacts).astype(bool)
        for trial_no, (onset, code, artifact) in enumerate(zip(onsets, codes, artifacts), start=1):
            if code not in label_map:
                continue
            a, b = onset + start_off, onset + stop_off
            signals.append(filtered[a:b].T)
            rows.append(
                dict(
                    subject=subject,
                    session=session,
                    run=run_no,
                    trial=trial_no,
                    label=label_map[code],
                    artifact=bool(artifact),
                    finite=not bad_sample[max(a - guard, 0) : b].any(),
                )
            )
    return signals, rows


def load_trials(cfg: dict | None = None) -> Trials:
    """All left/right trials of every subject and session, filtered and epoched; cached on disk."""
    cfg = cfg or load_config("data")
    key = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    cache = ROOT / cfg["processed_dir"] / f"trials_{key}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return Trials(z["X"], pd.DataFrame({c: z[c] for c in META_COLUMNS}))
    sos = bandpass_sos(cfg)
    signals, rows = [], []
    for subject in cfg["subjects"]:
        for session in cfg["sessions"]:
            path = ROOT / cfg["raw_dir"] / f"A{subject:02d}{session}.mat"
            s, r = _epoch_session(path, subject, session, cfg, sos)
            signals += s
            rows += r
    X = np.stack(signals).astype(np.float32)
    meta = pd.DataFrame(rows, columns=META_COLUMNS)
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Fixed-width strings keep the cache loadable without pickle.
    np.savez(cache, X=X, **{c: meta[c].to_numpy().astype("U1") if c == "session" else meta[c].to_numpy() for c in META_COLUMNS})
    return Trials(X, meta)
