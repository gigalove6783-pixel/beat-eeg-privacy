"""Download Lee2019 (OpenBMI) motor-imagery files one at a time and keep only analysis epochs.

Each ~609 MB session file is reduced to the 20 motor channels used for MI in the
original paper, causally band-passed 8-30 Hz at 1000 Hz, decimated to 250 Hz and cut
to 1-4 s after the cue, matching the BNCI2014-001 pipeline (`configs/data.yaml`). The
raw file is deleted afterwards because the full set (~66 GB) exceeds the free disk
space; SHA-256 of every raw file is recorded so the input can be re-verified.
"""
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import scipy.io as sio

from beat.config import ROOT, load_config
from beat.data import bandpass_sos, causal_bandpass

BASE_URL = ("https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/"
            "10.5524/100001_101000/100542/")
MOTOR = ["FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
         "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"]
RAW = ROOT / "data" / "raw" / "lee2019"
OUT = ROOT / "data" / "processed" / "lee2019"
FS_IN, DECIMATE = 1000, 4
LABELS = {2: 0, 1: 1}  # y_dec: 1 = right, 2 = left -> model label left 0, right 1


def reduce_phase(run, sos, cfg):
    names = [str(c) for c in np.atleast_1d(run.chan)]
    raw = np.asarray(run.x, dtype=np.float64)[:, [names.index(c) for c in MOTOR]]
    finite_raw = np.isfinite(raw)
    bad_sample = ~finite_raw.all(axis=1)
    filtered = causal_bandpass(sos, np.where(finite_raw, raw, 0.0))
    start = int(round(cfg["window_after_cue_s"][0] * FS_IN))
    stop = int(round(cfg["window_after_cue_s"][1] * FS_IN))
    guard = int(round(cfg["nonfinite_guard_s"] * FS_IN))
    X, finite = [], []
    for onset in np.atleast_1d(run.t).astype(int) - 1:  # MATLAB indices are 1-based
        a, b = onset + start, onset + stop
        X.append(filtered[a:b:DECIMATE].T)
        finite.append(not bad_sample[max(a - guard, 0):b].any())
    labels = np.array([LABELS[int(v)] for v in np.atleast_1d(run.y_dec)])
    return np.stack(X).astype(np.float32), labels, np.array(finite)


def fetch(url: str, target: Path) -> None:
    partial = target.with_suffix(".part")
    for attempt in range(2):  # one retry for a transient network failure
        try:
            _, headers = urllib.request.urlretrieve(url, partial)
            if partial.stat().st_size == int(headers["Content-Length"]):
                partial.replace(target)
                return
        except OSError:
            if attempt:
                raise
    raise OSError(f"incomplete download: {url}")


def main() -> None:
    cfg = load_config("data")
    sos = bandpass_sos(dict(cfg, sfreq=FS_IN))
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for subject in range(1, 55):
        for session in (1, 2):
            name = f"sess{session:02d}_subj{subject:02d}_EEG_MI.mat"
            out = OUT / f"sess{session}_subj{subject:02d}.npz"
            if out.exists():
                continue
            raw_path = RAW / name
            if not raw_path.exists():
                fetch(f"{BASE_URL}session{session}/s{subject}/{name}", raw_path)
            digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            mat = sio.loadmat(raw_path, squeeze_me=True, struct_as_record=False)
            parts = [reduce_phase(mat[f"EEG_MI_{phase}"], sos, cfg) for phase in ("train", "test")]
            np.savez(out,
                     X=np.concatenate([p[0] for p in parts]),
                     label=np.concatenate([p[1] for p in parts]),
                     finite=np.concatenate([p[2] for p in parts]),
                     phase=np.repeat(np.array(["train", "test"]), [len(p[1]) for p in parts]),
                     trial=np.concatenate([np.arange(1, len(p[1]) + 1) for p in parts]))
            with open(OUT / "SHA256SUMS", "a", encoding="utf-8") as f:
                f.write(f"{digest}  {name}\n")
            raw_path.unlink()
            print(f"{name} -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
