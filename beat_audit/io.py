"""Strict NPZ/CSV schema and segment-local, non-overlapping windows.

CSV commands use ``outputs``; features use ``outputs_0,...,outputs_d``.
Optional ``C`` is a constant CSV column or a scalar NPZ member. NPZ object
arrays/pickle are forbidden. Unicode label arrays are portable across platforms.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


def digest(value):
    """SHA-256 of canonical JSON, independent of dict insertion order."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class Stream:
    outputs: np.ndarray
    user: np.ndarray
    day: np.ndarray
    segment: np.ndarray
    order: np.ndarray
    target: np.ndarray
    kind: str
    alphabet: int
    rows: np.ndarray
    path: str = ""
    target_given: bool = False

    @property
    def users(self):
        return sorted(set(self.user))

    @property
    def segments(self):
        return sorted(set(zip(self.day, self.segment)))


def _labels(value, field, n):
    a = np.asarray(value)
    if a.shape != (n,):
        raise ValueError(f"{field}: supply one nonmissing label per output, shape ({n},)")
    if pd.isna(a).any():
        raise ValueError(f"{field}: replace missing labels with real pseudonymous labels")
    a = a.astype(str)
    if np.any(np.char.str_len(np.char.strip(a)) == 0):
        raise ValueError(f"{field}: supply nonempty labels for every row")
    return a


def from_arrays(data, *, kind=None, path=""):
    """Validate schema and return a Stream; defaults are derived before row masks."""
    for name in ("outputs", "user", "day"):
        if name not in data:
            raise ValueError(f"{name}: add the required {name} array/CSV column")
    x = np.asarray(data["outputs"])
    kind = kind or ("commands" if x.ndim == 1 else "features")
    if kind not in ("commands", "features"):
        raise ValueError("kind: choose commands or features")
    if not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
        raise ValueError("outputs: supply finite numeric transmitted outputs")
    n = len(x)
    if kind == "commands":
        if x.ndim != 1 or np.any(x != np.floor(x)) or np.any(x < 0) or n == 0:
            raise ValueError("outputs: commands must be integers 0..C-1; remap stored codes")
        c_value = np.asarray(data.get("C", int(x.max()) + 1))
        if c_value.size != 1 or not np.isfinite(c_value).all():
            raise ValueError("C: declare one positive integer alphabet size")
        c = float(c_value.item())
        if c != int(c) or c < 1 or x.max() >= c:
            raise ValueError("C: set a positive integer larger than every command code")
        c = int(c)
        x = x.astype(np.int64)
    else:
        if x.ndim != 2 or x.shape[1] == 0 or n == 0:
            raise ValueError("outputs: features require a nonempty [N,d] numeric array")
        c = 0
        x = x.astype(float)
    u, d = (_labels(data[key], key, n) for key in ("user", "day"))
    if len(set(u)) < 6:
        raise ValueError("user: supply at least 6 distinct users for user-disjoint folds")
    if len(set(d)) < 2:
        raise ValueError("day: supply at least two recording days; one day cannot audit drift")
    order = np.asarray(data.get("order", np.arange(n)), dtype=float)
    if order.shape != (n,) or not np.isfinite(order).all():
        raise ValueError("order: supply one finite time position per row")
    segment = None if "segment" not in data else _labels(data["segment"], "segment", n)
    if segment is None:
        segment = np.full(n, "", dtype="<U1")
    for user in sorted(set(u)):
        for day in sorted(set(d[u == user])):
            ix = np.flatnonzero((u == user) & (d == day))
            if len(np.unique(order[ix])) != len(ix):
                raise ValueError("order: use unique time positions within each user and day")
            ix = ix[np.argsort(order[ix], kind="stable")]
            if "segment" not in data:
                segment[ix[:len(ix) // 2]] = "A"
                segment[ix[len(ix) // 2:]] = "B"
    target = _labels(data.get("target", np.repeat("*", n)), "target", n)
    return Stream(x, u, d, segment, order, target, kind, c, np.arange(n), str(path), "target" in data)


def load_input(path, *, kind=None):
    """Read a portable NPZ or CSV without pickle, executable input or attacker scores."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        try:
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
        except ValueError as exc:
            raise ValueError("input: use numeric/Unicode NPZ arrays, never pickled objects") from exc
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype={k: str for k in ("user", "day", "segment", "target")})
        data = {name: frame[name].to_numpy() for name in frame.columns}
        feature_names = [key for key in data if key.startswith("outputs_")]
        if feature_names:
            try:
                feature_names.sort(key=lambda name: int(name.split("_", 1)[1]))
            except ValueError as exc:
                raise ValueError("outputs: name feature columns outputs_0, outputs_1, ...") from exc
            data["outputs"] = np.column_stack([data.pop(key) for key in feature_names])
        if "C" in data:
            levels = np.unique(data["C"])
            if len(levels) != 1:
                raise ValueError("C: use the same declared alphabet size on every CSV row")
            data["C"] = levels[0]
    else:
        raise ValueError("input: supply a .npz or .csv file")
    unknown = set(data) - {"outputs", "user", "day", "segment", "order", "target", "C"}
    if unknown:
        raise ValueError(f"input fields {sorted(unknown)}: remove unknown fields/attacker scores")
    return from_arrays(data, kind=kind, path=path)


def masked(stream, mask):
    m = np.asarray(mask)
    if m.dtype != bool or m.shape != stream.user.shape:
        raise ValueError("population.masks: supply one boolean per original input row")
    return Stream(
        *(getattr(stream, name)[m] for name in
          ("outputs", "user", "day", "segment", "order", "target")),
        stream.kind, stream.alphabet, stream.rows[m], stream.path, stream.target_given,
    )


def make_windows(stream, k, cues, *, summary="mean"):
    """Return segment-local feature summaries or [window,cue,command] counts.

    Rich features concatenate the mean, cue means in ``cues`` order (only when
    target was supplied), and log population variance for k >= 5. Missing cues
    use the window mean; variance has the fixed, non-fitted floor 1e-12. At k=1
    the summary is exactly the mean. No quantity is fitted across windows/users.
    """
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k: use positive integer window lengths")
    if summary not in ("mean", "rich"):
        raise ValueError("summary: choose mean or rich")
    blocks = {}
    cue_index = {cue: i for i, cue in enumerate(cues)}
    for user in stream.users:
        for day, segment in stream.segments:
            ix = np.flatnonzero(
                (stream.user == user) & (stream.day == day) & (stream.segment == segment)
            )
            ix = ix[np.argsort(stream.order[ix], kind="stable")]
            nw = len(ix) // k
            ix = ix[:nw * k].reshape(nw, k)
            if stream.kind == "features":
                value = stream.outputs[ix].mean(axis=1)
                if summary == "rich" and k > 1:
                    parts = [value]
                    if stream.target_given:
                        for cue in cues:
                            present = stream.target[ix] == cue
                            count = present.sum(axis=1)[:, None]
                            total = (stream.outputs[ix] * present[:, :, None]).sum(axis=1)
                            parts.append(np.divide(total, count, out=value.copy(), where=count > 0))
                    if k >= 5:
                        parts.append(np.log(np.maximum(stream.outputs[ix].var(axis=1), 1e-12)))
                    value = np.concatenate(parts, axis=1)
            else:
                value = np.zeros((nw, len(cues), stream.alphabet), dtype=float)
                if nw:
                    q = np.array([cue_index[t] for t in stream.target[ix].ravel()])
                    np.add.at(value, (np.repeat(np.arange(nw), k), q,
                                     stream.outputs[ix].ravel()), 1)
            blocks[user, day, segment] = value
    return blocks
