"""Yang2025 (WBCIC-SHU two-class MI): range reader, BDF reader, reduction and day guard.

One source serves every recording day: figshare article 22671172, a single ZIP64 archive.
Before seal 4 only the explicit day-1 allowlist plus top-level metadata may be read;
recording days 2-3 require a verified seal 4 (tag + RFC 3161 token).
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
import subprocess
import time
import urllib.error
import urllib.request
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT
from .data import LeakageError, Trials, bandpass_sos, causal_bandpass

ARTICLE = 22671172
API = f"https://api.figshare.com/v2/articles/{ARTICLE}"
TOP = "WBCIC_SHU Motor Imagery dataset/"
SOURCE = TOP + "sourcedata/2C dataset/"
SUBJECTS = tuple(range(1, 52))
DAYS = (1, 2, 3)
KINDS = ("data", "evt")
# L's 20 motor channels minus CPz, which Yang2025's montage does not have (190-d tangent vectors).
MOTOR = ("FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
         "CP5", "CP3", "CP1", "CP2", "CP4", "CP6")
METADATA = frozenset(TOP + name for name in (
    "CHANGES", "dataset_description.json", "LICENSE", "participants.json", "participants_2C.tsv",
    "participants_3C.tsv", "README.md", "task-motorimagery_channels.tsv",
    "task-motorimagery_coordsystem.json", "task-motorimagery_eeg.json",
    "task-motorimagery_electrodes.tsv", "task-motorimagery_events.json"))
CHUNK = 8 << 20


def member_path(subject: int, day: int, kind: str) -> str:
    if subject not in SUBJECTS or day not in DAYS or kind not in KINDS:
        raise ValueError(f"No Yang2025 member for subject {subject}, day {day}, {kind}")
    return f"{SOURCE}sub-{subject:03d}/ses-{day:02d}/eeg/{kind}.bdf"


DAY1 = frozenset(member_path(s, 1, k) for s in SUBJECTS for k in KINDS)
DAY23 = frozenset(member_path(s, d, k) for s in SUBJECTS for d in (2, 3) for k in KINDS)


def allowlist() -> frozenset:
    """Pre-seal allowlist: top-level metadata and the day-1 raw members, nothing else."""
    return METADATA | DAY1


OPENSSL = Path(r"C:\Program Files\Git\usr\bin\openssl.exe")
STAMP_DIR = Path("results/exp_m/seal4_timestamp")
# Pre-registered names of the stored RFC 3161 response, TSA CA certificate and TSA certificate.
STAMP_FILES = ("seal4.tsr", "cacert.pem", "tsa.crt")


def seal_imprint(commit: str) -> str:
    """Message imprint of the seal-4 timestamp request: SHA-256 of the 40-character seal
    commit id in ASCII without a newline (only this hash is sent to the TSA)."""
    return hashlib.sha256(commit.encode("ascii")).hexdigest()


def verify_timestamp(commit: str, stamp_dir, openssl=OPENSSL) -> dict:
    """`openssl ts -verify` of the stored response against the imprint of `commit`; exit 0 and
    "Verification: OK" are required. Returns the verification record with the token time."""
    stamp_dir = Path(stamp_dir)
    response, ca, tsa = (stamp_dir / name for name in STAMP_FILES)
    for path in (response, ca, tsa):
        if not path.is_file():
            raise LeakageError(f"Missing seal-4 timestamp file {path.name}")
    command = [str(openssl), "ts", "-verify", "-digest", seal_imprint(commit), "-in",
               str(response), "-CAfile", str(ca), "-untrusted", str(tsa)]
    result = subprocess.run(command, capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0 or "Verification: OK" not in output:
        raise LeakageError(f"The RFC 3161 token does not verify for commit {commit}: "
                           f"{output[-500:]}")
    reply = subprocess.run([str(openssl), "ts", "-reply", "-in", str(response), "-text"],
                           capture_output=True, text=True, check=True).stdout
    fields = dict(re.findall(r"^(Hash Algorithm|Time stamp): (.+?)\s*$", reply, re.M))
    if fields.get("Hash Algorithm") != "sha256":
        raise LeakageError("The RFC 3161 token does not carry a SHA-256 imprint")
    version = subprocess.run([str(openssl), "version"], capture_output=True, text=True,
                             check=True).stdout.strip()
    return dict(commit=commit, imprint_sha256=seal_imprint(commit),
                token_time=fields["Time stamp"], command=command, exit_code=result.returncode,
                output=output, openssl_version=version, verified_utc=utc_now())


def post_seal_allowlist(seal_commit: str, root=ROOT, openssl=OPENSSL) -> frozenset:
    """Day 1-3 allowlist, only once annotated tag seal-4 names `seal_commit` and the stored
    RFC 3161 response verifies against the SHA-256 imprint of that commit id. Each successful
    verification (command, output, token time) is appended to guard_verification.jsonl."""
    root = Path(root)

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root,
                                       stderr=subprocess.DEVNULL).decode().strip()
    try:
        if git("cat-file", "-t", "seal-4") != "tag":
            raise LeakageError("seal-4 must be an annotated tag")
        oid = git("rev-parse", "--verify", f"{seal_commit}^{{commit}}")
        if git("rev-parse", "--verify", "seal-4^{commit}") != oid:
            raise LeakageError("seal-4 does not name the supplied commit")
    except subprocess.CalledProcessError as exc:
        raise LeakageError("Unable to verify tag seal-4") from exc
    record = verify_timestamp(oid, root / STAMP_DIR, openssl)
    with open(root / STAMP_DIR / "guard_verification.jsonl", "a", encoding="utf-8",
              newline="\n") as f:
        f.write(json.dumps(record) + "\n")
    return METADATA | DAY1 | DAY23


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ------------------------------------------------------------------ range reader

class RangeZip:
    """Exact byte-range access to members of the remote ZIP64 archive.

    Every HTTP request is a bounded byte range (the archive tail and central directory, or a
    single member's local header and data) and is logged with UTC time, member, range,
    figshare version and ETag. Requests after the first carry If-Match on the first ETag.
    """

    def __init__(self, log_path, allowed=None, opener=None, resolve=None):
        self.log_path = Path(log_path)
        self.allowed = allowlist() if allowed is None else frozenset(allowed)
        self.opener = opener or urllib.request.urlopen
        self.resolve = resolve or resolve_source
        self.source = None
        self.etag = None
        self.entries = None

    # -- HTTP
    def _request(self, member, start, end):
        """Bytes [start, end] (inclusive) of the archive as a response stream."""
        if self.source is None:
            self.source = self.resolve(self.opener)
        headers = {"Range": f"bytes={start}-{end}"}
        if self.etag:
            headers["If-Match"] = self.etag
        try:
            response = self.opener(urllib.request.Request(self.source["url"], headers=headers),
                                   timeout=120)
        except urllib.error.HTTPError as exc:
            self._log(member, start, end, None, exc.code, None)
            if exc.code != 403:
                raise
            # Signed storage URLs expire; one re-resolution, then the request is repeated.
            self.source = self.resolve(self.opener)
            response = self.opener(urllib.request.Request(self.source["url"], headers=headers),
                                   timeout=120)
        etag = response.headers.get("ETag")
        self.etag = self.etag or etag
        expected = f"bytes {start}-{end}/{self.source['size']}"
        content_range = response.headers.get("Content-Range")
        self._log(member, start, end, etag, getattr(response, "status", None), content_range)
        if content_range != expected or etag != self.etag:
            response.close()
            raise OSError(f"Unexpected range response {content_range!r} / ETag {etag!r}")
        return response

    def _log(self, member, start, end, etag, status, content_range):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(dict(utc=utc_now(), member=member, start=start, end=end,
                                    figshare_version=self.source["version"],
                                    file_id=self.source["file_id"], etag=etag,
                                    status=status, content_range=content_range)) + "\n")

    def _bytes(self, member, start, end):
        with self._request(member, start, end) as response:
            data = response.read()
        if len(data) != end - start + 1:
            raise OSError(f"Short range read for {member}")
        return data

    # -- directory
    def directory(self):
        if self.entries is not None:
            return self.entries
        if self.source is None:
            self.source = self.resolve(self.opener)
        size = self.source["size"]
        tail_start = max(0, size - 65558)
        tail = self._bytes("<archive end records>", tail_start, size - 1)
        eocd = tail.rfind(b"PK\x05\x06")
        if eocd < 0:
            raise ValueError("No end-of-central-directory record")
        cd_size, cd_offset = struct.unpack_from("<II", tail, eocd + 12)
        if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
            locator = eocd - 20
            if tail[locator:locator + 4] != b"PK\x06\x07":
                raise ValueError("Missing ZIP64 locator")
            (z64,) = struct.unpack_from("<Q", tail, locator + 8)
            record = tail[z64 - tail_start:z64 - tail_start + 56]
            if record[:4] != b"PK\x06\x06":
                raise ValueError("ZIP64 end record not in the tail range")
            cd_size, cd_offset = struct.unpack_from("<QQ", record, 40)
        cd = self._bytes("<central directory>", cd_offset, cd_offset + cd_size - 1)
        self.entries = parse_central_directory(cd)
        return self.entries

    # -- members
    def check_allowed(self, name):
        if name not in self.allowed:
            raise LeakageError(f"Member outside the allowlist: {name}")

    def stream_member(self, name, out_path):
        """Inflate one allowlisted member to out_path; returns (sha256, size), CRC checked."""
        self.check_allowed(name)
        entry = self.directory()[name]
        head = self._bytes(name, entry["offset"], entry["offset"] + 29)
        if head[:4] != b"PK\x03\x04":
            raise ValueError(f"Bad local header for {name}")
        nlen, xlen = struct.unpack_from("<HH", head, 26)
        data_start = entry["offset"] + 30 + nlen + xlen
        first = entry["offset"] + 30
        end = data_start + entry["csize"] - 1
        if entry["method"] not in (0, 8):
            raise ValueError(f"Unsupported compression method {entry['method']} for {name}")
        inflater = zlib.decompressobj(-15) if entry["method"] == 8 else None
        sha, crc, written = hashlib.sha256(), 0, 0
        position, skip, retries = first, nlen + xlen, 0
        out_path = Path(out_path)
        with open(out_path, "wb") as out:
            while position <= end:
                try:
                    with self._request(name, position, end) as response:
                        while True:
                            block = response.read(CHUNK)
                            if not block:
                                break
                            position += len(block)
                            if skip:
                                taken = block[:skip]
                                if position - len(block) == first and taken[:nlen] != name.encode():
                                    raise ValueError(f"Local name differs for {name}")
                                block, skip = block[skip:], max(0, skip - len(taken))
                            plain = inflater.decompress(block) if inflater else block
                            sha.update(plain)
                            crc = zlib.crc32(plain, crc)
                            written += len(plain)
                            out.write(plain)
                except (OSError, urllib.error.URLError) as exc:
                    retries += 1
                    if retries > 3:
                        raise OSError(f"Range transfer failed for {name}") from exc
                    time.sleep(2)
            if inflater:
                plain = inflater.flush()
                sha.update(plain)
                crc = zlib.crc32(plain, crc)
                written += len(plain)
                out.write(plain)
        if written != entry["usize"] or crc != entry["crc"]:
            raise ValueError(f"Size/CRC mismatch for {name}")
        return sha.hexdigest(), written


def parse_central_directory(cd: bytes) -> dict:
    entries, pos = {}, 0
    while pos + 46 <= len(cd) and cd[pos:pos + 4] == b"PK\x01\x02":
        (flags, method, crc, csize, usize, nlen, xlen, clen,
         offset) = struct.unpack_from("<8xHH4xIIIHHH8xI", cd, pos)
        name = cd[pos + 46:pos + 46 + nlen].decode("utf-8" if flags & 0x800 else "cp437")
        extra = cd[pos + 46 + nlen:pos + 46 + nlen + xlen]
        values = [usize, csize, offset]
        x = 0
        while x + 4 <= len(extra):
            tag, size = struct.unpack_from("<HH", extra, x)
            if tag == 1:
                field = x + 4
                for i, value in enumerate(values):
                    if value == 0xFFFFFFFF:
                        (values[i],) = struct.unpack_from("<Q", extra, field)
                        field += 8
            x += 4 + size
        entries[name] = dict(method=method, crc=crc, usize=values[0], csize=values[1],
                             offset=values[2])
        pos += 46 + nlen + xlen + clen
    return entries


# ------------------------------------------------------------------ BDF reader (pure numpy)

def read_bdf_header(raw: bytes) -> dict:
    """Fixed BDF/EDF header fields and per-signal fields."""
    def text(a, b):
        return raw[a:b].decode("latin-1").strip()
    ns = int(text(252, 256))
    head = dict(start_date=text(168, 176), start_time=text(176, 184),
                header_bytes=int(text(184, 192)), reserved=text(192, 236),
                n_records=int(text(236, 244)), record_seconds=float(text(244, 252)), ns=ns)
    pos = 256
    for field, width in (("label", 16), ("transducer", 80), ("unit", 8), ("pmin", 8),
                         ("pmax", 8), ("dmin", 8), ("dmax", 8), ("prefilter", 80),
                         ("samples", 8), ("signal_reserved", 32)):
        values = [raw[pos + i * width:pos + (i + 1) * width].decode("latin-1").strip()
                  for i in range(ns)]
        if field in ("pmin", "pmax", "dmin", "dmax"):
            values = [float(v) for v in values]
        elif field == "samples":
            values = [int(v) for v in values]
        head[field] = values
        pos += width * ns
    if pos != head["header_bytes"]:
        raise ValueError("BDF header size mismatch")
    return head


def read_bdf(path, channels=None):
    """(header, {label: physical float64 samples}) of a 24-bit BDF file.

    `channels` restricts decoding to the named signals; annotation signals are returned raw
    as uint8 bytes (key = their label)."""
    raw = Path(path).read_bytes()
    head = read_bdf_header(raw[:256 + 256 * int(raw[252:256].decode("latin-1"))])
    spr = np.asarray(head["samples"])
    record_bytes = 3 * int(spr.sum())
    n = head["n_records"]
    if n < 0:
        n = (len(raw) - head["header_bytes"]) // record_bytes
    body = np.frombuffer(raw, np.uint8, n * record_bytes, head["header_bytes"]).reshape(n, -1)
    offsets = np.r_[0, np.cumsum(3 * spr)]
    out = {}
    for i, label in enumerate(head["label"]):
        if channels is not None and label not in channels:
            continue
        part = body[:, offsets[i]:offsets[i + 1]]
        if "Annotations" in label:
            out[label] = np.ascontiguousarray(part)
            continue
        b = part.reshape(-1, 3).astype(np.int32)
        digital = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16))
        digital = np.where(digital >= 1 << 23, digital - (1 << 24), digital)
        gain = (head["pmax"][i] - head["pmin"][i]) / (head["dmax"][i] - head["dmin"][i])
        out[label] = (digital - head["dmin"][i]) * gain + head["pmin"][i]
    head["n_records"] = n
    return head, out


def read_annotations(record_bytes) -> list[tuple[float, float | None, str]]:
    """BDF+/EDF+ time-stamped annotation lists -> (onset s, duration s, text); record-time
    keeping entries (empty text) are dropped. Bytes after the last TAL of a record (from the
    first double NUL) are padding and may hold stale data."""
    events = []
    for record in np.atleast_2d(record_bytes):
        used = bytes(record).split(b"\x00\x00", 1)[0]
        for tal in used.split(b"\x00"):
            if not tal:
                continue
            fields = tal.split(b"\x14")
            timing = fields[0].split(b"\x15")
            onset = float(timing[0])
            duration = float(timing[1]) if len(timing) > 1 and timing[1] else None
            for note in fields[1:]:
                if note:
                    events.append((onset, duration, note.decode("utf-8", "replace")))
    return events


# ------------------------------------------------------------------ events and reduction

# evt.bdf codes (confirmed on sub-001 day 1; recorded in docs/preregistration_m.md):
# 1 = left-hand, 2 = right-hand trial onset (start of the 1.5 s video cue; imagery
# 1.5-5.5 s), 9 = boundary between the five 40-trial runs, 7 and 8 = one each before the
# first trial (pre-task markers, unused).
LABELS = {"1": 0, "2": 1}
RUN_BOUNDARY = "9"
FS_IN = 1000
WINDOW_AFTER_ONSET_S = (2.5, 5.5)  # = 1-4 s after imagery onset (L's window)
N_TRIALS, N_RUNS = 200, 5


def trial_events(annotations):
    """(onset s, label, run, trial) rows of the MI trials in recorded order."""
    rows, run = [], 1
    for onset, _, code in annotations:
        if code == RUN_BOUNDARY:
            run += 1
        elif code in LABELS:
            rows.append((onset, LABELS[code], run, len(rows) + 1, int(code)))
    return pd.DataFrame(rows, columns=["onset_s", "label", "run", "trial", "code"])


OCCIPITAL = ("O1", "Oz", "O2")


def _start(head):
    return datetime.strptime(f"{head['start_date']} {head['start_time']}", "%d.%m.%y %H.%M.%S")


def vep_peak(signals, onsets_s):
    """Quality check of event alignment only (zero-phase, not part of the analysis signal):
    latency of the largest trial-averaged occipital 1-20 Hz deflection within -2..3 s."""
    from scipy.signal import butter, sosfiltfilt
    x = sosfiltfilt(butter(4, [1, 20], "bandpass", fs=FS_IN, output="sos"),
                    np.mean([signals[c] for c in OCCIPITAL], axis=0))
    on = np.round(np.asarray(onsets_s) * FS_IN).astype(int)
    on = on[(on >= 2 * FS_IN) & (on + 3 * FS_IN <= len(x))]
    if not len(on):
        return float("nan")
    epochs = np.stack([x[o - 2 * FS_IN:o + 3 * FS_IN] for o in on])
    erp = np.abs((epochs - epochs[:, int(1.5 * FS_IN):2 * FS_IN].mean(1, keepdims=True)).mean(0))
    return float(np.argmax(erp) / FS_IN - 2)


def reduce_recording(data_path, evt_path, cfg):
    """Causal 8-30 Hz at the source rate on the 19 motor channels, decimate to 250 Hz, cut
    2.5-5.5 s after trial onset; a trial is non-finite if any raw sample is non-finite in the
    guard span before its window or inside it (L's reduce_phase), or if its window extends
    past the end of the recording (its signals are then NaN). Malformed recordings raise
    ValueError or KeyError: unreadable header or annotations, a missing motor channel or
    annotation signal, a sampling rate other than 1000 Hz, or no class-coded trial."""
    head, signals = read_bdf(data_path, channels=set(MOTOR) | set(OCCIPITAL))
    ehead, events = read_bdf(evt_path)
    fs = head["samples"][head["label"].index(MOTOR[0])] / head["record_seconds"]
    if fs != FS_IN:
        raise ValueError(f"Sampling rate {fs} Hz, expected {FS_IN} Hz")
    # Event onsets count from the data start whatever the header start-time offset (header
    # times have 1 s resolution); the offset is recorded, never corrected (occipital visual
    # evoked peak after onset recorded per user as vep_peak_s).
    offset = (_start(ehead) - _start(head)).total_seconds()
    raw = np.column_stack([signals[c] for c in MOTOR])
    finite_raw = np.isfinite(raw)
    bad_sample = ~finite_raw.all(axis=1)
    sos = bandpass_sos(dict(cfg, sfreq=FS_IN))
    filtered = causal_bandpass(sos, np.where(finite_raw, raw, 0.0))
    annotations = read_annotations(events["BDF Annotations"])
    table = trial_events(annotations)
    if table.empty:
        raise ValueError("No class-coded trial")
    start = int(round(WINDOW_AFTER_ONSET_S[0] * FS_IN))
    stop = int(round(WINDOW_AFTER_ONSET_S[1] * FS_IN))
    guard = int(round(cfg["nonfinite_guard_s"] * FS_IN))
    step = FS_IN // 250
    X = np.full((len(table), len(MOTOR), (stop - start) // step), np.nan, np.float32)
    finite = np.zeros(len(table), bool)
    onsets = np.round(table.onset_s.to_numpy() * FS_IN).astype(int)
    past_end = onsets + stop > len(raw)
    for i in np.flatnonzero(~past_end):
        a, b = onsets[i] + start, onsets[i] + stop
        X[i] = filtered[a:b:step].T
        finite[i] = not bad_sample[max(a - guard, 0):b].any()
    info = dict(n_samples=len(raw), duration_s=len(raw) / FS_IN, start=f"{head['start_date']} "
                f"{head['start_time']}", event_start_offset_s=offset,
                offset_within_header_resolution=bool(abs(offset) <= 1),
                past_end_trials=int(past_end.sum()),
                vep_peak_s=vep_peak(signals, table.onset_s.to_numpy()),
                units=head["unit"][head["label"].index(MOTOR[0])],
                codes={c: int(n) for c, n in pd.Series([e[2] for e in annotations])
                       .value_counts().items()})
    return X, table.assign(finite=finite), info


def load_trials(directory, subjects=SUBJECTS, day=1, *, allow_later_days=False,
                skip_missing=False):
    """Reduced trials; day 1 = calibration ("T"), days 2-3 ("E") need post-seal authorization.

    Day 1 must hold at most 200 trials in five runs of at most 40 (cross-fitting uses the
    runs). Days 2-3 accept any run count and any trial count (all class-coded trials in
    recorded order); with skip_missing an absent processed file is skipped (missing
    user-day)."""
    if day not in DAYS:
        raise ValueError("Day must be 1, 2 or 3")
    if day != 1 and not allow_later_days:
        raise LeakageError("Recording days 2-3 are sealed")
    if day == 1 and skip_missing:
        raise ValueError("Every day-1 recording is required")
    arrays, frames = [], []
    for subject in subjects:
        if subject not in SUBJECTS:
            raise ValueError("Yang2025 two-class subject must be 1..51")
        path = Path(directory) / f"day{day}_sub{subject:03d}.npz"
        if skip_missing and not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            X, trial, run = data["X"], data["trial"], data["run"]
            if X.ndim != 3 or X.shape[1:] != (len(MOTOR), 750) or not len(X):
                raise ValueError("Expected n x 19 x 750 reduced Yang2025 signals")
            if not np.array_equal(trial, np.arange(1, len(X) + 1)):
                raise ValueError("Trials must preserve recorded order 1..n")
            # Two day-1 recordings (sub-023, sub-026) lack one class-coded trial (code 255).
            sizes = np.bincount(run, minlength=N_RUNS + 1)
            if day == 1 and (len(X) > N_TRIALS or np.any(np.diff(run) < 0) or sizes[0]
                             or len(sizes) != N_RUNS + 1
                             or (sizes[1:] > N_TRIALS // N_RUNS).any() or not sizes[1:].all()):
                raise ValueError("Day 1 requires five runs of at most 40 trials")
            if not np.isin(data["label"], [0, 1]).all():
                raise ValueError("Expected binary MI labels")
            finite = data["finite"].astype(bool)
            if not np.isfinite(X[finite]).all():
                raise ValueError("Rows marked retained contain non-finite signals")
            arrays.append(X)
            frames.append(pd.DataFrame(dict(
                subject=subject, session="T" if day == 1 else "E", run=run, trial=trial,
                label=data["label"], artifact=False, finite=finite, day=day)))
    if not arrays:
        raise ValueError(f"No reduced recording of day {day}")
    return Trials(np.concatenate(arrays), pd.concat(frames, ignore_index=True))


def read_checksums(path) -> dict:
    """{name: sha256} of a `sha256  name` checksum file; empty if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return {}
    return {line.split("  ", 1)[1]: line.split("  ", 1)[0]
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def sha256_file(path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            sha.update(block)
    return sha.hexdigest()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def resolve_source(opener=urllib.request.urlopen) -> dict:
    """Figshare version, file id, size and the signed storage URL of the archive.

    The download URL's redirect is read without following it, so resolving transfers no
    archive bytes."""
    with opener(API, timeout=60) as response:
        article = json.load(response)
    (file,) = article["files"]
    try:
        urllib.request.build_opener(_NoRedirect).open(file["download_url"], timeout=60).close()
        raise OSError("Figshare download URL did not redirect")
    except urllib.error.HTTPError as exc:
        if exc.code not in (301, 302, 303, 307, 308):
            raise
        url = exc.headers["Location"]
    return dict(url=url, version=article["version"], file_id=file["id"], size=file["size"],
                md5=file["computed_md5"], resolved_utc=utc_now())
