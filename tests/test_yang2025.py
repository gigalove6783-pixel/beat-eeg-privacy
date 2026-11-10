"""Yang2025 reader, allowlist, reduction and day guard on synthetic files (no download)."""
import hashlib
import io
import json
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from beat.config import load_config
from beat.data import LeakageError
from beat.yang2025 import (DAY1, DAY23, METADATA, MOTOR, OCCIPITAL, OPENSSL, STAMP_DIR,
                           RangeZip, allowlist, load_trials, member_path, post_seal_allowlist,
                           read_annotations, read_bdf, reduce_recording, seal_imprint)


def _field(values, width):
    return b"".join(str(v).encode("latin-1").ljust(width)[:width] for v in values)


def write_bdf(path, signals, fs, annotations=None, start=("20.03.19", "13.53.10")):
    """Minimal 24-bit BDF(+) writer: signals {label: physical samples}; 1 s records."""
    labels = list(signals)
    n_rec = len(next(iter(signals.values()))) // fs if signals else len(annotations)
    spr = [fs] * len(labels)
    pmin, pmax, dmin, dmax = -375000.0, 375000.0, -8388608, 8388607
    if annotations is not None:
        labels.append("BDF Annotations")
        spr.append(100)
    ns = len(labels)
    head = (b"\xffBIOSEMI" + _field(["X"], 80) + _field(["X"], 80) + _field([start[0]], 8)
            + _field([start[1]], 8) + _field([256 * (ns + 1)], 8)
            + _field(["BDF+C" if annotations is not None else "24BIT"], 44)
            + _field([n_rec], 8) + _field([1], 8) + _field([ns], 4))
    head += _field(labels, 16) + _field([""] * ns, 80) + _field(["uV"] * ns, 8)
    head += _field([pmin] * ns, 8) + _field([pmax] * ns, 8)
    head += _field([dmin] * ns, 8) + _field([dmax] * ns, 8)
    head += _field([""] * ns, 80) + _field(spr, 8) + _field([""] * ns, 32)
    gain = (pmax - pmin) / (dmax - dmin)
    records = []
    for r in range(n_rec):
        parts = []
        for label in labels[:len(signals)]:
            x = np.asarray(signals[label][r * fs:(r + 1) * fs])
            d = np.round((x - pmin) / gain + dmin).astype(np.int64) & 0xFFFFFF
            parts.append(np.column_stack([d & 255, (d >> 8) & 255, d >> 16]).astype(np.uint8)
                         .tobytes())
        if annotations is not None:
            tal = f"+{r}\x14\x14\x00".encode()
            for onset, code in annotations.get(r, []):
                tal += f"+{onset:.4f}\x150\x14{code}\x14\x00".encode()
            parts.append(tal.ljust(300, b"\x00"))
        records.append(b"".join(parts))
    path.write_bytes(head + b"".join(records))


def synthetic_recording(tmp_path, n_trials=10, fs=1000, seed=0, evt_start="13.53.10",
                        channels=MOTOR + OCCIPITAL, data_seconds=None):
    rng = np.random.default_rng(seed)
    seconds = 20 + 8 * n_trials
    signals = {c: rng.normal(0, 10, seconds * fs)[:(data_seconds or seconds) * fs]
               for c in channels}
    onsets = [(10.0 + 8 * i + (2 if i >= n_trials // 2 else 0), "1" if i % 2 else "2")
              for i in range(n_trials)]
    events = [(5.0, "7"), (7.0, "8")] + onsets[:n_trials // 2] + [(9.0 + 8 * (n_trials // 2), "9")] \
        + onsets[n_trials // 2:]
    by_record = {}
    for i, (onset, code) in enumerate(events):
        by_record.setdefault(i, []).append((onset, code))
    write_bdf(tmp_path / "data.bdf", signals, fs)
    write_bdf(tmp_path / "evt.bdf", {}, fs, annotations=by_record | {
        i: [] for i in range(len(events), seconds)}, start=("20.03.19", evt_start))
    return signals, onsets


def test_bdf_roundtrip_and_annotations(tmp_path):
    signals, onsets = synthetic_recording(tmp_path)
    head, decoded = read_bdf(tmp_path / "data.bdf", channels={"C3", "Cz"})
    assert set(decoded) == {"C3", "Cz"} and head["samples"][0] == 1000
    assert np.max(np.abs(decoded["C3"] - signals["C3"])) < 0.1  # 24-bit quantization
    _, events = read_bdf(tmp_path / "evt.bdf")
    notes = read_annotations(events["BDF Annotations"])
    assert [n[2] for n in notes].count("9") == 1
    assert [(round(o, 4), c) for o, _, c in notes if c in "12"] == onsets


def test_annotation_padding_is_ignored():
    record = np.frombuffer(b"+0\x14\x14\x00+1.5\x150\x141\x14\x00\x00\x00\x17`\xe3\x96?", np.uint8)
    assert read_annotations(record) == [(1.5, 0.0, "1")]


def test_reduction_is_causal_and_labels_runs(tmp_path):
    signals, onsets = synthetic_recording(tmp_path)
    cfg = load_config("data")
    X, table, info = reduce_recording(tmp_path / "data.bdf", tmp_path / "evt.bdf", cfg)
    assert X.shape == (10, 19, 750) and X.dtype == np.float32
    assert table.label.tolist() == [1 if c == "2" else 0 for _, c in onsets]
    assert table.run.tolist() == [1] * 5 + [2] * 5 and table.finite.all()
    assert info["codes"] == {"1": 5, "2": 5, "7": 1, "8": 1, "9": 1}
    assert info["event_start_offset_s"] == 0 and -2 <= info["vep_peak_s"] <= 3
    # Changing samples after trial 4's window (onset + 5.5 s) leaves trials 1-4 unchanged.
    cut = int((onsets[3][0] + 5.5) * 1000)
    altered = {c: np.r_[v[:cut], v[cut:] * 50] for c, v in signals.items()}
    write_bdf(tmp_path / "data.bdf", altered, 1000)
    X2, _, _ = reduce_recording(tmp_path / "data.bdf", tmp_path / "evt.bdf", cfg)
    np.testing.assert_array_equal(X2[:4], X[:4])
    assert not np.allclose(X2[5:], X[5:])


def test_late_windows_become_non_finite_and_offsets_are_recorded(tmp_path):
    cfg = load_config("data")
    signals, onsets = synthetic_recording(tmp_path)
    X, _, _ = reduce_recording(tmp_path / "data.bdf", tmp_path / "evt.bdf", cfg)
    # The recording ends 4 s after the last onset (window 2.5-5.5 s), and the event file
    # starts 5 s after the data file: neither aborts.
    synthetic_recording(tmp_path, evt_start="13.53.15", data_seconds=int(onsets[-1][0]) + 4)
    X2, table, info = reduce_recording(tmp_path / "data.bdf", tmp_path / "evt.bdf", cfg)
    assert info["past_end_trials"] == 1 and table.finite.tolist() == [True] * 9 + [False]
    assert np.isnan(X2[-1]).all()
    np.testing.assert_array_equal(X2[:-1], X[:-1])
    assert info["event_start_offset_s"] == 5 and not info["offset_within_header_resolution"]


@pytest.mark.parametrize("change, error", [
    (dict(fs=500), ValueError), (dict(channels=MOTOR[:-1] + OCCIPITAL), KeyError),
    (dict(n_trials=0), ValueError)])
def test_malformed_recordings_raise(tmp_path, change, error):
    synthetic_recording(tmp_path, **change)
    with pytest.raises(error):
        reduce_recording(tmp_path / "data.bdf", tmp_path / "evt.bdf", load_config("data"))


def test_allowlist_is_day1_and_metadata_only():
    assert allowlist() == METADATA | DAY1
    assert len(DAY1) == 102 and len(DAY23) == 204 and not DAY1 & DAY23
    assert all("/ses-01/" in path for path in DAY1)
    assert member_path(51, 1, "data").endswith("2C dataset/sub-051/ses-01/eeg/data.bdf")


class _Response(io.BytesIO):
    def __init__(self, data, headers):
        super().__init__(data)
        self.headers, self.status = headers, 206


class _Server:
    """In-memory archive answering Range requests like the storage server."""

    def __init__(self, blob):
        self.blob, self.requests = blob, []

    def __call__(self, request, timeout=None):
        spec = request.headers["Range"].split("=")[1]
        start, end = map(int, spec.split("-"))
        self.requests.append((start, end))
        return _Response(self.blob[start:end + 1], {
            "ETag": '"e"', "Content-Range": f"bytes {start}-{end}/{len(self.blob)}"})


def _archive(tmp_path):
    names = sorted(DAY1)[:2] + sorted(DAY23) + [
        "WBCIC_SHU Motor Imagery dataset/sourcedata/2C dataset/sub-001/ses-01/eeg/x.jpg"]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for i, name in enumerate(names):
            z.writestr(name, (name * 50).encode(), compress_type=zipfile.ZIP_DEFLATED
                       if i % 2 else zipfile.ZIP_STORED)
    blob = buffer.getvalue()
    server = _Server(blob)
    reader = RangeZip(tmp_path / "log.jsonl", opener=server, resolve=lambda opener: dict(
        url="mem://archive", version=5, file_id=1, size=len(blob)))
    return reader, server, names


def test_range_reader_streams_allowed_members_and_logs(tmp_path):
    reader, server, names = _archive(tmp_path)
    for name in names[:2]:
        digest, size = reader.stream_member(name, tmp_path / "member")
        content = (name * 50).encode()
        assert digest == hashlib.sha256(content).hexdigest() and size == len(content)
        assert (tmp_path / "member").read_bytes() == content
    log = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert len(log) == len(server.requests) and all(r["etag"] == '"e"' for r in log)
    assert {r["member"] for r in log} == {"<archive end records>", "<central directory>",
                                          *names[:2]}
    entries = reader.directory()
    for row in log:
        if row["member"] in entries:  # member requests stay inside that member's record
            entry = entries[row["member"]]
            assert entry["offset"] <= row["start"] <= row["end"]
            assert row["end"] < entry["offset"] + 30 + len(row["member"]) + 64 + entry["csize"]


def test_range_reader_refuses_every_day23_path_before_any_request(tmp_path):
    reader, server, names = _archive(tmp_path)
    reader.directory()
    before = len(server.requests)
    for name in sorted(DAY23) + [names[-1], "WBCIC_SHU Motor Imagery dataset/derivatives/x.mat"]:
        with pytest.raises(LeakageError):
            reader.stream_member(name, tmp_path / "never")
    assert len(server.requests) == before and not (tmp_path / "never").exists()


def test_loader_refuses_days_2_and_3(tmp_path):
    for day in (2, 3):
        with pytest.raises(LeakageError):
            load_trials(tmp_path, [1], day)


def test_loader_reads_day1_layout(tmp_path):
    rng = np.random.default_rng(1)
    np.savez(tmp_path / "day1_sub002.npz", X=rng.normal(size=(200, 19, 750)).astype(np.float32),
             label=np.tile([0, 1], 100), finite=np.ones(200, bool),
             run=np.repeat(np.arange(1, 6), 40), trial=np.arange(1, 201),
             onset_s=np.arange(200.0), code=np.tile([1, 2], 100))
    run = np.repeat(np.arange(1, 6), 40)[1:]  # one uncoded trial in run 1 (sub-023, sub-026)
    np.savez(tmp_path / "day1_sub003.npz", X=rng.normal(size=(199, 19, 750)).astype(np.float32),
             label=np.tile([0, 1], 100)[1:], finite=np.ones(199, bool), run=run,
             trial=np.arange(1, 200), onset_s=np.arange(199.0), code=np.tile([1, 2], 100)[1:])
    trials = load_trials(tmp_path, [2, 3], 1)
    assert len(trials) == 399 and set(trials.session) == {"T"} and trials.meta.day.eq(1).all()


def test_later_day_loader_accepts_other_layouts_and_skips_missing_files(tmp_path):
    rng = np.random.default_rng(2)
    n = 230  # more than 200 class-coded trials in three runs: all kept in recorded order
    np.savez(tmp_path / "day2_sub001.npz", X=rng.normal(size=(n, 19, 750)).astype(np.float32),
             label=np.arange(n) % 2, finite=np.ones(n, bool), run=np.repeat([1, 2, 3], [80, 80, 70]),
             trial=np.arange(1, n + 1), onset_s=np.arange(float(n)), code=np.arange(n) % 2 + 1)
    trials = load_trials(tmp_path, [1, 2], 2, allow_later_days=True, skip_missing=True)
    assert len(trials) == n and set(trials.subject) == {1} and set(trials.session) == {"E"}
    with pytest.raises(FileNotFoundError):
        load_trials(tmp_path, [1, 2], 2, allow_later_days=True)
    with pytest.raises(ValueError):
        load_trials(tmp_path, [1], 1, skip_missing=True)


# ------------------------------------------------------------------ post-seal timestamp guard

OPENSSL_BIN = str(OPENSSL) if sys.platform == "win32" else "openssl"
TS_CONFIG = """[ req ]
distinguished_name = dn
[ dn ]
[ ca_ext ]
basicConstraints = critical, CA:TRUE
keyUsage = critical, keyCertSign, cRLSign
subjectKeyIdentifier = hash
[ tsa_ext ]
basicConstraints = critical, CA:FALSE
keyUsage = critical, digitalSignature, nonRepudiation
extendedKeyUsage = critical, timeStamping
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid
[ tsa ]
default_tsa = tsa_config
[ tsa_config ]
serial = serial.txt
signer_digest = sha256
default_policy = 1.2.3.4.1
digests = sha256
accuracy = secs:1
ordering = no
tsa_name = no
ess_cert_id_chain = no
ess_cert_id_alg = sha256
"""


def _openssl(cwd, *args):
    subprocess.run([OPENSSL_BIN, *args], cwd=cwd, check=True, capture_output=True)


def _timestamp(directory, commit):
    """Local test TSA (CA, TSA certificate) and an RFC 3161 response over seal_imprint(commit),
    stored under the pre-registered names."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ts.cnf").write_text(TS_CONFIG)
    (directory / "serial.txt").write_text("01\n")
    key = ("-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes")
    _openssl(directory, "req", "-x509", *key, "-keyout", "ca.key", "-out", "cacert.pem",
             "-days", "2", "-subj", "/CN=test-ca", "-config", "ts.cnf", "-extensions", "ca_ext")
    _openssl(directory, "req", "-new", *key, "-keyout", "tsa.key", "-out", "tsa.csr",
             "-subj", "/CN=test-tsa", "-config", "ts.cnf")
    _openssl(directory, "x509", "-req", "-in", "tsa.csr", "-CA", "cacert.pem", "-CAkey",
             "ca.key", "-CAcreateserial", "-out", "tsa.crt", "-days", "2", "-extfile", "ts.cnf",
             "-extensions", "tsa_ext")
    _openssl(directory, "ts", "-query", "-digest", seal_imprint(commit), "-sha256", "-cert",
             "-out", "seal4.tsq")
    _openssl(directory, "ts", "-reply", "-queryfile", "seal4.tsq", "-inkey", "tsa.key",
             "-signer", "tsa.crt", "-config", "ts.cnf", "-out", "seal4.tsr")


def _git(repo, *args):
    return subprocess.check_output(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                                   cwd=repo, stderr=subprocess.DEVNULL).decode().strip()


@pytest.fixture
def sealed_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "file.txt").write_text("sealed\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "seal")
    _git(repo, "tag", "-a", "seal-4", "-m", "seal 4")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_post_seal_allowlist_requires_a_token_over_the_seal_commit(sealed_repo):
    repo, commit = sealed_repo
    stamp = repo / STAMP_DIR
    with pytest.raises(LeakageError, match="Missing"):  # no token yet
        post_seal_allowlist(commit, root=repo, openssl=OPENSSL_BIN)
    _timestamp(stamp, "0" * 40)  # a valid token, but over another commit id
    with pytest.raises(LeakageError, match="does not verify"):
        post_seal_allowlist(commit, root=repo, openssl=OPENSSL_BIN)
    assert not (stamp / "guard_verification.jsonl").exists()
    (stamp / "seal4.tsr").unlink()
    with pytest.raises(LeakageError, match="Missing"):
        post_seal_allowlist(commit, root=repo, openssl=OPENSSL_BIN)
    _timestamp(stamp, commit)
    allowed = post_seal_allowlist(commit[:12], root=repo, openssl=OPENSSL_BIN)
    assert DAY23 <= allowed and DAY1 <= allowed
    (record,) = [json.loads(line) for line in
                 (stamp / "guard_verification.jsonl").read_text().splitlines()]
    assert record["commit"] == commit and record["imprint_sha256"] == seal_imprint(commit)
    assert record["exit_code"] == 0 and record["token_time"].endswith("GMT")


def test_post_seal_allowlist_requires_the_annotated_seal_tag(sealed_repo):
    repo, commit = sealed_repo
    _timestamp(repo / STAMP_DIR, commit)
    (repo / "file.txt").write_text("later\n")
    _git(repo, "commit", "-qam", "later")
    later = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(LeakageError):  # seal-4 names the first commit, not the later one
        post_seal_allowlist(later, root=repo, openssl=OPENSSL_BIN)
    with pytest.raises(LeakageError):
        post_seal_allowlist(None, root=repo, openssl=OPENSSL_BIN)
    _git(repo, "tag", "-d", "seal-4")
    _git(repo, "tag", "seal-4", commit)  # lightweight tag
    with pytest.raises(LeakageError):
        post_seal_allowlist(commit, root=repo, openssl=OPENSSL_BIN)
    _git(repo, "tag", "-d", "seal-4")
    with pytest.raises(LeakageError):  # no tag at all
        post_seal_allowlist(commit, root=repo, openssl=OPENSSL_BIN)
