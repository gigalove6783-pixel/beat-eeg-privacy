"""Fetch allowlisted Yang2025 raw members one at a time, reduce them and delete the raw files.

    python scripts/fetch_yang2025.py                      # day 1 (pre-seal allowlist)
    python scripts/fetch_yang2025.py --days 2 3 --seal-commit HASH   # after seal 4 only

Every byte-range request is logged by the range reader (day 1: results/exp_m/download_log.jsonl;
days 2-3: results/exp_m/download_log_day23.jsonl, so the sealed day-1 log is never appended to).
SHA-256 of each raw member is recorded before the member is deleted, and SHA-256 of each
processed file when it is written; reduction and epoching live in beat/yang2025.py. At least
30 GB of disk stays free.

Every user-day gets one line in results/exp_m/day<days>_events.jsonl with its counts and status:
"reduced"; "absent" (a member is not in the archive); "unparseable" (reduce_recording raised
ValueError, KeyError, IndexError or UnicodeDecodeError). Absent and unparseable user-days have
no processed file and are missing cells in the final. Transfer failures abort the run; a rerun
resumes and skips user-days that already have a status line.
"""
import argparse
import json
import shutil

import numpy as np

from beat.config import ROOT, load_config
from beat.yang2025 import (METADATA, N_TRIALS, SUBJECTS, RangeZip, allowlist, member_path,
                           post_seal_allowlist, read_checksums, reduce_recording, sha256_file)

RAW = ROOT / "data" / "raw" / "yang2025"
PROCESSED = "data/processed/yang2025"
OUT = ROOT / PROCESSED
LOGS = {True: ROOT / "results" / "exp_m" / "download_log.jsonl",
        False: ROOT / "results" / "exp_m" / "download_log_day23.jsonl"}
EVENTS = ROOT / "results" / "exp_m"
CHECKSUMS = {1: ROOT / "checksums" / "yang2025_day1.sha256",
             2: ROOT / "checksums" / "yang2025_day23.sha256",
             3: ROOT / "checksums" / "yang2025_day23.sha256"}
MIN_FREE = 30 * 1024 ** 3
MALFORMED = (ValueError, KeyError, IndexError, UnicodeDecodeError)


def record(path, digest, name):
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"{digest}  {name}\n")


def ensure_space(needed):
    free = shutil.disk_usage(ROOT).free
    if free - needed < MIN_FREE:
        raise OSError(f"Fetching {needed} bytes would leave less than 30 GB free ({free} free)")


def fetch(reader, name, target, checksum_path):
    """Download one member, record its hash; returns the local path."""
    ensure_space(reader.directory()[name]["usize"])
    digest, _ = reader.stream_member(name, target)
    known = read_checksums(checksum_path).get(name)
    if known and known != digest:
        raise ValueError(f"Hash differs from the recorded hash: {name}")
    if not known:
        record(checksum_path, digest, name)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, nargs="+", default=[1])
    parser.add_argument("--subjects", type=int, nargs="+", default=list(SUBJECTS))
    parser.add_argument("--seal-commit")
    args = parser.parse_args(argv)
    if args.days == [1]:
        allowed = allowlist()
    elif set(args.days) <= {2, 3}:
        from beat.exp_m import load_cfg, verify_seal
        verify_seal(args.seal_commit, load_cfg())
        allowed = post_seal_allowlist(args.seal_commit)
    else:
        raise SystemExit("Fetch day 1 alone (pre-seal) or days 2-3 alone (post-seal)")
    cfg = load_config("data")
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    reader = RangeZip(LOGS[args.days == [1]], allowed)
    if args.days == [1]:
        meta_dir = RAW / "metadata"
        meta_dir.mkdir(exist_ok=True)
        for name in sorted(METADATA):
            target = meta_dir / name.rsplit("/", 1)[1]
            if name not in read_checksums(CHECKSUMS[1]):
                fetch(reader, name, target, CHECKSUMS[1])
    summary_path = EVENTS / f"day{'_'.join(map(str, args.days))}_events.jsonl"
    done = set()
    if summary_path.exists():
        done = {(r["subject"], r["day"]) for r in map(
            json.loads, summary_path.read_text(encoding="utf-8").splitlines())}
    for day in args.days:
        for subject in args.subjects:
            out = OUT / f"day{day}_sub{subject:03d}.npz"
            if out.exists() or (subject, day) in done:
                continue
            row = dict(subject=subject, day=day)
            names = {kind: member_path(subject, day, kind) for kind in ("evt", "data")}
            absent = [name for name in names.values() if name not in reader.directory()]
            if absent:
                write_status(summary_path, dict(row, status="absent", absent=absent))
                continue
            paths = {}
            for kind, name in names.items():
                target = RAW / f"sub-{subject:03d}_ses-{day:02d}_{kind}.bdf"
                known = read_checksums(CHECKSUMS[day]).get(name)
                if target.exists() and known and sha256_file(target) == known:
                    paths[kind] = target
                else:
                    paths[kind] = fetch(reader, name, target, CHECKSUMS[day])
            try:
                X, table, info = reduce_recording(paths["data"], paths["evt"], cfg)
            except MALFORMED as exc:
                row.update(status="unparseable", error=f"{type(exc).__name__}: {exc}")
            else:
                np.savez(out, X=X, label=table.label.to_numpy(), finite=table.finite.to_numpy(),
                         run=table.run.to_numpy(), trial=table.trial.to_numpy(),
                         onset_s=table.onset_s.to_numpy(), code=table.code.to_numpy())
                digest = sha256_file(out)
                record(CHECKSUMS[day], digest, f"{PROCESSED}/{out.name}")
                intervals = np.diff(table.onset_s.to_numpy())
                row.update(
                    status="reduced", trials=len(table),
                    extra_trials=max(0, len(table) - N_TRIALS),
                    runs={int(r): int(n) for r, n in table.run.value_counts().sort_index().items()},
                    left=int((table.label == 0).sum()), right=int((table.label == 1).sum()),
                    retained=int(table.finite.sum()),
                    onset_interval_s=dict(min=float(intervals.min()),
                                          median=float(np.median(intervals)))
                    if len(intervals) else None,
                    processed_sha256=digest, **info)
            write_status(summary_path, row)
            for path in paths.values():
                path.unlink()
            print(f"day {day} sub-{subject:03d}: {row['status']}", flush=True)


def write_status(path, row):
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
