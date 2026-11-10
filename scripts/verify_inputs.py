"""Check the input data on disk against the committed SHA-256 lists in checksums/.

    python scripts/verify_inputs.py                  # BNCI2014-001 raw + Lee2019 reduced .npz
    python scripts/verify_inputs.py --lee2019-raw    # also stream the 108 raw Lee2019 files (~66 GB)

- checksums/bnci2014_001.sha256: the 18 BNCI2014-001 .mat files in data/raw/bnci2014_001/, as
  printed by scripts/download_data.py.
- checksums/lee2019_processed.sha256: the 108 reduced files in data/processed/lee2019/, hashed on
  2026-11-11 after experiment L had run; a match shows the files are unchanged since then.
- checksums/lee2019_raw.sha256: the 108 raw files, hashed by scripts/prepare_lee2019.py before each
  was deleted. The raw files are not kept, so --lee2019-raw re-downloads each one from the same
  GigaDB mirror and hashes it while streaming; nothing is written to disk.

Exit status 0 when every checked file matches.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKSUMS = ROOT / "checksums"
SETS = {  # checksum list -> directory holding the files
    "bnci2014_001.sha256": ROOT / "data" / "raw" / "bnci2014_001",
    "lee2019_processed.sha256": ROOT / "data" / "processed" / "lee2019",
}
CHUNK = 1 << 20


def read_list(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    return entries


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(CHUNK), b""):
        digest.update(block)
    return digest.hexdigest()


def check_local(list_name: str, folder: Path) -> list[str]:
    errors = []
    expected = read_list(CHECKSUMS / list_name)
    for name, digest in sorted(expected.items()):
        path = folder / name
        if not path.is_file():
            errors.append(f"{list_name}: missing {path.relative_to(ROOT)}")
            continue
        with open(path, "rb") as f:
            if sha256_stream(f) != digest:
                errors.append(f"{list_name}: hash mismatch for {name}")
    print(f"{list_name}: {len(expected) - len(errors)}/{len(expected)} files match")
    return errors


def lee2019_raw_url(name: str) -> str:
    # The mirror and path layout used by scripts/prepare_lee2019.py (sessNN_subjNN_EEG_MI.mat).
    from prepare_lee2019 import BASE_URL
    session, subject = int(name[4:6]), int(name[11:13])
    return f"{BASE_URL}session{session}/s{subject}/{name}"


def check_lee2019_raw(limit: int | None) -> list[str]:
    errors = []
    expected = sorted(read_list(CHECKSUMS / "lee2019_raw.sha256").items())[:limit]
    for name, digest in expected:
        with urllib.request.urlopen(lee2019_raw_url(name), timeout=120) as response:
            ok = sha256_stream(response) == digest
        print(f"  {'ok  ' if ok else 'FAIL'} {name}", flush=True)
        if not ok:
            errors.append(f"lee2019_raw.sha256: hash mismatch for {name}")
    print(f"lee2019_raw.sha256: {len(expected) - len(errors)}/{len(expected)} streamed files match")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lee2019-raw", action="store_true",
                        help="stream the raw Lee2019 files from the GigaDB mirror and check them")
    parser.add_argument("--limit", type=int, default=None,
                        help="with --lee2019-raw: check only the first N files (sorted by name)")
    args = parser.parse_args(argv)
    errors = []
    for list_name, folder in SETS.items():
        errors += check_local(list_name, folder)
    if args.lee2019_raw:
        errors += check_lee2019_raw(args.limit)
    for e in errors:
        print(f"  {e}")
    print("inputs: " + ("FAIL" if errors else "pass"))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
