"""Resolve every DOI, arXiv identifier and URL in the reference lists and record the output.

    python scripts/check_references.py

Runs the network reference check of `beat.verify` (DOIs through the doi.org handle API, arXiv
identifiers through the arXiv API, other URLs by an HTTP GET) over the reference sections of
README.md, docs/report.md and docs/related_work.md, prints the result and writes it unchanged to
results/reference_check/references_<UTC time>.txt. Exit status 0 when every target resolves.
"""
from __future__ import annotations

import contextlib
import io
import sys
from datetime import datetime, timezone

from beat.verify import ROOT, check_references

OUT = ROOT / "results" / "reference_check"


def main() -> int:
    started = datetime.now(timezone.utc)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print(f"reference check, {started:%Y-%m-%d %H:%M} UTC")
        errors = check_references(ROOT)
        print(f"references: {'FAIL' if errors else 'pass'}")
    text = buffer.getvalue()
    sys.stdout.write(text)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"references_{started:%Y%m%dT%H%MZ}.txt"
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"recorded: {path.relative_to(ROOT).as_posix()}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
