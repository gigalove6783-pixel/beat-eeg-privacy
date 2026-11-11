"""Console entry point. Errors name an input field and a corrective action."""

import argparse
import json
from pathlib import Path
import sys
import time

from .api import AuditConfig, audit_stream


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(prog="beat-audit")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="audit one transmitted-output NPZ/CSV")
    run.add_argument("input")
    run.add_argument("--out", required=True)
    run.add_argument("--kind", choices=("features", "commands"))
    run.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100])
    run.add_argument("--cohort", default="unspecified")
    run.add_argument("--seed", type=int, default=20261112)
    run.add_argument("--fold-seed", type=int, default=20261111)
    run.add_argument("--day-order", nargs="+", help="all day labels, earliest to latest")
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--seconds-per-output", type=float)
    compare = commands.add_parser("compare", help="merge matched alternatives and builder utility")
    compare.add_argument("reports", nargs="+", help="alternative=path/to/report.json")
    compare.add_argument("--utility", required=True, help="JSON mapping alternative to utility rates")
    compare.add_argument("--out", required=True, help="output CSV table")
    compare.add_argument("--derived-from", action="append", default=[],
                         help="declare COMMAND_ALTERNATIVE=FEATURE_ALTERNATIVE (repeatable)")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            config = AuditConfig(
                k_grid=tuple(args.k), cohort=args.cohort, seed=args.seed, fold_seed=args.fold_seed,
                threads=args.threads, seconds_per_output=args.seconds_per_output,
                day_order=tuple(args.day_order) if args.day_order is not None else None,
            )
            report = audit_stream(args.input, config=config, kind=args.kind, out=args.out)
            from .resources import peak_memory_bytes
            print(json.dumps({"runtime_seconds": time.perf_counter() - started,
                              "peak_rss_bytes": peak_memory_bytes()}))
            bad = [cell for cell in report["cells"] if cell["status"] != "ok"]
            for cell in bad:
                print(f"{cell['status']}: {cell['reason']}", file=sys.stderr)
            return 1 if any(c["status"] == "failed" for c in bad) else (
                2 if not any(c["status"] == "ok" for c in report["cells"]) else 0)
        from .report import compare_reports, write_comparison
        if any("=" not in value for value in args.reports):
            raise ValueError("reports: supply each input as alternative=path/to/report.json")
        reports = dict(value.split("=", 1) for value in args.reports)
        if len(reports) != len(args.reports):
            raise ValueError("reports: supply unique alternative names")
        utility = json.loads(Path(args.utility).read_text(encoding="utf-8"))
        if any("=" not in value for value in args.derived_from):
            raise ValueError("derived_from: use --derived-from COMMAND=FEATURE")
        derivations = dict(value.split("=", 1) for value in args.derived_from)
        if len(derivations) != len(args.derived_from):
            raise ValueError("derived_from: declare each command alternative only once")
        write_comparison(compare_reports(reports, utility, derived_from=derivations), args.out)
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"beat-audit: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
