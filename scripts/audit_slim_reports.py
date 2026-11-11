"""Slim copies of the transmission ladder's reports, so that `beat-audit compare` (the builder's
case of the README, docs/plan_v3.md §5 A10) can be followed from a clean checkout.

The ladder's full `report.json` files (component-level evidence, up to 97 MB each) are not tracked
in Git. `report.slim.json`, written next to each, is derived from it by removing data only: no
value is recomputed or rounded (numbers are re-serialised from their parsed values, which
round-trips exactly), and the order of the kept fields is the full report's.

Kept, at every level, is what `beat_audit.report.compare_reports` and the `beat-audit compare` CLI
read, with the report's own small metadata:
- report: schema version, cohort, days, kind, day order, model seeds, configuration (settings and
  condition specifications), recorded hashes of source, configuration and inputs, warnings,
  interpretation, paired contrast estimates, smallest detected k, run time, unavailable analyses;
- cell: condition, relation, days, k, resolved A/B selectors, status, reason, analysed and
  excluded users, population payload and population hash, strongest comparator and wording,
  skipped comparators;
- comparator: AUC, interval, p, Holm p, wording, warning, reliability flag, draws, replicates,
  exact matchings, comparison bounds, window summary (settings).
The population payload (users, fold assignment of users, analysed row positions per model seed)
stays whole because `compare` recomputes its hash and refuses on a mismatch; it holds row
positions, not outputs or features. Everything else is removed, notably per comparator the
per-user and partner AUCs, training records, metrics and curves, operating points, score
histogram and the sequential, closed-set and gallery results. Each slim report records the SHA-256
and size of the full report it came from and the removed fields.

Usage: python scripts/audit_slim_reports.py [path/to/report.json ...]
(default: every report under results/audit/ladder/)
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LADDER = ROOT / "results" / "audit" / "ladder"
REPORT = ["schema_version", "purpose", "cohort", "days", "kind", "day_order", "day_order_source",
          "model_seeds", "configuration", "hashes", "cells", "contrasts", "warnings",
          "interpretation", "smallest_detected_k", "runtime_seconds", "unavailable_analyses"]
CELL = ["condition", "relation", "k", "days", "excluded_users", "analysed_users", "status",
        "reason", "analysis_requested", "comparison_users", "comparators", "skipped_comparators",
        "resolved_pairs", "population", "population_hash", "strongest", "strongest_interpretation",
        "wording"]
COMPARATOR = ["auc", "interval", "p", "holm_p", "wording", "warning", "unreliable", "draws",
              "replicates", "exact_matchings", "comparison_bounds", "window_summary"]


def keep(item, fields, where, removed):
    removed.update(where + key for key in item if key not in fields)
    return {key: value for key, value in item.items() if key in fields}


def slim_cell(cell, where, removed):
    out = keep(cell, CELL, where, removed)
    out["comparators"] = {name: keep(result, COMPARATOR, f"{where}comparators.<comparator>.",
                                     removed)
                          for name, result in cell["comparators"].items()}
    return out


def slim(path):
    """The slim report of the full report at path, as a dict."""
    path = Path(path)
    data = path.read_bytes()
    report = json.loads(data)
    removed = set()
    out = keep(report, REPORT, "", removed)
    for field in ("cells", "unavailable_analyses"):
        out[field] = [slim_cell(cell, f"{field}[].", removed) for cell in report[field]]
    return {"slim": {"derived_from": {"file": path.name, "sha256": hashlib.sha256(data).hexdigest(),
                                      "bytes": len(data)},
                     "removed": sorted(removed),
                     "note": "fields removed from the full report; kept values unchanged"},
            **out}


def write_slim(path):
    """Writes report.slim.json next to the full report (LF, deterministic); returns its path."""
    target = Path(path).with_name("report.slim.json")
    target.write_bytes((json.dumps(slim(path), separators=(",", ":")) + "\n").encode())
    return target


def main(argv=None):
    paths = [Path(p) for p in (sys.argv[1:] if argv is None else argv)]
    paths = paths or sorted(LADDER.glob("*/*/*/report.json"))
    if not paths:
        raise SystemExit("audit_slim_reports: no full report.json found; the slim reports are "
                         "derived from the ladder's untracked full reports")
    for path in paths:
        target = write_slim(path)
        print(f"{target.resolve().relative_to(ROOT).as_posix()}: {target.stat().st_size} bytes "
              f"(full report {Path(path).stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
