"""Protocol metadata and coverage guards, separate from numerical audit calculations."""

from datetime import date
from decimal import Decimal, InvalidOperation


POST_HOC_RANKING = (
    "Observed ranking on target users (post hoc), not selected on development users"
)


def chronological_days(labels, explicit=None, *, required=True):
    """Infer only numeric or ISO-date chronology; opaque defaults require an explicit order.

    Fully specified Condition pairs do not depend on inferred chronology. Their
    callers may therefore retain existing opaque labels without supplying an
    unused global order; this is recorded as ``explicit_conditions``.
    """
    days = sorted(set(map(str, labels)))
    if explicit is not None:
        order = list(map(str, explicit))
        if len(order) != len(days) or set(order) != set(days):
            raise ValueError("day_order: list every observed day exactly once in chronological order")
        return order, "explicit"
    try:
        keys = [Decimal(day) for day in days]
        numeric = all(key.is_finite() for key in keys)
    except InvalidOperation:
        numeric = False
    if numeric and len(set(keys)) == len(days):
        return [day for _, day in sorted(zip(keys, days))], "numeric_labels"
    try:
        keys = [date.fromisoformat(day) for day in days]
    except ValueError:
        keys = []
    if keys and len(set(keys)) == len(days):
        return [day for _, day in sorted(zip(keys, days))], "iso_dates"
    if required:
        raise ValueError(
            "day_order: chronology is ambiguous; provide AuditConfig(day_order=(first, ...)) "
            "or --day-order FIRST ... LAST, listing every day exactly once"
        )
    return None, "explicit_conditions"


def canonical_pairs(pairs):
    """Canonical resolved physical selectors: [[side_A, side_B], ...], [day, segment].

    Side and pair membership ordering is representational; physical A and B are
    not interchanged. Repeated pairs are retained because they change weights.
    This canonicalisation is metadata-only: execution order is never changed.
    """
    return sorted([
        [sorted([[str(day), str(segment)] for day, segment in side]) for side in pair]
        for pair in pairs
    ])


def unavailable_analyses(cells, days, users):
    """Explicit non-evaluable coverage rows, outside the requested/scored cell manifest."""
    seen = {(cell["relation"], cell["k"]) for cell in cells}
    reasons = {
        "same-day": "segment/condition: same-day analysis is unavailable; supply at least two "
                    "disjoint within-day segments and explicit A/B pairs, or omit segment "
                    "labels to derive halves",
        "cross-day": "day/condition: cross-day analysis is unavailable; supply explicit A/B "
                     "pairs from different recording days",
    }
    return [
        {
            "condition": relation, "relation": relation, "days": sorted(map(str, days)),
            "k": k, "status": "not_evaluable", "reason": reasons[relation],
            "analysis_requested": False, "analysed_users": [], "excluded_users": [],
            "comparison_users": sorted(map(str, users)), "comparators": {},
            "resolved_pairs": [], "population_hash": None,
        }
        for k in sorted({cell["k"] for cell in cells})
        for relation in ("same-day", "cross-day") if (relation, k) not in seen
    ]


def report_entries(report):
    """Include explicit coverage gaps, also when rendering an older in-memory report."""
    cells = list(report["cells"])
    if "unavailable_analyses" in report:
        return cells + report["unavailable_analyses"]
    users = sorted({u for cell in cells
                    for u in cell["analysed_users"] + cell.get("excluded_users", [])})
    return cells + unavailable_analyses(cells, report["days"], users)


def resolved_pairs_from_report(report, cell):
    """Read resolved selectors, or recover fully qualified selectors from legacy metadata.

    A legacy whole-day wildcard cannot safely be expanded without its input
    segment schema. Refuse it rather than compare an unverified partition.
    """
    if "resolved_pairs" in cell:
        return canonical_pairs(cell["resolved_pairs"])
    conditions = report.get("configuration", {}).get("conditions", [])
    condition = next((c for c in conditions if c["name"] == cell["condition"]), None)
    if condition is not None and all(
            selector["segment"] is not None
            for pair in condition["pairs"] for side in pair for selector in side):
        return canonical_pairs([
            [[(selector["day"], selector["segment"]) for selector in side] for side in pair]
            for pair in condition["pairs"]
        ])
    raise ValueError(
        "condition A/B selectors: resolved_pairs are missing; rerun this legacy report "
        "to record the resolved physical day/segment partition"
    )
