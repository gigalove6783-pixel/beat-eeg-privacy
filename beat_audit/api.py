"""Stable public interface for cohort adapters and builders.

Labels are canonical strings. A condition contains equally weighted physical-side
pairs, usually one per day (same-day) or later day (cross-day). A side is a tuple
of Segment selectors; ``segment=None`` selects all segments of the given day.
Each selected segment is windowed separately: windows never cross a boundary.
The one B-block matching is shared across *all* pairs, directions and model seeds.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Segment:
    """Select one segment of a day, or all its segments when segment is None."""

    day: str
    segment: str | None = None


@dataclass(frozen=True)
class Condition:
    """Named physical sides, without conflating reverse direction with a new condition.

    ``pairs`` is ``((side_a, side_b), ...)``, each side a tuple of Segment objects.
    E.g. Condition("cross", "cross-day",
    (((Segment("1"),), (Segment("2"),)),)).
    Same-day sides must be disjoint segments of the same day. Cross-day sides
    must have disjoint days. Every selected segment is required for eligibility.
    """

    name: str
    relation: str
    pairs: tuple[tuple[tuple[Segment, ...], tuple[Segment, ...]], ...]


@dataclass(frozen=True)
class Population:
    """Optional common population for comparisons of transmission alternatives.

    ``users``: sorted or unsorted canonical user labels.
    ``folds``: complete user -> integer fold assignment, or None to derive it
    from the eligible users and fold_seed. Folds must contain at least two users.
    ``masks``: optional model-seed -> row inclusion boolean vector. Masks are
    applied before windowing; original row indices enter the population hash.
    To compare alternatives, supply the same row layout, masks, users and folds.
    A declared user missing from an input is excluded, never silently removed
    from the eligibility denominator.
    """

    users: Sequence[str]
    folds: Mapping[str, int] | None = None
    masks: Mapping[str, Sequence[bool]] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditConfig:
    """Deterministic settings; defaults implement §3 of docs/plan_v3.md.

    Reduced inference draws are for explicitly labelled synthetic/CI runs, not
    equivalent to the full procedure. Feature comparators may be restricted only
    through an explicitly recorded configuration (e.g. the plan's runtime gate).
    ``cues`` declares optional cue levels, including levels absent from all rows.
    ``cohort`` is builder-supplied metadata, not inferred from a filename.
    ``day_order`` lists every observed day chronologically. Defaults infer only
    numeric or ISO-date labels; other labels require this order when conditions
    are derived automatically. Explicit Condition pairs retain their own order.
    """

    k_grid: tuple[int, ...] = (1, 5, 10, 20, 50, 100)
    fold_seed: int = 20261111
    seed: int = 20261112
    permutations: int = 10_000
    bootstraps: int = 2_000
    threads: int = 4
    cohort: str = "unspecified"
    comparators: tuple[str, ...] = (
        "cosine", "lda_cosine", "pair_logistic", "cosine_rich", "lda_cosine_rich")
    gallery: bool = True
    sequential: bool = True
    closed_set: bool = True
    seconds_per_output: float | None = None
    cues: tuple[str, ...] | None = None
    day_order: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible configuration content for hashing/provenance."""
        return asdict(self)


def audit_stream(
    inputs: str | Path | Mapping[str, str | Path],
    *,
    conditions: Sequence[Condition] | None = None,
    config: AuditConfig | None = None,
    population: Population | None = None,
    kind: str | None = None,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Audit one transmitted stream, optionally grouped across model seeds.

    ``inputs`` is one NPZ/CSV path or model-seed -> path. All grouped inputs must
    share users, days, segment schema, command alphabet and cue levels. Attacks
    are fitted separately per model seed; per-user statistics are averaged over
    seeds, and inference uses ONE physical-side matching for all seeds.
    ``conditions=None`` selects adjacent disjoint segments within every day and
    first day against each later day; explicit conditions are recommended for
    registered cohort runs. See Condition for unambiguous side definitions.
    ``config`` supplies k, seeds, thread limit and inference budgets.
    ``population`` supplies shared users, row masks and optional explicit folds.
    ``kind`` is "commands" or "features"; otherwise infer from array dimensions.
    ``out`` writes report.json, report.md and PNG figures; None only returns data.

    Returns a JSON-serialisable report with source/config/input hashes and a
    population hash per cell. Every cell is ``ok`` (including excluded_users),
    ``not_evaluable`` (with actionable reason), or ``failed`` (with error).
    Missing same-day/cross-day coverage is represented explicitly in
    ``unavailable_analyses`` without adding jobs to the requested ``cells``.
    Cells record canonical ``resolved_pairs`` for comparison compatibility.
    Invalid input raises ValueError naming the field and corrective action.
    Computation failures are recorded, not converted into non-detections.
    No stored BEAT artifacts, attacker scores or model weights are accepted.
    """
    from .engine import run_audit

    return run_audit(
        inputs, conditions=conditions, config=config or AuditConfig(),
        population=population, kind=kind, out=out,
    )


def audit_timing(
    inputs: str | Path | Mapping[str, str | Path],
    *,
    folds: Sequence[int] = (0,),
    conditions: Sequence[Condition] | None = None,
    config: AuditConfig | None = None,
    population: Population | None = None,
    kind: str | None = None,
) -> dict[str, Any]:
    """Time selected target folds with the full eligible development population.

    ``folds`` contains zero-based positions in the sorted fold-ID order, e.g.
    ``(0,)`` for a one-fold pilot. Original fold indices and attack seeds are
    retained. Restrict k/attacks through AuditConfig and conditions through the
    same public selectors used by audit_stream. Fitting, scoring, calibration
    and inference execute normally for selected folds; nothing is extrapolated.

    The returned report has ``purpose="timing_only"``, ``timing_folds``, overall
    ``runtime_seconds``/``peak_rss_bytes``, and ``timing_seconds`` per comparator.
    ``partner_auc`` names the actually scored targets. Population metadata still
    describes all eligible users required for development and original folds.
    Partial-fold statistics are diagnostics, NOT an audit verdict; compare
    refuses these reports. This entry never writes a publishable report.
    """
    from .engine import run_audit

    return run_audit(
        inputs, conditions=conditions, config=config or AuditConfig(),
        population=population, kind=kind, out=None, timing_folds=tuple(folds),
    )
