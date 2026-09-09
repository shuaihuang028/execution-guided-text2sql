"""Confidence stratification for execution-labeled preference data.

Execution equivalence on a single database instance is weaker than semantic
equivalence. Two SQL statements can agree on the stored rows and still encode
different logic, and the risk is not uniform across examples: it is governed by
how discriminative the gold result set is.

An empty gold result is matched by every query that also returns nothing, so an
"execution-correct" label there carries almost no evidence. A single scalar
(one row, one column) is the next weakest case, because an unrelated aggregate
can coincide on one number. Wide, multi-row results are far harder to hit by
accident.

This module derives that signature from the gold execution and maps it to a
confidence tier, so preference data can be filtered and, more importantly, so
the residual label noise can be quantified and reported rather than assumed
away.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from numbers import Real

from text2sql.execution.sandbox import ExecutionResult

Row = tuple[object, ...]


class ConfidenceTier(StrEnum):
    """Trust level of an execution-correct label, ordered most to least reliable."""

    HIGH = "high"
    MEDIUM_TEXT_SCALAR = "medium_text_scalar"
    LOW_NUMERIC_SCALAR = "low_numeric_scalar"
    LOW_EMPTY_RESULT = "low_empty_result"
    UNVERIFIABLE_GOLD = "unverifiable_gold"


@dataclass(frozen=True, slots=True)
class ResultSignature:
    """Discriminative shape of a gold result set."""

    gold_status: str
    row_count: int
    column_count: int
    distinct_row_count: int
    scalar_kind: str | None = None
    """For a 1x1 result: 'numeric', 'text', or 'null'; otherwise None."""

    @property
    def gold_succeeded(self) -> bool:
        return self.gold_status == "success"

    @property
    def is_empty(self) -> bool:
        return self.gold_succeeded and self.row_count == 0

    @property
    def is_single_scalar(self) -> bool:
        return self.gold_succeeded and self.row_count == 1 and self.column_count == 1

    @property
    def has_duplicate_rows(self) -> bool:
        return self.distinct_row_count < self.row_count

    def as_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "is_empty": self.is_empty,
            "is_single_scalar": self.is_single_scalar,
            "has_duplicate_rows": self.has_duplicate_rows,
        }


def _distinct_count(rows: Sequence[Row]) -> int:
    try:
        return len(set(rows))
    except TypeError:
        # Unhashable cell values (rare); fall back to a quadratic scan.
        unique: list[Row] = []
        for row in rows:
            if row not in unique:
                unique.append(row)
        return len(unique)


def _scalar_kind(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, Real):  # bool is an Integral, so it lands here too
        return "numeric"
    return "text"


def result_signature(gold: ExecutionResult) -> ResultSignature:
    """Summarize the gold result shape that an execution label is evidence from."""

    if not gold.succeeded:
        return ResultSignature(str(gold.status), 0, 0, 0)
    rows = gold.rows
    column_count = len(gold.columns) if gold.columns else (len(rows[0]) if rows else 0)
    scalar_kind = (
        _scalar_kind(rows[0][0]) if len(rows) == 1 and column_count == 1 else None
    )
    return ResultSignature(
        gold_status=str(gold.status),
        row_count=len(rows),
        column_count=column_count,
        distinct_row_count=_distinct_count(rows),
        scalar_kind=scalar_kind,
    )


def confidence_tier(signature: ResultSignature) -> ConfidenceTier:
    """Map a gold result signature to the trust level of its execution labels.

    A single numeric scalar is the weakest non-empty evidence: an unrelated
    aggregate over the same table can land on the same number, and the space of
    plausible counts is small. A single text scalar (a name, a code, a date) is
    far harder to hit by accident, so it is separated out rather than lumped in.
    """

    if not signature.gold_succeeded:
        return ConfidenceTier.UNVERIFIABLE_GOLD
    if signature.is_empty:
        return ConfidenceTier.LOW_EMPTY_RESULT
    if signature.is_single_scalar:
        if signature.scalar_kind == "text":
            return ConfidenceTier.MEDIUM_TEXT_SCALAR
        return ConfidenceTier.LOW_NUMERIC_SCALAR
    return ConfidenceTier.HIGH


def is_high_confidence(signature: ResultSignature) -> bool:
    return confidence_tier(signature) is ConfidenceTier.HIGH
