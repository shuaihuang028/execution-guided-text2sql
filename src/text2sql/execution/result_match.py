from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

Row = tuple[object, ...]


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    matches: bool
    ordered: bool
    reason: str | None = None


def _mask_literals_and_comments(sql: str) -> str:
    # Preserve character positions while hiding quoted strings, identifiers, and comments.
    patterns = re.compile(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|--[^\n]*|/\*.*?\*/", re.DOTALL)
    return patterns.sub(lambda match: " " * len(match.group(0)), sql)


def query_has_order_by(sql: str) -> bool:
    return re.search(r"\border\s+by\b", _mask_literals_and_comments(sql), re.IGNORECASE) is not None


def _values_equal(left: object, right: object, rel_tol: float, abs_tol: float) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, Real) and isinstance(right, Real):
        return math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol)
    if isinstance(left, bytes) and isinstance(right, str):
        try:
            return left.decode("utf-8") == right
        except UnicodeDecodeError:
            return False
    if isinstance(left, str) and isinstance(right, bytes):
        return _values_equal(right, left, rel_tol, abs_tol)
    return left == right


def _rows_equal(left: Row, right: Row, rel_tol: float, abs_tol: float) -> bool:
    return len(left) == len(right) and all(
        _values_equal(a, b, rel_tol, abs_tol) for a, b in zip(left, right)
    )


def _unordered_equal(left: Sequence[Row], right: Sequence[Row], rel_tol: float, abs_tol: float) -> bool:
    if len(left) != len(right):
        return False
    # Fast path preserves duplicate rows exactly.
    try:
        if Counter(left) == Counter(right):
            return True
    except TypeError:
        pass

    unmatched = list(right)
    for left_row in left:
        for index, right_row in enumerate(unmatched):
            if _rows_equal(left_row, right_row, rel_tol, abs_tol):
                unmatched.pop(index)
                break
        else:
            return False
    return not unmatched


def compare_rows(
    expected: Sequence[Row],
    actual: Sequence[Row],
    *,
    ordered: bool,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-8,
) -> ComparisonResult:
    if len(expected) != len(actual):
        return ComparisonResult(False, ordered, "row_count_mismatch")
    if any(len(left) != len(right) for left, right in zip(expected, actual)):
        return ComparisonResult(False, ordered, "column_count_mismatch")

    if ordered:
        matches = all(
            _rows_equal(left, right, rel_tol, abs_tol) for left, right in zip(expected, actual)
        )
    else:
        matches = _unordered_equal(expected, actual, rel_tol, abs_tol)
    return ComparisonResult(matches, ordered, None if matches else "value_mismatch")

