from text2sql.execution.sandbox import ExecutionResult, ExecutionStatus
from text2sql.preference.confidence import (
    ConfidenceTier,
    confidence_tier,
    is_high_confidence,
    result_signature,
)


def _ok(rows, columns):
    return ExecutionResult(status=ExecutionStatus.SUCCESS, rows=rows, columns=columns)


def test_empty_gold_result_is_low_confidence():
    signature = result_signature(_ok((), ("name",)))
    assert signature.is_empty
    assert confidence_tier(signature) is ConfidenceTier.LOW_EMPTY_RESULT
    assert not is_high_confidence(signature)


def test_single_numeric_scalar_is_lowest_non_empty_confidence():
    """A count can be matched by an unrelated aggregate landing on the same number."""

    signature = result_signature(_ok(((42,),), ("total",)))
    assert signature.is_single_scalar
    assert signature.scalar_kind == "numeric"
    assert confidence_tier(signature) is ConfidenceTier.LOW_NUMERIC_SCALAR


def test_single_text_scalar_is_medium_confidence():
    signature = result_signature(_ok((("Springfield High",),), ("school",)))
    assert signature.scalar_kind == "text"
    assert confidence_tier(signature) is ConfidenceTier.MEDIUM_TEXT_SCALAR


def test_single_null_scalar_is_treated_as_numeric_risk():
    signature = result_signature(_ok(((None,),), ("value",)))
    assert signature.scalar_kind == "null"
    assert confidence_tier(signature) is ConfidenceTier.LOW_NUMERIC_SCALAR


def test_boolean_scalar_counts_as_numeric():
    signature = result_signature(_ok(((True,),), ("flag",)))
    assert signature.scalar_kind == "numeric"


def test_scalar_kind_is_none_for_non_scalar_results():
    assert result_signature(_ok(((1,), (2,)), ("id",))).scalar_kind is None
    assert result_signature(_ok((), ("id",))).scalar_kind is None


def test_single_row_multi_column_is_high_confidence():
    signature = result_signature(_ok((("ann", 3),), ("name", "count")))
    assert not signature.is_single_scalar
    assert signature.scalar_kind is None
    assert confidence_tier(signature) is ConfidenceTier.HIGH


def test_multi_row_single_column_is_high_confidence():
    signature = result_signature(_ok(((1,), (2,), (3,)), ("id",)))
    assert confidence_tier(signature) is ConfidenceTier.HIGH
    assert signature.row_count == 3


def test_failed_gold_is_unverifiable():
    failed = ExecutionResult(status=ExecutionStatus.TIMEOUT, error="timed out")
    signature = result_signature(failed)
    assert not signature.gold_succeeded
    assert confidence_tier(signature) is ConfidenceTier.UNVERIFIABLE_GOLD


def test_duplicate_rows_are_detected():
    signature = result_signature(_ok(((1,), (1,), (2,)), ("id",)))
    assert signature.row_count == 3
    assert signature.distinct_row_count == 2
    assert signature.has_duplicate_rows


def test_column_count_falls_back_to_row_width():
    signature = result_signature(_ok((("a", "b"),), ()))
    assert signature.column_count == 2


def test_unhashable_cells_still_counted():
    signature = result_signature(_ok((([1],), ([1],)), ("blob",)))
    assert signature.row_count == 2
    assert signature.distinct_row_count == 1


def test_signature_serializes_derived_flags():
    payload = result_signature(_ok((), ("x",))).as_dict()
    assert payload["is_empty"] is True
    assert payload["row_count"] == 0
    assert payload["gold_status"] == "success"
