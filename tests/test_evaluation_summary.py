from text2sql.evaluation.execution_accuracy import EvaluationRecord, EvaluationSummary


def test_gold_failures_remain_in_primary_accuracy_denominator():
    records = [
        EvaluationRecord("a:1", "a", "SELECT 1", "SELECT 1", True, "success", "success", False),
        EvaluationRecord(
            "b:2", "b", "SELECT 1", "SELECT 1", False, "success", "timeout", False
        ),
    ]
    summary = EvaluationSummary.from_records(records)
    assert summary.execution_accuracy == 0.5
    assert summary.scored_execution_accuracy == 1.0
    assert summary.gold_error_count == 1
