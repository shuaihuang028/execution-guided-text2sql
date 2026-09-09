from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from text2sql.execution.result_match import compare_rows, query_has_order_by
from text2sql.execution.sandbox import ExecutionStatus, SQLiteSandbox


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    example_id: str
    db_id: str
    predicted_sql: str
    gold_sql: str
    execution_correct: bool
    prediction_status: str
    gold_status: str
    ordered_comparison: bool
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    total: int
    scored: int
    execution_correct: int
    executable: int
    execution_accuracy: float
    scored_execution_accuracy: float
    executable_rate: float
    prediction_status_counts: dict[str, int]
    gold_error_count: int

    @classmethod
    def from_records(cls, records: Iterable[EvaluationRecord]) -> EvaluationSummary:
        items = list(records)
        scored = [record for record in items if record.gold_status == ExecutionStatus.SUCCESS]
        correct = sum(record.execution_correct for record in scored)
        executable = sum(
            record.prediction_status == ExecutionStatus.SUCCESS for record in scored
        )
        return cls(
            total=len(items),
            scored=len(scored),
            execution_correct=correct,
            executable=executable,
            execution_accuracy=correct / len(items) if items else 0.0,
            scored_execution_accuracy=correct / len(scored) if scored else 0.0,
            executable_rate=executable / len(scored) if scored else 0.0,
            prediction_status_counts=dict(Counter(record.prediction_status for record in items)),
            gold_error_count=len(items) - len(scored),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_prediction(
    *,
    example_id: str,
    db_id: str,
    database_path: str,
    gold_sql: str,
    predicted_sql: str,
    sandbox: SQLiteSandbox,
    gold_sandbox: SQLiteSandbox | None = None,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-8,
) -> EvaluationRecord:
    gold = (gold_sandbox or sandbox).execute(database_path, gold_sql)
    prediction = sandbox.execute(database_path, predicted_sql)
    ordered = query_has_order_by(gold_sql)

    if not gold.succeeded:
        return EvaluationRecord(
            example_id,
            db_id,
            predicted_sql,
            gold_sql,
            False,
            prediction.status,
            gold.status,
            ordered,
            f"gold_execution_error: {gold.error}",
        )
    if not prediction.succeeded:
        return EvaluationRecord(
            example_id,
            db_id,
            predicted_sql,
            gold_sql,
            False,
            prediction.status,
            gold.status,
            ordered,
            prediction.error,
        )

    comparison = compare_rows(
        gold.rows,
        prediction.rows,
        ordered=ordered,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )
    return EvaluationRecord(
        example_id,
        db_id,
        predicted_sql,
        gold_sql,
        comparison.matches,
        prediction.status,
        gold.status,
        ordered,
        comparison.reason,
    )
