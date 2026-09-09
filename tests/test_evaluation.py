from text2sql.evaluation.execution_accuracy import EvaluationSummary, evaluate_prediction
from text2sql.evaluation.output_parser import extract_sql
from text2sql.execution.sandbox import SQLiteSandbox


def test_extract_sql_from_fence():
    assert extract_sql("Here:\n```sql\nSELECT 1;\n```") == "SELECT 1;"
    assert extract_sql("SQL: SELECT 2") == "SELECT 2"


def test_evaluate_prediction_uses_gold_ordering(sample_database):
    sandbox = SQLiteSandbox()
    unordered = evaluate_prediction(
        example_id="shop:1",
        db_id="shop",
        database_path=str(sample_database),
        gold_sql="SELECT name FROM customers",
        predicted_sql="SELECT name FROM customers ORDER BY id DESC",
        sandbox=sandbox,
    )
    ordered = evaluate_prediction(
        example_id="shop:2",
        db_id="shop",
        database_path=str(sample_database),
        gold_sql="SELECT name FROM customers ORDER BY id",
        predicted_sql="SELECT name FROM customers ORDER BY id DESC",
        sandbox=sandbox,
    )
    assert unordered.execution_correct
    assert not ordered.execution_correct
    summary = EvaluationSummary.from_records([unordered, ordered])
    assert summary.execution_accuracy == 0.5
    assert summary.scored_execution_accuracy == 0.5
    assert summary.executable_rate == 1.0
