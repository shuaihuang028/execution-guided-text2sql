from text2sql.evaluation.error_analysis import classify_evaluation, compare_evaluations


def test_classify_common_failures():
    assert classify_evaluation({"gold_status": "success", "execution_correct": True}) == "correct"
    assert (
        classify_evaluation(
            {
                "gold_status": "success",
                "execution_correct": False,
                "prediction_status": "error",
                "error": "no such column: t.fake",
            }
        )
        == "missing_column"
    )
    assert (
        classify_evaluation(
            {
                "gold_status": "success",
                "execution_correct": False,
                "prediction_status": "success",
                "error": "row_count_mismatch",
            }
        )
        == "wrong_row_set"
    )


def test_compare_evaluations_counts_regressions_and_fixes():
    baseline = [
        {"example_id": "db:1", "db_id": "db", "execution_correct": True},
        {"example_id": "db:2", "db_id": "db", "execution_correct": False},
    ]
    candidate = [
        {
            "example_id": "db:1",
            "db_id": "db",
            "execution_correct": False,
            "gold_status": "success",
            "prediction_status": "error",
            "error": "syntax error",
        },
        {
            "example_id": "db:2",
            "db_id": "db",
            "execution_correct": True,
            "gold_status": "success",
            "prediction_status": "success",
        },
    ]
    result = compare_evaluations(baseline, candidate, {"db:1": "hard", "db:2": "simple"})
    assert result["transitions"] == {
        "correct_to_incorrect": 1,
        "incorrect_to_correct": 1,
    }
    assert result["by_database"][0]["correct_delta"] == 0
    assert result["by_difficulty"]["simple"]["execution_correct"] == 1
    assert result["by_difficulty"]["simple"]["categories"] == {"correct": 1}
