from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def classify_evaluation(record: dict[str, Any]) -> str:
    if record.get("gold_status") != "success":
        return "gold_execution_error"
    if record.get("execution_correct"):
        return "correct"
    status = str(record.get("prediction_status", "unknown"))
    error = str(record.get("error", "")).casefold()
    if status == "timeout":
        return "timeout"
    if status == "too_many_rows":
        return "too_many_rows"
    if status == "success":
        if "row_count_mismatch" in error:
            return "wrong_row_set"
        if "column_count_mismatch" in error:
            return "wrong_projection"
        if "value_mismatch" in error:
            return "wrong_values"
        return "result_mismatch"
    if "no such column" in error:
        return "missing_column"
    if "no such table" in error:
        return "missing_table"
    if "ambiguous column" in error:
        return "ambiguous_column"
    if "no such function" in error:
        return "unsupported_function"
    if "misuse of aggregate" in error:
        return "aggregation_error"
    if any(token in error for token in ("syntax error", "incomplete input", "unrecognized token")):
        return "syntax_error"
    return "other_execution_error"


def compare_evaluations(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    difficulties: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    baseline_by_id = {str(item["example_id"]): item for item in baseline}
    candidate_by_id = {str(item["example_id"]): item for item in candidate}
    if baseline_by_id.keys() != candidate_by_id.keys():
        missing = sorted(baseline_by_id.keys() ^ candidate_by_id.keys())
        raise ValueError(f"Evaluation files have different example IDs: {missing[:5]}")

    categories: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    difficulty_counts: dict[str, Counter[str]] = defaultdict(Counter)
    difficulty_categories: dict[str, Counter[str]] = defaultdict(Counter)
    database_counts: dict[str, Counter[str]] = defaultdict(Counter)
    details: list[dict[str, Any]] = []
    for example_id, post in candidate_by_id.items():
        pre = baseline_by_id[example_id]
        pre_correct = bool(pre.get("execution_correct"))
        post_correct = bool(post.get("execution_correct"))
        transition = (
            "correct_to_correct"
            if pre_correct and post_correct
            else "correct_to_incorrect"
            if pre_correct
            else "incorrect_to_correct"
            if post_correct
            else "incorrect_to_incorrect"
        )
        category = classify_evaluation(post)
        difficulty = (difficulties or {}).get(example_id) or "unknown"
        db_id = str(post["db_id"])
        categories[category] += 1
        transitions[transition] += 1
        difficulty_counts[difficulty]["total"] += 1
        difficulty_counts[difficulty]["execution_correct"] += int(post_correct)
        difficulty_categories[difficulty][category] += 1
        database_counts[db_id]["total"] += 1
        database_counts[db_id]["pre_correct"] += int(pre_correct)
        database_counts[db_id]["post_correct"] += int(post_correct)
        details.append(
            {
                "example_id": example_id,
                "db_id": db_id,
                "difficulty": difficulty,
                "transition": transition,
                "post_sft_category": category,
                "post_sft_error": post.get("error"),
            }
        )

    by_database = []
    for db_id, counts in database_counts.items():
        by_database.append(
            {
                "db_id": db_id,
                **dict(counts),
                "correct_delta": counts["post_correct"] - counts["pre_correct"],
            }
        )
    by_database.sort(key=lambda item: (-item["correct_delta"], item["db_id"]))
    return {
        "total": len(candidate_by_id),
        "transitions": dict(transitions),
        "post_sft_categories": dict(categories.most_common()),
        "by_difficulty": {
            key: {**dict(value), "categories": dict(difficulty_categories[key].most_common())}
            for key, value in sorted(difficulty_counts.items())
        },
        "by_database": by_database,
        "details": details,
    }
