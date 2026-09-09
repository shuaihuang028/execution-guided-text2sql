from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from text2sql.data.bird import BirdExample, database_path
from text2sql.execution.sandbox import SQLiteSandbox


def official_set_match(
    predicted_rows: tuple[tuple[object, ...], ...],
    gold_rows: tuple[tuple[object, ...], ...],
) -> bool:
    """Reproduce BIRD's official EX row comparison: exact equality of Python sets."""

    return set(predicted_rows) == set(gold_rows)


def compare_with_official_semantics(
    *,
    examples: list[BirdExample],
    predictions: dict[str, str],
    local_records: dict[str, dict],
    database_root: str | Path,
    timeout_seconds: float = 30.0,
    max_rows: int = 1_000_000,
) -> tuple[list[dict], dict[str, object]]:
    sandbox = SQLiteSandbox(timeout_seconds, max_rows)
    details: list[dict] = []
    transitions: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    official_correct = 0

    for example in examples:
        prediction = sandbox.execute(
            database_path(database_root, example.db_id), predictions[example.example_id]
        )
        gold = sandbox.execute(database_path(database_root, example.db_id), example.sql)
        correct = (
            prediction.succeeded
            and gold.succeeded
            and official_set_match(prediction.rows, gold.rows)
        )
        local_correct = bool(local_records[example.example_id]["execution_correct"])
        transition = f"local_{int(local_correct)}_official_{int(correct)}"
        transitions[transition] += 1
        status_counts[str(prediction.status)] += 1
        official_correct += int(correct)
        details.append(
            {
                "example_id": example.example_id,
                "db_id": example.db_id,
                "local_correct": local_correct,
                "official_semantics_correct": correct,
                "prediction_status": str(prediction.status),
                "gold_status": str(gold.status),
                "transition": transition,
            }
        )

    total = len(examples)
    local_correct_count = sum(
        bool(local_records[example.example_id]["execution_correct"])
        for example in examples
    )
    summary: dict[str, object] = {
        "total": total,
        "local_correct": local_correct_count,
        "local_execution_accuracy": local_correct_count / total if total else 0.0,
        "official_semantics_correct": official_correct,
        "official_semantics_execution_accuracy": official_correct / total if total else 0.0,
        "accuracy_delta": (official_correct - local_correct_count) / total if total else 0.0,
        "agreement_rate": sum(
            count
            for transition, count in transitions.items()
            if transition in {"local_0_official_0", "local_1_official_1"}
        )
        / total
        if total
        else 0.0,
        "transitions": dict(transitions),
        "prediction_status_counts": dict(status_counts),
        "official_comparison_contract": "set(predicted_rows) == set(gold_rows)",
        "execution_note": (
            "Official row-set semantics reproduced inside this repository's read-only, "
            "row-limited sandbox; process-level timeout behavior is not byte-for-byte identical."
        ),
    }
    return details, summary


def load_jsonl_by_id(path: str | Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            example_id = str(record["example_id"])
            if example_id in records:
                raise ValueError(f"Duplicate example_id {example_id!r} at line {line_number}")
            records[example_id] = record
    return records
