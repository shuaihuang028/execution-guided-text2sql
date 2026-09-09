from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from text2sql.data.bird import BirdExample, database_path
from text2sql.evaluation.generate import PromptRepository
from text2sql.execution.result_match import compare_rows, query_has_order_by
from text2sql.execution.sandbox import ExecutionStatus, SQLiteSandbox


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip().rstrip(";").casefold()


def deduplicate_candidates(candidates: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for candidate in candidates:
        key = normalize_sql(str(candidate["predicted_sql"]))
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def choose_pair(
    labeled_candidates: list[dict],
    *,
    gold_sql: str | None = None,
    allow_gold_fallback: bool = False,
) -> tuple[str, str, str, str] | None:
    positives = [candidate for candidate in labeled_candidates if candidate["execution_correct"]]
    negatives = [candidate for candidate in labeled_candidates if not candidate["execution_correct"]]
    if not negatives:
        return None
    if positives:
        chosen = min(
            positives,
            key=lambda item: (len(str(item["predicted_sql"])), item["candidate_index"]),
        )
        source = "generated"
    elif allow_gold_fallback and gold_sql:
        chosen = {"predicted_sql": gold_sql}
        source = "gold_fallback"
    else:
        return None

    hard = [item for item in negatives if item["prediction_status"] == ExecutionStatus.SUCCESS]
    pool = hard or negatives
    chosen_normalized = normalize_sql(str(chosen["predicted_sql"]))
    rejected = max(
        pool,
        key=lambda item: (
            SequenceMatcher(
                None, chosen_normalized, normalize_sql(str(item["predicted_sql"]))
            ).ratio(),
            -len(str(item["predicted_sql"])),
            -int(item["candidate_index"]),
        ),
    )
    negative_type = "hard" if rejected["prediction_status"] == ExecutionStatus.SUCCESS else "easy"
    return (
        str(chosen["predicted_sql"]),
        str(rejected["predicted_sql"]),
        source,
        negative_type,
    )


def build_preference_pairs(
    *,
    examples: list[BirdExample],
    candidates_path: str | Path,
    database_root: str | Path,
    prompts: PromptRepository,
    pairs_output: str | Path,
    labels_output: str | Path,
    timeout_seconds: float = 3.0,
    gold_timeout_seconds: float = 30.0,
    max_rows: int = 100_000,
    gold_max_rows: int = 1_000_000,
    allow_gold_fallback: bool = False,
) -> dict[str, object]:
    by_id = {example.example_id: example for example in examples}
    with Path(candidates_path).open("r", encoding="utf-8") as handle:
        candidate_records = [json.loads(line) for line in handle if line.strip()]
    ids = [str(record["example_id"]) for record in candidate_records]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate file contains duplicate example IDs")
    unexpected = set(ids) - by_id.keys()
    if unexpected:
        raise ValueError(f"Candidate file contains {len(unexpected)} out-of-split examples")

    sandbox = SQLiteSandbox(timeout_seconds, max_rows)
    gold_sandbox = SQLiteSandbox(gold_timeout_seconds, gold_max_rows)
    pairs_path = Path(pairs_output)
    labels_path = Path(labels_output)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()

    with pairs_path.open("w", encoding="utf-8") as pair_handle, labels_path.open(
        "w", encoding="utf-8"
    ) as label_handle:
        for candidate_record in candidate_records:
            example = by_id[str(candidate_record["example_id"])]
            db_path = database_path(database_root, example.db_id)
            gold = gold_sandbox.execute(db_path, example.sql)
            ordered = query_has_order_by(example.sql)
            candidates = deduplicate_candidates(candidate_record["candidates"])
            counts["candidate_examples"] += 1
            counts["raw_candidates"] += len(candidate_record["candidates"])
            counts["unique_candidates"] += len(candidates)
            labeled = []
            for candidate in candidates:
                prediction = sandbox.execute(db_path, str(candidate["predicted_sql"]))
                correct = False
                comparison_reason = prediction.error
                if gold.succeeded and prediction.succeeded:
                    comparison = compare_rows(gold.rows, prediction.rows, ordered=ordered)
                    correct = comparison.matches
                    comparison_reason = comparison.reason
                status_counts[str(prediction.status)] += 1
                labeled.append(
                    {
                        **candidate,
                        "execution_correct": correct,
                        "prediction_status": str(prediction.status),
                        "execution_error": prediction.error,
                        "comparison_reason": comparison_reason,
                        "execution_seconds": prediction.elapsed_seconds,
                    }
                )
            if any(item["execution_correct"] for item in labeled):
                counts["examples_with_generated_positive"] += 1
            if any(not item["execution_correct"] for item in labeled):
                counts["examples_with_negative"] += 1
            if any(
                not item["execution_correct"]
                and item["prediction_status"] == ExecutionStatus.SUCCESS
                for item in labeled
            ):
                counts["examples_with_hard_negative"] += 1
            if not gold.succeeded:
                counts["gold_errors"] += 1

            label_handle.write(
                json.dumps(
                    {
                        "example_id": example.example_id,
                        "db_id": example.db_id,
                        "gold_status": str(gold.status),
                        "candidates": labeled,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if not gold.succeeded:
                continue
            pair = choose_pair(
                labeled,
                gold_sql=example.sql,
                allow_gold_fallback=allow_gold_fallback,
            )
            if pair is None:
                continue
            chosen, rejected, source, negative_type = pair
            counts["pairs"] += 1
            counts[f"pair_source_{source}"] += 1
            counts[f"negative_type_{negative_type}"] += 1
            pair_handle.write(
                json.dumps(
                    {
                        "example_id": example.example_id,
                        "db_id": example.db_id,
                        "prompt": prompts.messages(example),
                        "chosen": [{"role": "assistant", "content": chosen}],
                        "rejected": [{"role": "assistant", "content": rejected}],
                        "pair_source": source,
                        "negative_type": negative_type,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    candidate_examples = counts["candidate_examples"]
    raw_candidates = counts["raw_candidates"]
    return {
        **dict(counts),
        "candidate_status_counts": dict(status_counts),
        "generated_positive_coverage": counts["examples_with_generated_positive"]
        / candidate_examples
        if candidate_examples
        else 0.0,
        "generated_only_pair_coverage": counts["pair_source_generated"]
        / candidate_examples
        if candidate_examples
        else 0.0,
        "within_example_unique_rate": counts["unique_candidates"] / raw_candidates
        if raw_candidates
        else 0.0,
        "allow_gold_fallback": allow_gold_fallback,
    }

