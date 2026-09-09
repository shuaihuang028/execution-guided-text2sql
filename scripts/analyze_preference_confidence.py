"""Quantify how much each execution-correct preference label can be trusted.

Re-executes gold SQL for every candidate-sampled training example, derives the
discriminative shape of the gold result, and reports the confidence tier
distribution across candidate examples, examples with a generated positive, and
the pairs that actually reached DPO. Also emits a high-confidence-only subset of
the preference pairs for ablation.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from text2sql.data.bird import database_path, load_bird_examples
from text2sql.execution.sandbox import SQLiteSandbox
from text2sql.preference.confidence import (
    ConfidenceTier,
    confidence_tier,
    result_signature,
)

_SANDBOX: SQLiteSandbox | None = None


def _init_worker(timeout_seconds: float, max_rows: int) -> None:
    global _SANDBOX
    _SANDBOX = SQLiteSandbox(timeout_seconds, max_rows)


def _signature_for(task: tuple[str, str, str]) -> tuple[str, dict[str, object]]:
    example_id, db_path, sql = task
    assert _SANDBOX is not None
    signature = result_signature(_SANDBOX.execute(db_path, sql))
    payload = signature.as_dict()
    payload["confidence_tier"] = str(confidence_tier(signature))
    return example_id, payload


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--signatures-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--high-confidence-pairs-output", type=Path)
    parser.add_argument("--gold-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--gold-max-rows", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = {item.example_id: item for item in load_bird_examples(args.data_path)}
    labels = _read_jsonl(args.labels)
    pairs = _read_jsonl(args.pairs)
    paired_ids = {str(record["example_id"]) for record in pairs}

    tasks = []
    for record in labels:
        example_id = str(record["example_id"])
        example = examples[example_id]
        tasks.append(
            (example_id, str(database_path(args.database_root, example.db_id)), example.sql)
        )

    signatures: dict[str, dict[str, object]] = {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.gold_timeout_seconds, args.gold_max_rows),
    ) as pool:
        for index, (example_id, payload) in enumerate(
            pool.map(_signature_for, tasks, chunksize=16), start=1
        ):
            signatures[example_id] = payload
            if index % 500 == 0:
                print(f"gold executed {index}/{len(tasks)}", flush=True)

    # Tier distributions over three progressively narrower populations.
    tier_all: Counter[str] = Counter()
    tier_with_positive: Counter[str] = Counter()
    tier_paired: Counter[str] = Counter()
    difficulty_paired: Counter[str] = Counter()
    difficulty_unpaired: Counter[str] = Counter()
    positives_by_tier: dict[str, int] = defaultdict(int)
    empty_positive_examples = 0

    for record in labels:
        example_id = str(record["example_id"])
        tier = str(signatures[example_id]["confidence_tier"])
        has_positive = any(item["execution_correct"] for item in record["candidates"])
        tier_all[tier] += 1
        if has_positive:
            tier_with_positive[tier] += 1
            positives_by_tier[tier] += sum(
                1 for item in record["candidates"] if item["execution_correct"]
            )
            if tier == str(ConfidenceTier.LOW_EMPTY_RESULT):
                empty_positive_examples += 1
        difficulty = examples[example_id].difficulty or "unlabeled"
        if example_id in paired_ids:
            tier_paired[tier] += 1
            difficulty_paired[difficulty] += 1
        else:
            difficulty_unpaired[difficulty] += 1

    high = str(ConfidenceTier.HIGH)
    text_scalar = str(ConfidenceTier.MEDIUM_TEXT_SCALAR)
    paired_total = sum(tier_paired.values())

    def pair_tier(record: dict) -> str:
        return str(signatures[str(record["example_id"])]["confidence_tier"])

    high_confidence_pairs = [record for record in pairs if pair_tier(record) == high]
    medium_or_better_pairs = [
        record for record in pairs if pair_tier(record) in {high, text_scalar}
    ]

    if args.high_confidence_pairs_output:
        args.high_confidence_pairs_output.parent.mkdir(parents=True, exist_ok=True)
        with args.high_confidence_pairs_output.open("w", encoding="utf-8") as handle:
            for record in high_confidence_pairs:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    args.signatures_output.parent.mkdir(parents=True, exist_ok=True)
    with args.signatures_output.open("w", encoding="utf-8") as handle:
        for example_id, payload in signatures.items():
            handle.write(
                json.dumps({"example_id": example_id, **payload}, ensure_ascii=False) + "\n"
            )

    metrics = {
        "candidate_examples": len(labels),
        "examples_with_generated_positive": sum(tier_with_positive.values()),
        "preference_pairs": len(pairs),
        "confidence_tier_all_candidate_examples": dict(tier_all),
        "confidence_tier_examples_with_positive": dict(tier_with_positive),
        "confidence_tier_preference_pairs": dict(tier_paired),
        "positive_candidates_by_tier": dict(positives_by_tier),
        "empty_result_positive_examples": empty_positive_examples,
        "high_confidence_pairs": len(high_confidence_pairs),
        "high_confidence_pair_fraction": (
            len(high_confidence_pairs) / paired_total if paired_total else 0.0
        ),
        "medium_or_better_pairs": len(medium_or_better_pairs),
        "medium_or_better_pair_fraction": (
            len(medium_or_better_pairs) / paired_total if paired_total else 0.0
        ),
        "low_confidence_pair_fraction": (
            (paired_total - len(high_confidence_pairs)) / paired_total if paired_total else 0.0
        ),
        "difficulty_paired": dict(difficulty_paired),
        "difficulty_unpaired": dict(difficulty_unpaired),
        "gold_timeout_seconds": args.gold_timeout_seconds,
        "gold_max_rows": args.gold_max_rows,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
