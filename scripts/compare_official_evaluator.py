from __future__ import annotations

import argparse
import json
from pathlib import Path

from text2sql.data.bird import load_bird_examples
from text2sql.evaluation.generate import load_manifest_ids
from text2sql.evaluation.official_parity import (
    compare_with_official_semantics,
    load_jsonl_by_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare guarded local EX with BIRD set EX")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--local-details", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--details-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_bird_examples(args.data_path)
    manifest_ids = load_manifest_ids(args.manifest)
    if manifest_ids is not None:
        examples = [item for item in examples if item.example_id in manifest_ids]
    predictions_raw = load_jsonl_by_id(args.predictions)
    predictions = {
        example_id: str(record.get("predicted_sql", record.get("prediction", "")))
        for example_id, record in predictions_raw.items()
    }
    local_records = load_jsonl_by_id(args.local_details)
    expected = {example.example_id for example in examples}
    if predictions.keys() != expected or local_records.keys() != expected:
        raise ValueError("Predictions/local details do not exactly cover the selected examples")
    details, summary = compare_with_official_semantics(
        examples=examples,
        predictions=predictions,
        local_records=local_records,
        database_root=args.database_root,
        timeout_seconds=args.timeout_seconds,
        max_rows=args.max_rows,
    )
    args.details_output.parent.mkdir(parents=True, exist_ok=True)
    with args.details_output.open("w", encoding="utf-8") as handle:
        for record in details:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

