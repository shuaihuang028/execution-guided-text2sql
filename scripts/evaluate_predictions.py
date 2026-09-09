from __future__ import annotations

import argparse
import json
from pathlib import Path

from text2sql.data.bird import database_path, load_bird_examples
from text2sql.evaluation.execution_accuracy import EvaluationSummary, evaluate_prediction
from text2sql.evaluation.generate import load_manifest_ids
from text2sql.evaluation.output_parser import extract_sql
from text2sql.execution.sandbox import SQLiteSandbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run guarded local execution evaluation")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--details-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3.0)
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--gold-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--gold-max-rows", type=int, default=1_000_000)
    return parser.parse_args()


def load_predictions(path: Path) -> dict[str, str]:
    predictions: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            example_id = str(record["example_id"])
            if example_id in predictions:
                raise ValueError(f"Duplicate example_id {example_id!r} at line {line_number}")
            generation = record.get("predicted_sql", record.get("prediction"))
            if generation is None:
                raise ValueError(f"Missing predicted_sql at line {line_number}")
            predictions[example_id] = extract_sql(str(generation))
    return predictions


def main() -> None:
    args = parse_args()
    examples = load_bird_examples(args.data_path)
    manifest_ids = load_manifest_ids(args.manifest)
    if manifest_ids is not None:
        examples = [example for example in examples if example.example_id in manifest_ids]
    if args.limit is not None:
        examples = examples[: args.limit]
    predictions = load_predictions(args.predictions)
    expected_ids = {example.example_id for example in examples}
    missing = expected_ids - predictions.keys()
    unexpected = predictions.keys() - expected_ids
    if missing or unexpected:
        raise ValueError(
            f"Prediction coverage mismatch: {len(missing)} missing, {len(unexpected)} unexpected"
        )

    sandbox = SQLiteSandbox(args.timeout_seconds, args.max_rows)
    gold_sandbox = SQLiteSandbox(args.gold_timeout_seconds, args.gold_max_rows)
    records = [
        evaluate_prediction(
            example_id=example.example_id,
            db_id=example.db_id,
            database_path=str(database_path(args.database_root, example.db_id)),
            gold_sql=example.sql,
            predicted_sql=predictions[example.example_id],
            sandbox=sandbox,
            gold_sandbox=gold_sandbox,
        )
        for example in examples
    ]
    summary = EvaluationSummary.from_records(records)

    args.details_output.parent.mkdir(parents=True, exist_ok=True)
    with args.details_output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
