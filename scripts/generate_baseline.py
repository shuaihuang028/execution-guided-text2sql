from __future__ import annotations

import argparse
import json
from pathlib import Path

from text2sql.data.bird import load_bird_examples
from text2sql.evaluation.generate import (
    PromptRepository,
    completed_prediction_ids,
    generate_predictions,
    load_manifest_ids,
    pending_examples,
    prediction_file_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic pre-SFT predictions")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--column-meanings", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-tables", type=int, default=8)
    parser.add_argument("--max-description-chars", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_bird_examples(args.data_path)
    selected = pending_examples(
        examples,
        load_manifest_ids(args.manifest),
        completed_prediction_ids(args.output),
        args.limit,
    )
    prompts = PromptRepository(
        args.database_root,
        args.column_meanings,
        args.max_tables,
        args.max_description_chars,
    )
    metrics = generate_predictions(
        model_path=args.model_path,
        examples=selected,
        prompts=prompts,
        output_path=args.output,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    metrics.update(prediction_file_stats(args.output))
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
