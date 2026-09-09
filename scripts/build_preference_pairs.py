from __future__ import annotations

import argparse
import json
from pathlib import Path

from text2sql.data.bird import load_bird_examples
from text2sql.evaluation.generate import PromptRepository, load_manifest_ids
from text2sql.preference.build_pairs import build_preference_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build execution-labeled DPO pairs")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--column-meanings", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--pairs-output", type=Path, required=True)
    parser.add_argument("--labels-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--max-tables", type=int, default=6)
    parser.add_argument("--max-description-chars", type=int, default=96)
    parser.add_argument("--timeout-seconds", type=float, default=3.0)
    parser.add_argument("--gold-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--gold-max-rows", type=int, default=1_000_000)
    parser.add_argument("--allow-gold-fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_ids = load_manifest_ids(args.manifest) or set()
    examples = [
        example
        for example in load_bird_examples(args.data_path)
        if example.example_id in manifest_ids
    ]
    prompts = PromptRepository(
        args.database_root,
        args.column_meanings,
        args.max_tables,
        args.max_description_chars,
    )
    metrics = build_preference_pairs(
        examples=examples,
        candidates_path=args.candidates,
        database_root=args.database_root,
        prompts=prompts,
        pairs_output=args.pairs_output,
        labels_output=args.labels_output,
        timeout_seconds=args.timeout_seconds,
        gold_timeout_seconds=args.gold_timeout_seconds,
        max_rows=args.max_rows,
        gold_max_rows=args.gold_max_rows,
        allow_gold_fallback=args.allow_gold_fallback,
    )
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

