from __future__ import annotations

import argparse
import json
from pathlib import Path

from text2sql.data.bird import load_bird_examples
from text2sql.evaluation.generate import PromptRepository, load_manifest_ids, pending_examples
from text2sql.preference.generate_candidates import (
    candidate_file_stats,
    completed_candidate_ids,
    generate_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample resumable train-only SQL candidates")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--column-meanings", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--candidates-per-example", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batch-input-tokens", type=int, default=10_000)
    parser.add_argument(
        "--vectorized-candidate-max-prompt-tokens", type=int, default=3_000
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-tables", type=int, default=6)
    parser.add_argument("--max-description-chars", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = pending_examples(
        load_bird_examples(args.data_path),
        load_manifest_ids(args.manifest),
        completed_candidate_ids(args.output),
        args.limit,
    )
    prompts = PromptRepository(
        args.database_root,
        args.column_meanings,
        args.max_tables,
        args.max_description_chars,
    )
    metrics = generate_candidates(
        model_path=args.model_path,
        examples=selected,
        prompts=prompts,
        output_path=args.output,
        batch_size=args.batch_size,
        max_batch_input_tokens=args.max_batch_input_tokens,
        vectorized_candidate_max_prompt_tokens=(
            args.vectorized_candidate_max_prompt_tokens
        ),
        candidates_per_example=args.candidates_per_example,
        temperature=args.temperature,
        top_p=args.top_p,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    metrics.update(candidate_file_stats(args.output))
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
