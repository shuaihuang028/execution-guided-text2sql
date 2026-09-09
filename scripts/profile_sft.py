from __future__ import annotations

import argparse
import json
from pathlib import Path

from text2sql.data.bird import load_bird_examples
from text2sql.evaluation.generate import PromptRepository, load_manifest_ids
from text2sql.training.profile import ProfileConfig, run_sft_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile QLoRA SFT on one GPU")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--column-meanings", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--example-id", action="append")
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--step-log-output", type=Path, required=True)
    parser.add_argument("--adapter-output", type=Path)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=6144)
    parser.add_argument("--max-tables", type=int, default=6)
    parser.add_argument("--max-description-chars", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_ids = load_manifest_ids(args.manifest) or set()
    examples = [
        example
        for example in load_bird_examples(args.data_path)
        if example.example_id in manifest_ids
    ]
    if args.example_id:
        requested_ids = set(args.example_id)
        examples = [example for example in examples if example.example_id in requested_ids]
        missing_ids = requested_ids - {example.example_id for example in examples}
        if missing_ids:
            raise ValueError(f"Requested example IDs are not in the manifest: {sorted(missing_ids)}")
    prompts = PromptRepository(
        args.database_root,
        args.column_meanings,
        max_tables=args.max_tables,
        max_description_chars=args.max_description_chars,
    )
    config = ProfileConfig(
        model_path=str(args.model_path),
        max_seq_length=args.max_seq_length,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
    )
    metrics = run_sft_profile(
        config=config,
        examples=examples,
        prompts=prompts,
        metrics_output=args.metrics_output,
        step_log_output=args.step_log_output,
        adapter_output=args.adapter_output,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
