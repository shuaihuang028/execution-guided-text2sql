from __future__ import annotations

import argparse
import json
from pathlib import Path

from text2sql.data.bird import load_bird_examples
from text2sql.evaluation.generate import PromptRepository, load_manifest_ids
from text2sql.training.train import SFTConfig, run_sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a resumable completion-only QLoRA adapter")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--column-meanings", type=Path)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=6144)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--validation-limit", type=int, default=128)
    parser.add_argument("--max-tables", type=int, default=6)
    parser.add_argument("--max-description-chars", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_bird_examples(args.data_path)
    train_ids = load_manifest_ids(args.train_manifest) or set()
    validation_ids = load_manifest_ids(args.validation_manifest) or set()
    train_examples = [example for example in examples if example.example_id in train_ids]
    validation_examples = [
        example for example in examples if example.example_id in validation_ids
    ]
    prompts = PromptRepository(
        args.database_root,
        args.column_meanings,
        max_tables=args.max_tables,
        max_description_chars=args.max_description_chars,
    )
    config = SFTConfig(
        model_path=str(args.model_path),
        output_dir=str(args.output_dir),
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        validation_limit=args.validation_limit,
        seed=args.seed,
    )
    metrics = run_sft(
        config=config,
        train_examples=train_examples,
        validation_examples=validation_examples,
        prompts=prompts,
        metrics_output=args.metrics_output,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
