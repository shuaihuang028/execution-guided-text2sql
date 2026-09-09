from __future__ import annotations

import argparse
import json
from pathlib import Path

from text2sql.training.dpo import DPOTrainingConfig, run_dpo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DPO on execution-labeled pairs")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-seq-length", type=int, default=6144)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DPOTrainingConfig(
        adapter_path=str(args.adapter_path),
        output_dir=str(args.output_dir),
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        validation_fraction=args.validation_fraction,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
    )
    metrics = run_dpo(
        config=config,
        preferences_path=args.preferences,
        metrics_output=args.metrics_output,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

