from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from text2sql.training.dpo import DPOTrainingConfig, run_dpo, select_best_sweep_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a resumable LR/beta DPO smoke sweep")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[5e-6, 1e-5, 2e-5])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.05, 0.1])
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=6144)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.removeprefix("checkpoint-")), path))
        except ValueError:
            continue
    return max(checkpoints)[1] if checkpoints else None


def main() -> None:
    import torch

    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    runs = []
    for learning_rate in args.learning_rates:
        for beta in args.betas:
            name = f"lr_{learning_rate:.0e}_beta_{beta:g}".replace("+", "")
            output_dir = args.output_root / name
            metrics_path = output_dir / "metrics.json"
            if metrics_path.is_file():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                metrics = run_dpo(
                    config=DPOTrainingConfig(
                        adapter_path=str(args.adapter_path),
                        output_dir=str(output_dir),
                        max_seq_length=args.max_seq_length,
                        max_steps=args.max_steps,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                        learning_rate=learning_rate,
                        beta=beta,
                        validation_fraction=args.validation_fraction,
                        save_steps=args.max_steps,
                        seed=args.seed,
                    ),
                    preferences_path=args.preferences,
                    metrics_output=metrics_path,
                    resume_from_checkpoint=latest_checkpoint(output_dir),
                )
            runs.append(metrics)
            gc.collect()
            torch.cuda.empty_cache()

    best = select_best_sweep_run(runs)
    summary = {
        "selection_metric": "max eval_rewards/accuracies, tie-break min eval_loss",
        "runs": [
            {
                "learning_rate": run["config"]["learning_rate"],
                "beta": run["config"]["beta"],
                "output_dir": run["config"]["output_dir"],
                "peak_vram_mib": run["peak_vram_mib"],
                **run["validation_metrics"],
            }
            for run in runs
        ],
        "best": {
            "learning_rate": best["config"]["learning_rate"],
            "beta": best["config"]["beta"],
            "output_dir": best["config"]["output_dir"],
        },
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

