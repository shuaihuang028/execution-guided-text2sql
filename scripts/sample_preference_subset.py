"""Draw a deterministic, size-matched random subset of preference pairs.

Used as the control arm for the confidence-filtering ablation: same pair count
as the filtered subset, original quality mix retained.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from text2sql.preference.subsets import sample_preference_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--signatures",
        type=Path,
        help="Optional confidence signatures, to report the sampled tier mix",
    )
    parser.add_argument("--metrics-output", type=Path)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    records = _read_jsonl(args.pairs)
    subset = sample_preference_subset(records, args.size, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in subset:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics: dict[str, object] = {
        "source_pairs": len(records),
        "sampled_pairs": len(subset),
        "seed": args.seed,
        "output": str(args.output),
    }
    if args.signatures:
        tiers = {
            str(item["example_id"]): str(item["confidence_tier"])
            for item in _read_jsonl(args.signatures)
        }
        metrics["source_tier_mix"] = dict(
            Counter(tiers[str(item["example_id"])] for item in records)
        )
        metrics["sampled_tier_mix"] = dict(
            Counter(tiers[str(item["example_id"])] for item in subset)
        )
    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
