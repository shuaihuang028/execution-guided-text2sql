from __future__ import annotations

import argparse
import json
from pathlib import Path

from text2sql.data.bird import load_bird_examples
from text2sql.evaluation.error_analysis import compare_evaluations


def _jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and post-SFT errors")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--details-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    difficulties = {
        item.example_id: item.difficulty for item in load_bird_examples(args.data_path)
    }
    result = compare_evaluations(
        _jsonl(args.baseline), _jsonl(args.candidate), difficulties
    )
    details = result.pop("details")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with args.details_output.open("w", encoding="utf-8") as handle:
        for item in details:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
