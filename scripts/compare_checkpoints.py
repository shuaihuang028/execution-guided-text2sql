"""Paired McNemar comparisons between checkpoints scored on the same dev set.

Each `--evaluation` is a label=path pair pointing at an evaluate_predictions
details file. Each `--compare` names two labels as baseline:candidate. Examples
whose gold query failed to execute are excluded from every comparison, since
neither checkpoint can be scored on them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from text2sql.evaluation.significance import mcnemar_test


def _labelled(value: str, separator: str) -> tuple[str, str]:
    if separator not in value:
        raise argparse.ArgumentTypeError(f"Expected 'a{separator}b', got {value!r}")
    left, right = value.split(separator, 1)
    if not left or not right:
        raise argparse.ArgumentTypeError(f"Both sides of {separator!r} must be non-empty")
    return left, right


def _evaluation(value: str) -> tuple[str, Path]:
    label, path = _labelled(value, "=")
    return label, Path(path)


def _comparison(value: str) -> tuple[str, str]:
    return _labelled(value, ":")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=_evaluation, action="append", required=True)
    parser.add_argument("--compare", type=_comparison, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    evaluations = dict(args.evaluation)

    records: dict[str, dict[str, dict]] = {}
    for label, path in evaluations.items():
        records[label] = {str(item["example_id"]): item for item in _load(path)}

    # Only compare on examples every arm scored and whose gold actually ran.
    common = set.intersection(*(set(value) for value in records.values()))
    scored = {
        example_id
        for example_id in common
        if all(records[label][example_id]["gold_status"] == "success" for label in records)
    }
    correct = {
        label: {
            example_id
            for example_id in scored
            if records[label][example_id]["execution_correct"]
        }
        for label in records
    }

    comparisons = []
    for baseline, candidate in args.compare:
        for label in (baseline, candidate):
            if label not in correct:
                raise ValueError(f"Unknown label {label!r}; known: {sorted(correct)}")
        result = mcnemar_test(correct[baseline], correct[candidate], scored)
        comparisons.append(
            {"baseline": baseline, "candidate": candidate, **result.as_dict()}
        )

    summary = {
        "scored_examples": len(scored),
        "excluded_unscorable": len(common) - len(scored),
        "arms": {
            label: {
                "evaluation": str(evaluations[label]),
                "correct": len(correct[label]),
                "accuracy": len(correct[label]) / len(scored) if scored else 0.0,
            }
            for label in sorted(correct)
        },
        "test": "McNemar exact binomial, two-sided",
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
