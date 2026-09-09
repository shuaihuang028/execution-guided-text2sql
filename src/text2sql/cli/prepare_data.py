from __future__ import annotations

import argparse
import json
from pathlib import Path

from text2sql.data.bird import load_bird_examples
from text2sql.data.split import split_by_database


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a database-grouped train/validation manifest")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, default=Path("data/processed"))
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _write_manifest(path: Path, examples) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(
                json.dumps(
                    {"example_id": example.example_id, "db_id": example.db_id},
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    examples = load_bird_examples(args.data_path)
    split = split_by_database(examples, args.validation_fraction, args.seed)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    _write_manifest(args.output_directory / "train_manifest.jsonl", split.train)
    _write_manifest(args.output_directory / "validation_manifest.jsonl", split.validation)
    metadata = {
        "seed": args.seed,
        "validation_fraction_requested": args.validation_fraction,
        "train_examples": len(split.train),
        "validation_examples": len(split.validation),
        "train_db_ids": split.train_db_ids,
        "validation_db_ids": split.validation_db_ids,
        "database_overlap": sorted(set(split.train_db_ids) & set(split.validation_db_ids)),
    }
    (args.output_directory / "split_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
