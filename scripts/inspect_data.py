from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from text2sql.data.bird import database_path, iter_bird_examples
from text2sql.data.prompt_builder import PromptBuilder
from text2sql.data.schema import (
    attach_descriptions,
    extract_sqlite_schema,
    load_all_column_meanings_json,
    load_database_descriptions,
)
from text2sql.data.schema_pruning import prune_schema


def percentile(sorted_values: list[int], probability: float) -> int:
    if not sorted_values:
        raise ValueError("Cannot compute a percentile for an empty sequence")
    index = round((len(sorted_values) - 1) * probability)
    return sorted_values[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect BIRD examples and exact prompt lengths")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--column-meanings", type=Path)
    parser.add_argument("--max-tables", type=int)
    parser.add_argument("--max-description-chars", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install the data extra: python -m pip install -e '.[data]'") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    builder = PromptBuilder(max_description_chars=args.max_description_chars)
    compact_descriptions = (
        load_all_column_meanings_json(args.column_meanings) if args.column_meanings else {}
    )
    schema_cache = {}
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    sequence_lengths: list[int] = []
    longest: list[tuple[int, str, str]] = []
    database_counts: Counter[str] = Counter()

    for index, example in enumerate(iter_bird_examples(args.data_path)):
        if args.limit is not None and index >= args.limit:
            break
        if example.db_id not in schema_cache:
            path = database_path(args.database_root, example.db_id)
            schema = extract_sqlite_schema(path, example.db_id)
            descriptions = load_database_descriptions(path.parent)
            for table_name, columns in compact_descriptions.get(
                example.db_id.casefold(), {}
            ).items():
                descriptions.setdefault(table_name, {}).update(columns)
            schema_cache[example.db_id] = attach_descriptions(schema, descriptions)
        schema = schema_cache[example.db_id]
        if args.max_tables is not None:
            schema = prune_schema(schema, example.question, example.evidence, args.max_tables)
        messages = builder.build_messages(example, schema)
        prompt_token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        completion_token_ids = tokenizer.encode(example.sql, add_special_tokens=False)
        sequence_token_ids = tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": example.sql}],
            tokenize=True,
            add_generation_prompt=False,
        )
        prompt_lengths.append(len(prompt_token_ids))
        completion_lengths.append(len(completion_token_ids))
        sequence_lengths.append(len(sequence_token_ids))
        longest.append((len(sequence_token_ids), example.example_id, example.db_id))
        database_counts[example.db_id] += 1

    prompt_lengths.sort()
    completion_lengths.sort()
    sequence_lengths.sort()
    longest.sort(reverse=True)

    def stats(values: list[int]) -> dict[str, int]:
        return {
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values),
        }

    report = {
        "examples": len(sequence_lengths),
        "databases": len(database_counts),
        "model": args.model,
        "max_tables": args.max_tables,
        "max_description_chars": args.max_description_chars,
        "prompt_tokens": stats(prompt_lengths),
        "completion_tokens": stats(completion_lengths),
        "training_sequence_tokens": stats(sequence_lengths),
        "longest_sequences": [
            {"tokens": tokens, "example_id": example_id, "db_id": db_id}
            for tokens, example_id, db_id in longest[:10]
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
