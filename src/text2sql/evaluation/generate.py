from __future__ import annotations

import json
import random
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

from text2sql.data.bird import BirdExample, database_path
from text2sql.data.prompt_builder import PromptBuilder
from text2sql.data.schema import (
    attach_descriptions,
    extract_sqlite_schema,
    load_all_column_meanings_json,
    load_database_descriptions,
)
from text2sql.data.schema_pruning import prune_schema
from text2sql.evaluation.output_parser import extract_sql


def load_manifest_ids(path: str | Path | None) -> set[str] | None:
    if path is None:
        return None
    selected: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                selected.add(str(json.loads(line)["example_id"]))
    return selected


def pending_examples(
    examples: Iterable[BirdExample],
    manifest_ids: set[str] | None,
    completed_ids: set[str],
    limit: int | None,
) -> list[BirdExample]:
    pending = [
        example
        for example in examples
        if (manifest_ids is None or example.example_id in manifest_ids)
        and example.example_id not in completed_ids
    ]
    return pending[:limit] if limit is not None else pending


def completed_prediction_ids(path: str | Path) -> set[str]:
    output = Path(path)
    if not output.is_file():
        return set()
    with output.open("r", encoding="utf-8") as handle:
        return {
            str(json.loads(line)["example_id"])
            for line in handle
            if line.strip()
        }


def prediction_file_stats(path: str | Path) -> dict[str, int]:
    """Summarize the complete append-only prediction file across resumed segments."""

    with Path(path).open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    ids = [str(record["example_id"]) for record in records]
    return {
        "output_examples_total": len(records),
        "output_unique_ids": len(set(ids)),
        "output_duplicate_ids": len(ids) - len(set(ids)),
        "output_generated_tokens_total": sum(
            int(record.get("generated_tokens", 0)) for record in records
        ),
    }


def batched(items: Sequence[BirdExample], batch_size: int) -> Iterable[Sequence[BirdExample]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


class PromptRepository:
    def __init__(
        self,
        database_root: str | Path,
        column_meanings: str | Path | None,
        max_tables: int,
        max_description_chars: int,
    ) -> None:
        self.database_root = Path(database_root)
        self.max_tables = max_tables
        self.builder = PromptBuilder(max_description_chars=max_description_chars)
        self.compact_descriptions = (
            load_all_column_meanings_json(column_meanings) if column_meanings else {}
        )
        self.schema_cache = {}

    def messages(self, example: BirdExample) -> list[dict[str, str]]:
        if example.db_id not in self.schema_cache:
            path = database_path(self.database_root, example.db_id)
            schema = extract_sqlite_schema(path, example.db_id)
            descriptions = load_database_descriptions(path.parent)
            for table_name, columns in self.compact_descriptions.get(
                example.db_id.casefold(), {}
            ).items():
                descriptions.setdefault(table_name, {}).update(columns)
            self.schema_cache[example.db_id] = attach_descriptions(schema, descriptions)
        schema = prune_schema(
            self.schema_cache[example.db_id],
            example.question,
            example.evidence,
            self.max_tables,
        )
        return self.builder.build_messages(example, schema)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def generate_predictions(
    *,
    model_path: str | Path,
    examples: Sequence[BirdExample],
    prompts: PromptRepository,
    output_path: str | Path,
    batch_size: int = 1,
    max_input_tokens: int = 8192,
    max_new_tokens: int = 384,
    seed: int = 42,
) -> dict[str, float | int | str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    seed_everything(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    generated_tokens = 0
    started = time.perf_counter()
    peak_input_tokens = 0

    with output.open("a", encoding="utf-8") as handle, torch.inference_mode():
        for batch in batched(examples, batch_size):
            rendered = [
                tokenizer.apply_chat_template(
                    prompts.messages(example), tokenize=False, add_generation_prompt=True
                )
                for example in batch
            ]
            encoded = tokenizer(rendered, return_tensors="pt", padding=True, truncation=False)
            input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
            peak_input_tokens = max(peak_input_tokens, max(input_lengths))
            if peak_input_tokens > max_input_tokens:
                raise ValueError(
                    f"Input has {peak_input_tokens} tokens, exceeding limit {max_input_tokens}; "
                    "adjust schema pruning rather than truncating silently"
                )
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            input_width = encoded["input_ids"].shape[1]
            batch_started = time.perf_counter()
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
            batch_elapsed = time.perf_counter() - batch_started
            for index, example in enumerate(batch):
                completion_ids = output_ids[index, input_width:]
                completion_values = completion_ids.tolist()
                if tokenizer.eos_token_id in completion_values:
                    completion_values = completion_values[
                        : completion_values.index(tokenizer.eos_token_id) + 1
                    ]
                actual_completion_ids = torch.tensor(completion_values, device=completion_ids.device)
                raw_generation = tokenizer.decode(
                    actual_completion_ids, skip_special_tokens=True
                ).strip()
                generated_tokens += len(completion_values)
                record = {
                    "example_id": example.example_id,
                    "db_id": example.db_id,
                    "prediction": raw_generation,
                    "predicted_sql": extract_sql(raw_generation),
                    "input_tokens": int(input_lengths[index]),
                    "generated_tokens": len(completion_values),
                    "batch_elapsed_seconds": batch_elapsed,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

    elapsed = time.perf_counter() - started
    return {
        "model": str(model_path),
        "examples_generated": len(examples),
        "generated_tokens": generated_tokens,
        "elapsed_seconds": elapsed,
        "generated_tokens_per_second": generated_tokens / elapsed if elapsed else 0.0,
        "peak_input_tokens": peak_input_tokens,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
    }
