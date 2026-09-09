from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

from text2sql.data.bird import BirdExample
from text2sql.evaluation.generate import PromptRepository
from text2sql.evaluation.output_parser import extract_sql


def example_seed(seed: int, example_id: str) -> int:
    """Derive a stable per-example seed so resumed generation is reproducible."""

    digest = hashlib.sha256(f"{seed}:{example_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def batch_seed(seed: int, examples: Sequence[BirdExample]) -> int:
    return example_seed(seed, "|".join(example.example_id for example in examples))


def batched(
    examples: Sequence[BirdExample], batch_size: int
) -> Iterable[Sequence[BirdExample]]:
    for start in range(0, len(examples), batch_size):
        yield examples[start : start + batch_size]


def candidate_chunk_size(
    max_prompt_tokens: int,
    candidates_per_example: int,
    vectorized_candidate_max_prompt_tokens: int,
) -> int:
    return (
        1
        if max_prompt_tokens >= vectorized_candidate_max_prompt_tokens
        else candidates_per_example
    )


def completed_candidate_ids(path: str | Path) -> set[str]:
    output = Path(path)
    if not output.is_file():
        return set()
    with output.open("r", encoding="utf-8") as handle:
        return {
            str(json.loads(line)["example_id"])
            for line in handle
            if line.strip()
        }


def candidate_file_stats(path: str | Path) -> dict[str, int | float]:
    with Path(path).open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    ids = [str(record["example_id"]) for record in records]
    candidates = [candidate for record in records for candidate in record["candidates"]]
    normalized = [
        " ".join(str(candidate["predicted_sql"]).split()).strip().rstrip(";").casefold()
        for candidate in candidates
    ]
    unique_per_example = sum(
        len(
            {
                " ".join(str(candidate["predicted_sql"]).split())
                .strip()
                .rstrip(";")
                .casefold()
                for candidate in record["candidates"]
            }
        )
        for record in records
    )
    return {
        "output_examples_total": len(records),
        "output_unique_ids": len(set(ids)),
        "output_duplicate_ids": len(ids) - len(set(ids)),
        "output_candidates_total": len(candidates),
        "output_unique_candidates_within_examples": unique_per_example,
        "within_example_unique_rate": unique_per_example / len(normalized)
        if normalized
        else 0.0,
        "output_generated_tokens_total": sum(
            int(candidate.get("generated_tokens", 0)) for candidate in candidates
        ),
    }


def generate_candidates(
    *,
    model_path: str | Path,
    examples: Sequence[BirdExample],
    prompts: PromptRepository,
    output_path: str | Path,
    batch_size: int = 1,
    max_batch_input_tokens: int = 10_000,
    vectorized_candidate_max_prompt_tokens: int = 3_000,
    candidates_per_example: int = 4,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_input_tokens: int = 8192,
    max_new_tokens: int = 384,
    seed: int = 42,
) -> dict[str, float | int | str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (
        batch_size <= 0
        or max_batch_input_tokens <= 0
        or vectorized_candidate_max_prompt_tokens <= 0
    ):
        raise ValueError("Batch and token-budget settings must be positive")
    if candidates_per_example < 2:
        raise ValueError("candidates_per_example must be at least 2")
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be positive and top_p must be in (0, 1]")

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
    started = time.perf_counter()
    generated_tokens = 0
    peak_input_tokens = 0
    minimum_effective_batch_size = batch_size
    minimum_candidate_chunk_size = candidates_per_example
    torch.cuda.reset_peak_memory_stats()

    with output.open("a", encoding="utf-8") as handle, torch.inference_mode():
        cursor = 0
        while cursor < len(examples):
            batch = examples[cursor : cursor + batch_size]
            rendered = [
                tokenizer.apply_chat_template(
                    prompts.messages(example), tokenize=False, add_generation_prompt=True
                )
                for example in batch
            ]
            encoded = tokenizer(
                rendered, return_tensors="pt", padding=True, truncation=False
            )
            input_lengths = [int(value) for value in encoded["attention_mask"].sum(dim=1)]
            effective_batch_size = max(
                1,
                min(
                    len(batch),
                    max_batch_input_tokens // max(input_lengths),
                ),
            )
            if effective_batch_size < len(batch):
                batch = batch[:effective_batch_size]
                rendered = rendered[:effective_batch_size]
                encoded = tokenizer(
                    rendered, return_tensors="pt", padding=True, truncation=False
                )
                input_lengths = [
                    int(value) for value in encoded["attention_mask"].sum(dim=1)
                ]
            minimum_effective_batch_size = min(minimum_effective_batch_size, len(batch))
            local_seed = batch_seed(seed, batch)
            peak_input_tokens = max(peak_input_tokens, max(input_lengths))
            if peak_input_tokens > max_input_tokens:
                raise ValueError(
                    f"A prompt has {peak_input_tokens} input tokens, exceeding "
                    f"{max_input_tokens}; adjust schema pruning rather than truncating"
                )
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            input_width = encoded["input_ids"].shape[1]
            chunk_size = candidate_chunk_size(
                max(input_lengths),
                candidates_per_example,
                vectorized_candidate_max_prompt_tokens,
            )
            minimum_candidate_chunk_size = min(minimum_candidate_chunk_size, chunk_size)
            candidate_records_by_example: list[list[dict]] = [
                [] for _ in range(len(batch))
            ]
            batch_started = time.perf_counter()
            for candidate_start in range(0, candidates_per_example, chunk_size):
                current_chunk_size = min(
                    chunk_size, candidates_per_example - candidate_start
                )
                chunk_seed = (
                    local_seed
                    if chunk_size == candidates_per_example
                    else example_seed(local_seed, f"candidate_chunk:{candidate_start}")
                )
                torch.manual_seed(chunk_seed)
                torch.cuda.manual_seed_all(chunk_seed)
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=current_chunk_size,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                for example_index in range(len(batch)):
                    sequence_start = example_index * current_chunk_size
                    sequences = output_ids[
                        sequence_start : sequence_start + current_chunk_size
                    ]
                    for chunk_index, sequence in enumerate(sequences):
                        completion = sequence[input_width:].tolist()
                        if tokenizer.eos_token_id in completion:
                            completion = completion[
                                : completion.index(tokenizer.eos_token_id) + 1
                            ]
                        raw = tokenizer.decode(
                            completion, skip_special_tokens=True
                        ).strip()
                        generated_tokens += len(completion)
                        candidate_records_by_example[example_index].append(
                            {
                                "candidate_index": candidate_start + chunk_index,
                                "generation": raw,
                                "predicted_sql": extract_sql(raw),
                                "generated_tokens": len(completion),
                            }
                        )
            torch.cuda.synchronize()
            batch_elapsed = time.perf_counter() - batch_started
            for example_index, example in enumerate(batch):
                record = {
                    "example_id": example.example_id,
                    "db_id": example.db_id,
                    "batch_seed": local_seed,
                    "effective_batch_size": len(batch),
                    "candidate_chunk_size": chunk_size,
                    "input_tokens": input_lengths[example_index],
                    "batch_elapsed_seconds": batch_elapsed,
                    "candidates": candidate_records_by_example[example_index],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            cursor += len(batch)

    elapsed = time.perf_counter() - started
    return {
        "model": str(model_path),
        "examples_generated_this_run": len(examples),
        "batch_size": batch_size,
        "max_batch_input_tokens": max_batch_input_tokens,
        "minimum_effective_batch_size": minimum_effective_batch_size,
        "vectorized_candidate_max_prompt_tokens": (
            vectorized_candidate_max_prompt_tokens
        ),
        "minimum_candidate_chunk_size": minimum_candidate_chunk_size,
        "candidates_per_example": candidates_per_example,
        "temperature": temperature,
        "top_p": top_p,
        "generated_tokens_this_run": generated_tokens,
        "elapsed_seconds_this_run": elapsed,
        "generated_tokens_per_second": generated_tokens / elapsed if elapsed else 0.0,
        "peak_input_tokens": peak_input_tokens,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
    }
