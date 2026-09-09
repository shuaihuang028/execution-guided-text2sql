from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DPOTrainingConfig:
    adapter_path: str
    output_dir: str
    max_seq_length: int = 6144
    num_train_epochs: float = 1.0
    max_steps: int = -1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    beta: float = 0.1
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    validation_fraction: float = 0.05
    logging_steps: int = 1
    save_steps: int = 100
    save_total_limit: int = 2
    seed: int = 42


def load_preference_records(path: str | Path) -> list[dict]:
    required = {"example_id", "db_id", "prompt", "chosen", "rejected"}
    records: list[dict] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = required - record.keys()
            if missing:
                raise ValueError(f"Preference line {line_number} is missing {sorted(missing)}")
            example_id = str(record["example_id"])
            if example_id in seen:
                raise ValueError(f"Duplicate preference example_id {example_id!r}")
            seen.add(example_id)
            records.append(record)
    if len(records) < 2:
        raise ValueError("At least two preference pairs are required")
    return records


def split_preference_records(
    records: list[dict], validation_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")

    by_database: dict[str, list[dict]] = {}
    for record in records:
        by_database.setdefault(str(record["db_id"]), []).append(record)

    def score(value: str) -> bytes:
        return hashlib.sha256(f"{seed}:{value}".encode()).digest()

    if len(by_database) == 1:
        ordered = sorted(records, key=lambda record: score(str(record["example_id"])))
        validation_count = max(1, round(len(ordered) * validation_fraction))
        validation_count = min(validation_count, len(ordered) - 1)
        return ordered[validation_count:], ordered[:validation_count]

    target = max(1, round(len(records) * validation_fraction))
    validation_databases: set[str] = set()
    validation_count = 0
    ordered_databases = sorted(by_database, key=score)
    for db_id in ordered_databases[:-1]:
        if validation_count >= target:
            break
        validation_databases.add(db_id)
        validation_count += len(by_database[db_id])
    validation = [
        record for record in records if str(record["db_id"]) in validation_databases
    ]
    train = [
        record for record in records if str(record["db_id"]) not in validation_databases
    ]
    return train, validation


def preference_sequence_length(record: dict, tokenizer) -> int:
    prompt = list(record["prompt"])
    chosen = list(record["chosen"])
    rejected = list(record["rejected"])
    chosen_ids = tokenizer.apply_chat_template(
        [*prompt, *chosen], tokenize=True, add_generation_prompt=False
    )
    rejected_ids = tokenizer.apply_chat_template(
        [*prompt, *rejected], tokenize=True, add_generation_prompt=False
    )
    return max(len(chosen_ids), len(rejected_ids))


def filter_overlength_records(
    records: list[dict], tokenizer, max_seq_length: int
) -> tuple[list[dict], list[dict], list[int]]:
    retained: list[dict] = []
    removed: list[dict] = []
    lengths: list[int] = []
    for record in records:
        length = preference_sequence_length(record, tokenizer)
        if length <= max_seq_length:
            retained.append(record)
            lengths.append(length)
        else:
            removed.append(record)
    return retained, removed, lengths


def select_best_sweep_run(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("No completed DPO sweep runs")

    def rank(run: dict) -> tuple[float, float]:
        validation = run["validation_metrics"]
        return (
            float(validation["eval_rewards/accuracies"]),
            -float(validation["eval_loss"]),
        )

    return max(runs, key=rank)


def run_dpo(
    *,
    config: DPOTrainingConfig,
    preferences_path: str | Path,
    metrics_output: str | Path,
    resume_from_checkpoint: str | Path | None = None,
) -> dict[str, object]:
    import bitsandbytes
    import peft
    import torch
    import transformers
    import trl
    from datasets import Dataset
    from peft import PeftConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    if config.batch_size <= 0 or config.gradient_accumulation_steps <= 0:
        raise ValueError("Batch size and gradient accumulation steps must be positive")
    if config.learning_rate <= 0 or config.beta <= 0:
        raise ValueError("Learning rate and beta must be positive")

    adapter_config = PeftConfig.from_pretrained(config.adapter_path)
    base_model_path = str(adapter_config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    records = load_preference_records(preferences_path)
    records, overlength, sequence_lengths = filter_overlength_records(
        records, tokenizer, config.max_seq_length
    )
    if len(records) < 2:
        raise ValueError("Fewer than two preference pairs remain after length filtering")
    train_records, validation_records = split_preference_records(
        records, config.validation_fraction, config.seed
    )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = PeftModel.from_pretrained(model, config.adapter_path, is_trainable=True)
    model.config.use_cache = False
    trainable_parameters, total_parameters = model.get_nb_trainable_parameters()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        beta=config.beta,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=config.max_seq_length,
        truncation_mode="keep_start",
        eval_strategy="no",
        logging_steps=config.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        report_to=[],
        remove_unused_columns=False,
        seed=config.seed,
        data_seed=config.seed,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=Dataset.from_list(train_records),
        eval_dataset=Dataset.from_list(validation_records),
        processing_class=tokenizer,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train_result = trainer.train(
        resume_from_checkpoint=str(resume_from_checkpoint)
        if resume_from_checkpoint is not None
        else None
    )
    training_seconds = time.perf_counter() - started
    evaluation = trainer.evaluate()
    final_adapter = output_dir / "final_adapter"
    trainer.accelerator.unwrap_model(trainer.model).save_pretrained(
        final_adapter, selected_adapters=["default"]
    )
    tokenizer.save_pretrained(final_adapter)
    log_path = output_dir / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as handle:
        for record in trainer.state.log_history:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics: dict[str, object] = {
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "trl": trl.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "base_model_path": base_model_path,
        "preference_pairs_input": len(records) + len(overlength),
        "preference_pairs_retained": len(records),
        "overlength_pairs_removed": len(overlength),
        "train_pairs": len(train_records),
        "validation_pairs": len(validation_records),
        "sequence_tokens": {
            "min": min(sequence_lengths),
            "max": max(sequence_lengths),
            "mean": sum(sequence_lengths) / len(sequence_lengths),
        },
        "trainable_parameters": trainable_parameters,
        "total_parameters_loaded": total_parameters,
        "training_seconds": training_seconds,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
        "train_metrics": train_result.metrics,
        "validation_metrics": evaluation,
        "final_adapter": str(final_adapter),
        "log_path": str(log_path),
    }
    metrics_path = Path(metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics
