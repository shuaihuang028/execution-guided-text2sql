from __future__ import annotations

import json
import math
import platform
import random
import shutil
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from text2sql.data.bird import BirdExample
from text2sql.evaluation.generate import PromptRepository, seed_everything
from text2sql.training.sft_data import CompletionOnlyCollator, tokenize_sft_example


@dataclass(frozen=True, slots=True)
class SFTConfig:
    model_path: str
    output_dir: str
    max_seq_length: int = 6144
    num_train_epochs: float = 1.0
    max_steps: int = -1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True
    save_steps: int = 100
    save_total_limit: int = 2
    logging_steps: int = 1
    validation_limit: int = 128
    seed: int = 42


def _token_stats(items: list[dict]) -> dict[str, float | int]:
    lengths = sorted(len(item["input_ids"]) for item in items)
    return {
        "count": len(lengths),
        "min": lengths[0],
        "median": statistics.median(lengths),
        "p95": lengths[round((len(lengths) - 1) * 0.95)],
        "max": lengths[-1],
    }


def _validation_subset(
    examples: list[BirdExample], limit: int, seed: int
) -> list[BirdExample]:
    if limit <= 0 or limit >= len(examples):
        return list(examples)
    selected = list(examples)
    random.Random(seed).shuffle(selected)
    return selected[:limit]


def _training_order(item_count: int, example_count: int, seed: int) -> list[int]:
    if item_count <= 0:
        raise ValueError("No SFT training examples were selected")
    order: list[int] = []
    cycle = 0
    while len(order) < example_count:
        indices = list(range(item_count))
        random.Random(seed + cycle).shuffle(indices)
        order.extend(indices)
        cycle += 1
    return order[:example_count]


def _checkpoint_directories(output_dir: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        if path.is_dir() and (path / "training_state.pt").is_file():
            checkpoints.append((step, path))
    return sorted(checkpoints)


def _resolve_checkpoint(output_dir: Path, requested: str | Path | None) -> Path | None:
    if requested is None:
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory {output_dir} is not empty; pass --resume-from-checkpoint "
                "latest or choose a new directory"
            )
        return None
    if str(requested).casefold() == "latest":
        checkpoints = _checkpoint_directories(output_dir)
        if not checkpoints:
            raise ValueError(f"No complete checkpoint found under {output_dir}")
        return checkpoints[-1][1]
    checkpoint = Path(requested)
    if not checkpoint.is_dir() or not (checkpoint / "training_state.pt").is_file():
        raise FileNotFoundError(f"Incomplete checkpoint: {checkpoint}")
    return checkpoint


def _save_checkpoint(
    *,
    output_dir: Path,
    step: int,
    micro_step: int,
    tokens_seen: int,
    model,
    tokenizer,
    optimizer,
    scheduler,
    save_total_limit: int,
) -> Path:
    import numpy as np
    import torch

    checkpoint = output_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    torch.save(
        {
            "optimizer_step": step,
            "micro_step": micro_step,
            "tokens_seen": tokens_seen,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        },
        checkpoint / "training_state.pt",
    )
    if save_total_limit > 0:
        checkpoints = _checkpoint_directories(output_dir)
        for _, expired in checkpoints[:-save_total_limit]:
            shutil.rmtree(expired)
    return checkpoint


def _restore_training_state(checkpoint: Path, model, optimizer, scheduler) -> dict:
    import numpy as np
    import torch

    active_adapter = model.active_adapters[0]
    model.load_adapter(checkpoint, active_adapter, is_trainable=True)
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state


def _truncate_log_after_step(log_path: Path, saved_step: int) -> int:
    """Remove records written after the last durable checkpoint before resuming."""

    if not log_path.is_file():
        return 0
    retained: list[str] = []
    removed = 0
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if int(json.loads(line)["step"]) <= saved_step:
                retained.append(line)
            else:
                removed += 1
    log_path.write_text("".join(retained), encoding="utf-8")
    return removed


def run_sft(
    *,
    config: SFTConfig,
    train_examples: list[BirdExample],
    validation_examples: list[BirdExample],
    prompts: PromptRepository,
    metrics_output: str | Path,
    resume_from_checkpoint: str | Path | None = None,
) -> dict[str, object]:
    import bitsandbytes
    import peft
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if config.batch_size <= 0 or config.gradient_accumulation_steps <= 0:
        raise ValueError("Batch size and gradient accumulation steps must be positive")
    if config.num_train_epochs <= 0 and config.max_steps <= 0:
        raise ValueError("num_train_epochs or max_steps must be positive")

    seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _resolve_checkpoint(output_dir, resume_from_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    validation_examples = _validation_subset(
        validation_examples, config.validation_limit, config.seed
    )
    tokenization_started = time.perf_counter()
    train_dataset = [
        tokenize_sft_example(example, prompts, tokenizer, config.max_seq_length)
        for example in train_examples
    ]
    eval_dataset = [
        tokenize_sft_example(example, prompts, tokenizer, config.max_seq_length)
        for example in validation_examples
    ]
    tokenization_seconds = time.perf_counter() - tokenization_started

    batches_per_epoch = math.ceil(len(train_dataset) / config.batch_size)
    if config.max_steps > 0:
        total_optimizer_steps = config.max_steps
        total_micro_steps = config.max_steps * config.gradient_accumulation_steps
    else:
        total_micro_steps = math.ceil(batches_per_epoch * config.num_train_epochs)
        total_optimizer_steps = math.ceil(
            total_micro_steps / config.gradient_accumulation_steps
        )
    example_order = _training_order(
        len(train_dataset), total_micro_steps * config.batch_size, config.seed
    )
    ordered_dataset = [train_dataset[index] for index in example_order]

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.config.use_cache = False
    model.train()
    trainable_parameters, total_parameters = model.get_nb_trainable_parameters()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    warmup_steps = round(total_optimizer_steps * config.warmup_ratio)

    def learning_rate_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        remaining = max(total_optimizer_steps - warmup_steps, 1)
        return max((total_optimizer_steps - step) / remaining, 0.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    completed_micro_steps = 0
    optimizer_step = 0
    tokens_seen = 0
    resumed_micro_steps = 0
    resumed_tokens_seen = 0
    if checkpoint is not None:
        state = _restore_training_state(checkpoint, model, optimizer, scheduler)
        completed_micro_steps = int(state["micro_step"])
        optimizer_step = int(state["optimizer_step"])
        tokens_seen = int(state["tokens_seen"])
        resumed_micro_steps = completed_micro_steps
        resumed_tokens_seen = tokens_seen

    remaining_dataset = ordered_dataset[completed_micro_steps * config.batch_size :]
    dataloader = DataLoader(
        remaining_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=CompletionOnlyCollator(tokenizer.pad_token_id),
    )
    log_path = output_dir / "train_log.jsonl"
    log_mode = "a" if checkpoint is not None else "w"
    discarded_log_records = (
        _truncate_log_after_step(log_path, optimizer_step) if checkpoint is not None else 0
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    step_started = started
    accumulated_loss = 0.0
    micro_in_group = 0
    current_group_size = 0
    optimizer.zero_grad(set_to_none=True)

    with log_path.open(log_mode, encoding="utf-8") as log_handle:
        for batch in dataloader:
            if micro_in_group == 0:
                current_group_size = min(
                    config.gradient_accumulation_steps,
                    total_micro_steps - completed_micro_steps,
                )
            completed_micro_steps += 1
            batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
            tokens_seen += int(batch["attention_mask"].sum().item())
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw_loss = model(**batch).loss
                loss = raw_loss / current_group_size
            accumulated_loss += float(raw_loss.detach().cpu())
            loss.backward()
            micro_in_group += 1
            if micro_in_group < current_group_size:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            optimizer_step += 1
            elapsed = time.perf_counter() - step_started
            record = {
                "step": optimizer_step,
                "loss": accumulated_loss / micro_in_group,
                "learning_rate": scheduler.get_last_lr()[0],
                "step_seconds": elapsed,
                "tokens_seen": tokens_seen,
                "tokens_per_second": (tokens_seen - resumed_tokens_seen)
                / (time.perf_counter() - started),
                "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
            }
            if optimizer_step % config.logging_steps == 0:
                log_handle.write(json.dumps(record) + "\n")
                log_handle.flush()
            accumulated_loss = 0.0
            micro_in_group = 0
            current_group_size = 0
            step_started = time.perf_counter()
            if optimizer_step % config.save_steps == 0 or optimizer_step == total_optimizer_steps:
                _save_checkpoint(
                    output_dir=output_dir,
                    step=optimizer_step,
                    micro_step=completed_micro_steps,
                    tokens_seen=tokens_seen,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    save_total_limit=config.save_total_limit,
                )
            if optimizer_step >= total_optimizer_steps:
                break

    training_seconds = time.perf_counter() - started
    evaluation_loss: float | None = None
    if eval_dataset:
        model.eval()
        losses: list[float] = []
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=CompletionOnlyCollator(tokenizer.pad_token_id),
        )
        with torch.inference_mode():
            for batch in eval_loader:
                batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    losses.append(float(model(**batch).loss.detach().cpu()))
        evaluation_loss = statistics.fmean(losses)

    final_adapter = output_dir / "final_adapter"
    final_adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_adapter)
    tokenizer.save_pretrained(final_adapter)
    processed_this_run = ordered_dataset[
        resumed_micro_steps * config.batch_size : completed_micro_steps * config.batch_size
    ]
    metrics: dict[str, object] = {
        "config": asdict(config),
        "prompt_config": {
            "max_tables": prompts.max_tables,
            "max_description_chars": prompts.builder.max_description_chars,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "bitsandbytes": bitsandbytes.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "resume_from_checkpoint": str(checkpoint) if checkpoint else None,
        "discarded_uncheckpointed_log_records": discarded_log_records,
        "tokenization_seconds": tokenization_seconds,
        "train_sequence_tokens": _token_stats(train_dataset),
        "validation_sequence_tokens": _token_stats(eval_dataset) if eval_dataset else None,
        "optimizer_steps": optimizer_step,
        "micro_steps": completed_micro_steps,
        "tokens_seen": tokens_seen,
        "training_seconds_this_run": training_seconds,
        "tokens_per_second_this_run": (
            sum(len(item["input_ids"]) for item in processed_this_run) / training_seconds
            if training_seconds
            else 0.0
        ),
        "validation_loss": evaluation_loss,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
        "trainable_parameters": trainable_parameters,
        "total_parameters_loaded": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
    }
    output_path = Path(metrics_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
