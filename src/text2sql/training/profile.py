from __future__ import annotations

import json
import math
import platform
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from text2sql.data.bird import BirdExample
from text2sql.evaluation.generate import PromptRepository, seed_everything
from text2sql.training.sft_data import CompletionOnlyCollator, tokenize_sft_example


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    model_path: str
    max_seq_length: int = 6144
    max_steps: int = 200
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True
    seed: int = 42


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def _select_examples(
    examples: list[BirdExample], count: int, seed: int
) -> list[BirdExample]:
    if not examples:
        raise ValueError("No training examples were selected")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    repeats = math.ceil(count / len(shuffled))
    return (shuffled * repeats)[:count]


def run_sft_profile(
    *,
    config: ProfileConfig,
    examples: list[BirdExample],
    prompts: PromptRepository,
    metrics_output: str | Path,
    step_log_output: str | Path,
    adapter_output: str | Path | None = None,
) -> dict[str, object]:
    import bitsandbytes
    import peft
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    seed_everything(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    micro_steps = config.max_steps * config.gradient_accumulation_steps
    selected = _select_examples(examples, micro_steps * config.batch_size, config.seed)
    tokenization_started = time.perf_counter()
    tokenized = [
        tokenize_sft_example(example, prompts, tokenizer, config.max_seq_length)
        for example in selected
    ]
    tokenization_seconds = time.perf_counter() - tokenization_started
    sequence_lengths = [len(item["input_ids"]) for item in tokenized]

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
    lora = LoraConfig(
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
    )
    model = get_peft_model(model, lora)
    model.config.use_cache = False
    model.train()

    trainable_parameters, total_parameters = model.get_nb_trainable_parameters()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    warmup_steps = round(config.max_steps * config.warmup_ratio)

    def learning_rate_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        remaining = max(config.max_steps - warmup_steps, 1)
        return max((config.max_steps - step) / remaining, 0.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    dataloader = DataLoader(
        tokenized,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=CompletionOnlyCollator(tokenizer.pad_token_id),
    )

    log_path = Path(step_log_output)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    accumulated_tokens = 0
    optimizer_step_times: list[float] = []
    optimizer_step = 0
    step_started = time.perf_counter()
    accumulated_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    with log_path.open("w", encoding="utf-8") as log_handle:
        for micro_step, batch in enumerate(dataloader, start=1):
            batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
            tokens = int(batch["attention_mask"].sum().item())
            accumulated_tokens += tokens
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / config.gradient_accumulation_steps
            accumulated_loss += float(loss.detach().cpu())
            loss.backward()

            if micro_step % config.gradient_accumulation_steps != 0:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

            optimizer_step += 1
            elapsed = time.perf_counter() - step_started
            optimizer_step_times.append(elapsed)
            record = {
                "step": optimizer_step,
                "loss": accumulated_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "step_seconds": elapsed,
                "tokens_seen": accumulated_tokens,
                "tokens_per_second": accumulated_tokens
                / (time.perf_counter() - training_started),
                "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
            }
            log_handle.write(json.dumps(record) + "\n")
            log_handle.flush()
            step_started = time.perf_counter()
            accumulated_loss = 0.0
            if optimizer_step >= config.max_steps:
                break

    training_seconds = time.perf_counter() - training_started
    if adapter_output is not None:
        model.save_pretrained(adapter_output)
        tokenizer.save_pretrained(adapter_output)

    metrics = {
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
        "examples_tokenized": len(tokenized),
        "tokenization_seconds": tokenization_seconds,
        "sequence_tokens": {
            "min": min(sequence_lengths),
            "median": statistics.median(sequence_lengths),
            "p95": _percentile([float(value) for value in sequence_lengths], 0.95),
            "max": max(sequence_lengths),
        },
        "optimizer_steps": optimizer_step,
        "tokens_seen": accumulated_tokens,
        "training_seconds": training_seconds,
        "tokens_per_second": accumulated_tokens / training_seconds,
        "step_seconds": {
            "median": statistics.median(optimizer_step_times),
            "p95": _percentile(optimizer_step_times, 0.95),
        },
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_vram_mib": torch.cuda.max_memory_reserved() / 2**20,
        "trainable_parameters": trainable_parameters,
        "total_parameters_loaded": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
    }
    output_path = Path(metrics_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
