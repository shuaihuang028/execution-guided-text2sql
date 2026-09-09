from __future__ import annotations

from dataclasses import dataclass

import torch

from text2sql.data.bird import BirdExample
from text2sql.evaluation.generate import PromptRepository


def make_completion_labels(prompt_ids: list[int], full_ids: list[int]) -> list[int]:
    """Mask the prompt and retain assistant completion tokens as language-model labels."""

    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Prompt token IDs are not a prefix of the full conversation")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("The assistant completion is empty")
    return [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]


def tokenize_sft_example(
    example: BirdExample,
    prompts: PromptRepository,
    tokenizer,
    max_seq_length: int,
) -> dict[str, list[int] | str | int]:
    messages = prompts.messages(example)
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": example.sql}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if len(full_ids) > max_seq_length:
        raise ValueError(
            f"{example.example_id} has {len(full_ids)} tokens, exceeding {max_seq_length}; "
            "do not silently truncate target SQL"
        )
    labels = make_completion_labels(prompt_ids, full_ids)
    return {
        "example_id": example.example_id,
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "length": len(full_ids),
    }


@dataclass(frozen=True, slots=True)
class CompletionOnlyCollator:
    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        longest = max(len(feature["input_ids"]) for feature in features)
        padded_length = (
            (longest + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
        ) * self.pad_to_multiple_of
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            padding = padded_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_masks.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
