import pytest
import torch

from text2sql.training.sft_data import CompletionOnlyCollator, make_completion_labels


def test_completion_labels_mask_every_prompt_token():
    assert make_completion_labels([1, 2, 3], [1, 2, 3, 4, 5]) == [-100, -100, -100, 4, 5]


def test_completion_labels_reject_non_prefix():
    with pytest.raises(ValueError, match="not a prefix"):
        make_completion_labels([1, 9], [1, 2, 3])


def test_completion_collator_masks_padding():
    collator = CompletionOnlyCollator(pad_token_id=0, pad_to_multiple_of=4)
    batch = collator(
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
            {"input_ids": [4], "attention_mask": [1], "labels": [4]},
        ]
    )
    assert batch["input_ids"].shape == torch.Size([2, 4])
    assert batch["attention_mask"].tolist()[1] == [1, 0, 0, 0]
    assert batch["labels"].tolist()[1] == [4, -100, -100, -100]
