import json

import pytest

from text2sql.training.dpo import (
    load_preference_records,
    select_best_sweep_run,
    split_preference_records,
)


def _record(example_id, db_id=None):
    return {
        "example_id": example_id,
        "db_id": db_id or example_id.split(":")[0],
        "prompt": [{"role": "user", "content": "question"}],
        "chosen": [{"role": "assistant", "content": "SELECT 1"}],
        "rejected": [{"role": "assistant", "content": "SELECT 2"}],
    }


def test_load_preferences_validates_unique_ids(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps(_record("db:1")) + "\n" + json.dumps(_record("db:1")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_preference_records(path)


def test_preference_split_is_deterministic_and_disjoint():
    records = [
        _record(f"db{index // 5}:{index}", f"db{index // 5}") for index in range(20)
    ]
    train_a, validation_a = split_preference_records(records, 0.1, 42)
    train_b, validation_b = split_preference_records(records, 0.1, 42)
    assert train_a == train_b
    assert validation_a == validation_b
    assert len(train_a) == 15
    assert len(validation_a) == 5
    assert {item["example_id"] for item in train_a}.isdisjoint(
        item["example_id"] for item in validation_a
    )
    assert {item["db_id"] for item in train_a}.isdisjoint(
        item["db_id"] for item in validation_a
    )


def test_sweep_selection_uses_accuracy_then_loss():
    runs = [
        {
            "name": "low-accuracy",
            "validation_metrics": {"eval_rewards/accuracies": 0.6, "eval_loss": 0.2},
        },
        {
            "name": "high-loss",
            "validation_metrics": {"eval_rewards/accuracies": 0.7, "eval_loss": 0.5},
        },
        {
            "name": "winner",
            "validation_metrics": {"eval_rewards/accuracies": 0.7, "eval_loss": 0.4},
        },
    ]
    assert select_best_sweep_run(runs)["name"] == "winner"
