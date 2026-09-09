from __future__ import annotations

import json

import pytest

from text2sql.data.bird import load_bird_examples


def test_loads_jsonl_and_normalizes_fields(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "db_id": "shop",
                "question_id": 7,
                "question": " Total sales? ",
                "evidence": " use orders.total ",
                "SQL": " SELECT SUM(total) FROM orders ",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    examples = load_bird_examples(path)
    assert examples[0].example_id == "shop:7"
    assert examples[0].question == "Total sales?"
    assert examples[0].sql == "SELECT SUM(total) FROM orders"


def test_rejects_missing_required_fields(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('[{"question": "hello", "SQL": "SELECT 1"}]', encoding="utf-8")
    with pytest.raises(ValueError, match="db_id"):
        load_bird_examples(path)

