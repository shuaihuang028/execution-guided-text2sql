import json

import pytest

from text2sql.evaluation.official_parity import load_jsonl_by_id, official_set_match


def test_official_set_match_ignores_order_and_duplicates():
    assert official_set_match(((1,), (1,), (2,)), ((2,), (1,)))


def test_official_set_match_keeps_exact_python_value_semantics():
    assert not official_set_match(((1,),), ((2,),))


def test_jsonl_loader_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "items.jsonl"
    path.write_text(
        json.dumps({"example_id": "db:1"})
        + "\n"
        + json.dumps({"example_id": "db:1"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_jsonl_by_id(path)

