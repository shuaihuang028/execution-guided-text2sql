import json

from text2sql.data.bird import BirdExample
from text2sql.evaluation.generate import pending_examples, prediction_file_stats


def test_pending_examples_filters_manifest_completed_and_limit():
    examples = [
        BirdExample("a:1", "a", "q", "SELECT 1"),
        BirdExample("b:2", "b", "q", "SELECT 2"),
        BirdExample("c:3", "c", "q", "SELECT 3"),
    ]
    pending = pending_examples(examples, {"a:1", "b:2"}, {"a:1"}, limit=1)
    assert [example.example_id for example in pending] == ["b:2"]


def test_prediction_file_stats_covers_all_resumed_segments(tmp_path):
    output = tmp_path / "predictions.jsonl"
    records = [
        {"example_id": "db:1", "generated_tokens": 4},
        {"example_id": "db:2", "generated_tokens": 7},
        {"example_id": "db:2", "generated_tokens": 7},
    ]
    output.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    assert prediction_file_stats(output) == {
        "output_examples_total": 3,
        "output_unique_ids": 2,
        "output_duplicate_ids": 1,
        "output_generated_tokens_total": 18,
    }
