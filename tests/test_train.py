import json

from text2sql.data.bird import BirdExample
from text2sql.training.train import (
    _token_stats,
    _truncate_log_after_step,
    _validation_subset,
)


def _examples(count: int) -> list[BirdExample]:
    return [
        BirdExample(
            example_id=f"db:{index}",
            db_id="db",
            question=f"question {index}",
            sql="SELECT 1",
        )
        for index in range(count)
    ]


def test_validation_subset_is_bounded_and_deterministic():
    examples = _examples(10)
    first = _validation_subset(examples, limit=4, seed=42)
    second = _validation_subset(examples, limit=4, seed=42)
    assert [item.example_id for item in first] == [item.example_id for item in second]
    assert len(first) == 4
    assert len({item.example_id for item in first}) == 4


def test_validation_subset_zero_means_all_examples():
    examples = _examples(3)
    assert _validation_subset(examples, limit=0, seed=42) == examples


def test_token_stats_reports_exact_distribution():
    items = [
        {"input_ids": [1]},
        {"input_ids": [1, 2, 3]},
        {"input_ids": [1, 2, 3, 4, 5]},
    ]
    assert _token_stats(items) == {
        "count": 3,
        "min": 1,
        "median": 3,
        "p95": 5,
        "max": 5,
    }


def test_resume_truncates_records_newer_than_checkpoint(tmp_path):
    log = tmp_path / "train_log.jsonl"
    log.write_text(
        "".join(json.dumps({"step": step}) + "\n" for step in range(1, 6)),
        encoding="utf-8",
    )
    assert _truncate_log_after_step(log, saved_step=3) == 2
    assert [json.loads(line)["step"] for line in log.read_text().splitlines()] == [1, 2, 3]
