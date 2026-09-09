import json
import sqlite3

from text2sql.data.bird import BirdExample
from text2sql.preference.build_pairs import (
    build_preference_pairs,
    choose_pair,
    deduplicate_candidates,
)
from text2sql.preference.generate_candidates import (
    batch_seed,
    batched,
    candidate_chunk_size,
    candidate_file_stats,
    example_seed,
)


def test_example_seed_is_stable_and_example_specific():
    assert example_seed(42, "db:1") == example_seed(42, "db:1")
    assert example_seed(42, "db:1") != example_seed(42, "db:2")


def test_batches_and_batch_seeds_are_stable():
    examples = [
        BirdExample(f"db:{index}", "db", "q", "SELECT 1") for index in range(3)
    ]
    batches = list(batched(examples, 2))
    assert [len(batch) for batch in batches] == [2, 1]
    assert batch_seed(42, batches[0]) == batch_seed(42, batches[0])
    assert batch_seed(42, batches[0]) != batch_seed(42, batches[1])


def test_long_prompts_sample_candidates_serially():
    assert candidate_chunk_size(2999, 4, 3000) == 4
    assert candidate_chunk_size(3000, 4, 3000) == 1


def test_candidate_stats_and_deduplication(tmp_path):
    path = tmp_path / "candidates.jsonl"
    record = {
        "example_id": "db:1",
        "candidates": [
            {"candidate_index": 0, "predicted_sql": " SELECT 1; ", "generated_tokens": 3},
            {"candidate_index": 1, "predicted_sql": "select   1", "generated_tokens": 4},
            {"candidate_index": 2, "predicted_sql": "SELECT 2", "generated_tokens": 3},
        ],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    stats = candidate_file_stats(path)
    assert stats["output_candidates_total"] == 3
    assert stats["output_unique_candidates_within_examples"] == 2
    assert len(deduplicate_candidates(record["candidates"])) == 2


def test_choose_pair_prefers_generated_positive_and_hard_negative():
    labeled = [
        {
            "candidate_index": 0,
            "predicted_sql": "SELECT name FROM t",
            "execution_correct": True,
            "prediction_status": "success",
        },
        {
            "candidate_index": 1,
            "predicted_sql": "SELECT name FROM t WHERE id > 2",
            "execution_correct": False,
            "prediction_status": "success",
        },
        {
            "candidate_index": 2,
            "predicted_sql": "SELECT missing FROM t",
            "execution_correct": False,
            "prediction_status": "error",
        },
    ]
    pair = choose_pair(labeled)
    assert pair == (
        "SELECT name FROM t",
        "SELECT name FROM t WHERE id > 2",
        "generated",
        "hard",
    )


def test_gold_fallback_is_opt_in():
    candidates = [
        {
            "candidate_index": 0,
            "predicted_sql": "SELECT 2",
            "execution_correct": False,
            "prediction_status": "success",
        }
    ]
    assert choose_pair(candidates, gold_sql="SELECT 1") is None
    assert choose_pair(candidates, gold_sql="SELECT 1", allow_gold_fallback=True)[2] == (
        "gold_fallback"
    )


class _Prompts:
    def messages(self, example):
        return [{"role": "user", "content": example.question}]


def test_build_preference_pairs_executes_and_labels(tmp_path):
    db_root = tmp_path / "databases"
    db_dir = db_root / "toy"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "toy.sqlite"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE items (value INTEGER)")
    connection.executemany("INSERT INTO items VALUES (?)", [(1,), (2,)])
    connection.commit()
    connection.close()

    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "example_id": "toy:0",
                "db_id": "toy",
                "candidates": [
                    {"candidate_index": 0, "predicted_sql": "SELECT value FROM items"},
                    {
                        "candidate_index": 1,
                        "predicted_sql": "SELECT value FROM items WHERE value = 1",
                    },
                    {"candidate_index": 2, "predicted_sql": "SELECT nope FROM items"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pairs = tmp_path / "pairs.jsonl"
    labels = tmp_path / "labels.jsonl"
    metrics = build_preference_pairs(
        examples=[BirdExample("toy:0", "toy", "all values", "SELECT value FROM items")],
        candidates_path=candidates,
        database_root=db_root,
        prompts=_Prompts(),
        pairs_output=pairs,
        labels_output=labels,
    )
    pair = json.loads(pairs.read_text(encoding="utf-8"))
    assert metrics["pairs"] == 1
    assert metrics["negative_type_hard"] == 1
    assert pair["chosen"][0]["content"] == "SELECT value FROM items"
    assert "WHERE value = 1" in pair["rejected"][0]["content"]
