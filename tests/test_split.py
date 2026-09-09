from text2sql.data.bird import BirdExample
from text2sql.data.split import split_by_database


def test_database_grouped_split_has_no_schema_leakage():
    examples = [
        BirdExample(f"{db_id}:{index}", db_id, "q", "SELECT 1")
        for db_id in ("a", "b", "c", "d")
        for index in range(3)
    ]
    split = split_by_database(examples, validation_fraction=0.25, seed=42)
    assert set(split.train_db_ids).isdisjoint(split.validation_db_ids)
    assert len(split.train) + len(split.validation) == len(examples)
    assert split == split_by_database(examples, validation_fraction=0.25, seed=42)

