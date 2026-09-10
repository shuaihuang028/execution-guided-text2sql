import pytest

from text2sql.preference.subsets import sample_preference_subset


def _records(count: int) -> list[dict]:
    return [{"example_id": f"db{index % 4}:{index}", "payload": index} for index in range(count)]


def test_sample_is_deterministic_for_a_seed():
    records = _records(50)
    first = sample_preference_subset(records, 10, 42)
    second = sample_preference_subset(records, 10, 42)
    assert [item["example_id"] for item in first] == [item["example_id"] for item in second]


def test_different_seeds_give_different_subsets():
    records = _records(50)
    a = {item["example_id"] for item in sample_preference_subset(records, 10, 1)}
    b = {item["example_id"] for item in sample_preference_subset(records, 10, 2)}
    assert a != b


def test_sample_returns_requested_size_and_is_a_subset():
    records = _records(50)
    subset = sample_preference_subset(records, 17, 7)
    assert len(subset) == 17
    ids = {item["example_id"] for item in records}
    assert {item["example_id"] for item in subset} <= ids


def test_sample_preserves_input_order():
    records = _records(50)
    subset = sample_preference_subset(records, 20, 3)
    positions = [records.index(item) for item in subset]
    assert positions == sorted(positions)


def test_sample_does_not_depend_on_input_ordering():
    """Regenerating from a seed must not require the original file order."""

    records = _records(50)
    shuffled = list(reversed(records))
    forward = {item["example_id"] for item in sample_preference_subset(records, 12, 5)}
    backward = {item["example_id"] for item in sample_preference_subset(shuffled, 12, 5)}
    assert forward == backward


def test_sampling_everything_returns_everything():
    records = _records(9)
    subset = sample_preference_subset(records, 9, 42)
    assert [item["example_id"] for item in subset] == [item["example_id"] for item in records]


def test_oversized_sample_is_rejected():
    with pytest.raises(ValueError, match="Cannot sample"):
        sample_preference_subset(_records(5), 6, 42)


def test_non_positive_size_is_rejected():
    with pytest.raises(ValueError, match="size must be positive"):
        sample_preference_subset(_records(5), 0, 42)


def test_duplicate_ids_are_rejected():
    records = [{"example_id": "a"}, {"example_id": "a"}]
    with pytest.raises(ValueError, match="duplicate example_id"):
        sample_preference_subset(records, 1, 42)
