import pytest

from text2sql.evaluation.significance import (
    exact_binomial_two_sided,
    mcnemar_test,
)


def test_binomial_symmetric_split_is_not_significant():
    assert exact_binomial_two_sided(5, 10) == pytest.approx(1.0)


def test_binomial_all_one_way_is_two_over_two_to_the_n():
    # Both tails of the extreme outcome: 2 * (1/2^6)
    assert exact_binomial_two_sided(6, 6) == pytest.approx(2 / 64)
    assert exact_binomial_two_sided(0, 6) == pytest.approx(2 / 64)


def test_binomial_is_symmetric():
    assert exact_binomial_two_sided(3, 11) == pytest.approx(exact_binomial_two_sided(8, 11))


def test_binomial_never_exceeds_one():
    for trials in range(12):
        for successes in range(trials + 1):
            assert 0.0 <= exact_binomial_two_sided(successes, trials) <= 1.0


def test_binomial_no_trials_is_one():
    assert exact_binomial_two_sided(0, 0) == 1.0


def test_binomial_rejects_impossible_counts():
    with pytest.raises(ValueError, match="Require"):
        exact_binomial_two_sided(5, 3)


def test_mcnemar_counts_only_discordant_pairs():
    scored = {f"e{i}" for i in range(10)}
    baseline = {"e0", "e1", "e2", "e3"}
    candidate = {"e0", "e1", "e4", "e5", "e6"}
    result = mcnemar_test(baseline, candidate, scored)
    assert result.only_baseline_correct == 2  # e2, e3
    assert result.only_candidate_correct == 3  # e4, e5, e6
    assert result.discordant == 5
    assert result.total == 10


def test_mcnemar_identical_predictions_are_not_significant():
    scored = {f"e{i}" for i in range(20)}
    same = {"e1", "e2", "e3"}
    result = mcnemar_test(same, same, scored)
    assert result.discordant == 0
    assert result.p_value == 1.0
    assert not result.is_significant()
    assert result.accuracy_delta == 0.0


def test_mcnemar_large_one_sided_gap_is_significant():
    scored = {f"e{i}" for i in range(200)}
    baseline = set()
    candidate = {f"e{i}" for i in range(30)}
    result = mcnemar_test(baseline, candidate, scored)
    assert result.only_candidate_correct == 30
    assert result.only_baseline_correct == 0
    assert result.is_significant()
    assert result.accuracy_delta == pytest.approx(0.15)


def test_mcnemar_small_gap_is_not_significant():
    """The regime this project lands in: a couple of points on ~1,500 examples."""

    scored = {f"e{i}" for i in range(1534)}
    baseline = {f"b{i}" for i in range(20)} | {f"shared{i}" for i in range(400)}
    candidate = {f"c{i}" for i in range(26)} | {f"shared{i}" for i in range(400)}
    # Discordant 20 vs 26 -> well within coin-flip noise.
    result = mcnemar_test(baseline, candidate, scored | baseline | candidate)
    assert result.only_baseline_correct == 20
    assert result.only_candidate_correct == 26
    assert not result.is_significant()


def test_mcnemar_ignores_ids_outside_the_scored_set():
    scored = {"a", "b"}
    result = mcnemar_test({"a", "zzz"}, {"b", "yyy"}, scored)
    assert result.baseline_correct == 1
    assert result.candidate_correct == 1
    assert result.total == 2


def test_result_serializes_accuracies():
    scored = {f"e{i}" for i in range(4)}
    payload = mcnemar_test({"e0"}, {"e0", "e1"}, scored).as_dict()
    assert payload["baseline_accuracy"] == pytest.approx(0.25)
    assert payload["candidate_accuracy"] == pytest.approx(0.5)
    assert payload["accuracy_delta"] == pytest.approx(0.25)
