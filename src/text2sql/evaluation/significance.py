"""Paired significance testing for two checkpoints scored on the same examples.

Two checkpoints evaluated on the same dev set produce *paired* binary outcomes,
so comparing their accuracies with an unpaired test throws away the pairing and
loses power. McNemar's test uses only the examples where the two disagree: under
the null hypothesis that neither checkpoint is better, each disagreement is a
fair coin flip.

The exact binomial form is used rather than the chi-square approximation, which
is unreliable when the discordant count is small — exactly the regime a 1,534
example dev set lands in when two checkpoints differ by a point or two.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import comb


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """Outcome of a paired comparison between a baseline and a candidate."""

    total: int
    baseline_correct: int
    candidate_correct: int
    only_baseline_correct: int
    """Discordant pairs the baseline won (b)."""
    only_candidate_correct: int
    """Discordant pairs the candidate won (c)."""
    p_value: float

    @property
    def discordant(self) -> int:
        return self.only_baseline_correct + self.only_candidate_correct

    @property
    def accuracy_delta(self) -> float:
        return (self.candidate_correct - self.baseline_correct) / self.total if self.total else 0.0

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "baseline_correct": self.baseline_correct,
            "candidate_correct": self.candidate_correct,
            "baseline_accuracy": self.baseline_correct / self.total if self.total else 0.0,
            "candidate_accuracy": self.candidate_correct / self.total if self.total else 0.0,
            "accuracy_delta": self.accuracy_delta,
            "only_baseline_correct": self.only_baseline_correct,
            "only_candidate_correct": self.only_candidate_correct,
            "discordant": self.discordant,
            "p_value": self.p_value,
            "significant_at_0.05": self.is_significant(),
        }


def exact_binomial_two_sided(successes: int, trials: int) -> float:
    """Two-sided exact binomial p-value against p=0.5.

    Doubling the smaller tail is the standard convention for McNemar's exact
    test and is what statistical packages report for this symmetric case.
    """

    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError("Require 0 <= successes <= trials")
    if trials == 0:
        return 1.0
    tail = min(successes, trials - successes)
    cumulative = sum(comb(trials, k) for k in range(tail + 1)) / (2**trials)
    return min(1.0, 2.0 * cumulative)


def mcnemar_test(
    baseline_correct: Iterable[str],
    candidate_correct: Iterable[str],
    scored_ids: Iterable[str],
) -> McNemarResult:
    """Compare two checkpoints over the identical set of scored examples."""

    scored = set(scored_ids)
    baseline = set(baseline_correct) & scored
    candidate = set(candidate_correct) & scored
    only_baseline = len(baseline - candidate)
    only_candidate = len(candidate - baseline)
    return McNemarResult(
        total=len(scored),
        baseline_correct=len(baseline),
        candidate_correct=len(candidate),
        only_baseline_correct=only_baseline,
        only_candidate_correct=only_candidate,
        p_value=exact_binomial_two_sided(only_candidate, only_baseline + only_candidate),
    )
