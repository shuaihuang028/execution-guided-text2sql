"""Verify the reference-model semantics and reward arithmetic of a DPO run.

`DPOTrainer(model=peft_model, ref_model=None)` does not mean "no reference
model". The behaviour depends on the TRL version and on whether the policy is a
PEFT model, so it has to be checked against the artifacts a run actually
produced rather than assumed from the argument name.

For the pinned stack in this repository (trl 1.12.0, peft 0.20.0), the trainer
snapshots the initial adapter into a frozen ``ref`` adapter beside each policy
checkpoint. The functions here assert the two invariants that make the run a
valid DPO experiment: the reference adapter must stay bit-identical to the SFT
adapter it started from, and the policy adapter must move away from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_ADAPTER_FILENAME = "adapter_model.safetensors"


def _normalize_key(key: str) -> str:
    """Strip PEFT wrapper prefixes so adapters saved in different roles align."""

    return key.replace("base_model.model.", "").replace(".default", "")


def load_adapter_tensors(path: str | Path) -> dict[str, object]:
    """Load one adapter's tensors, accepting either the file or its directory."""

    from safetensors.torch import load_file

    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / _ADAPTER_FILENAME
    if not candidate.is_file():
        raise FileNotFoundError(f"No adapter weights at {candidate}")
    return {_normalize_key(key): value for key, value in load_file(str(candidate)).items()}


def relative_drift(reference: dict, other: dict) -> float:
    """Relative L2 distance ``||other - reference|| / ||reference||`` over shared tensors."""

    shared = sorted(set(reference) & set(other))
    if not shared:
        raise ValueError("Adapters share no tensors; keys are incompatible")
    squared_difference = 0.0
    squared_norm = 0.0
    for key in shared:
        left = reference[key].float()
        right = other[key].float()
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch for {key}: {left.shape} vs {right.shape}")
        squared_difference += float((left - right).pow(2).sum())
        squared_norm += float(left.pow(2).sum())
    if squared_norm == 0.0:
        raise ValueError("Reference adapter has zero norm")
    return (squared_difference**0.5) / (squared_norm**0.5)


@dataclass(frozen=True, slots=True)
class ReferenceCheck:
    reference_drift: float
    policy_drift: float
    shared_tensors: int

    @property
    def reference_is_frozen(self) -> bool:
        return self.reference_drift == 0.0

    @property
    def policy_moved(self) -> bool:
        return self.policy_drift > 0.0

    @property
    def is_valid_dpo_run(self) -> bool:
        return self.reference_is_frozen and self.policy_moved

    def as_dict(self) -> dict[str, object]:
        return {
            "reference_drift": self.reference_drift,
            "policy_drift": self.policy_drift,
            "shared_tensors": self.shared_tensors,
            "reference_is_frozen": self.reference_is_frozen,
            "policy_moved": self.policy_moved,
            "is_valid_dpo_run": self.is_valid_dpo_run,
        }


def verify_reference_frozen(
    sft_adapter: str | Path, dpo_checkpoint: str | Path
) -> ReferenceCheck:
    """Check a DPO checkpoint's reference against the SFT adapter it started from."""

    sft = load_adapter_tensors(sft_adapter)
    checkpoint = Path(dpo_checkpoint)
    policy = load_adapter_tensors(checkpoint)
    reference = load_adapter_tensors(checkpoint / "ref")
    return ReferenceCheck(
        reference_drift=relative_drift(sft, reference),
        policy_drift=relative_drift(sft, policy),
        shared_tensors=len(set(sft) & set(policy)),
    )


def implicit_reward(policy_logp: float, reference_logp: float, beta: float) -> float:
    """DPO implicit reward ``beta * (log pi_theta - log pi_ref)`` for one completion."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    return beta * (policy_logp - reference_logp)


def reward_margin(
    chosen_policy_logp: float,
    chosen_reference_logp: float,
    rejected_policy_logp: float,
    rejected_reference_logp: float,
    beta: float,
) -> float:
    """Margin between the chosen and rejected implicit rewards."""

    return implicit_reward(chosen_policy_logp, chosen_reference_logp, beta) - implicit_reward(
        rejected_policy_logp, rejected_reference_logp, beta
    )


def check_reward_consistency(metrics: dict, *, tolerance: float = 1e-6) -> dict[str, object]:
    """Confirm TRL's reported rewards satisfy margin = chosen - rejected.

    Guards against silently misreading a metrics key, and catches a swapped
    chosen/rejected mapping, which would otherwise look like ordinary training.
    """

    def value(suffix: str) -> float:
        for key in (f"eval_rewards/{suffix}", f"rewards/{suffix}"):
            if key in metrics:
                return float(metrics[key])
        raise KeyError(f"Metrics contain no rewards/{suffix} entry")

    chosen = value("chosen")
    rejected = value("rejected")
    reported = value("margins")
    expected = chosen - rejected
    difference = abs(reported - expected)
    return {
        "rewards_chosen": chosen,
        "rewards_rejected": rejected,
        "reported_margin": reported,
        "expected_margin": expected,
        "absolute_difference": difference,
        "is_consistent": difference <= tolerance,
        "prefers_chosen": reported > 0.0,
    }
