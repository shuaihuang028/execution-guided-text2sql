from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
save_file = pytest.importorskip("safetensors.torch").save_file

from text2sql.training.dpo_verification import (
    check_reward_consistency,
    implicit_reward,
    load_adapter_tensors,
    relative_drift,
    reward_margin,
    verify_reference_frozen,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SFT_ADAPTER = REPOSITORY_ROOT / "artifacts" / "sft_qlora_final" / "final_adapter"
DPO_CHECKPOINT = REPOSITORY_ROOT / "artifacts" / "dpo_qlora_final" / "checkpoint-254"


def _write_adapter(directory: Path, tensors: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(directory / "adapter_model.safetensors"))
    return directory


def _base_tensors(scale: float = 1.0) -> dict:
    torch.manual_seed(0)
    return {
        "base_model.model.layers.0.q_proj.lora_A.weight.default": torch.ones(4, 8) * scale,
        "base_model.model.layers.0.q_proj.lora_B.weight.default": torch.ones(8, 4) * scale,
    }


def test_load_adapter_strips_peft_prefixes(tmp_path):
    _write_adapter(tmp_path / "a", _base_tensors())
    keys = set(load_adapter_tensors(tmp_path / "a"))
    assert keys == {
        "layers.0.q_proj.lora_A.weight",
        "layers.0.q_proj.lora_B.weight",
    }


def test_load_adapter_accepts_file_or_directory(tmp_path):
    directory = _write_adapter(tmp_path / "a", _base_tensors())
    from_dir = load_adapter_tensors(directory)
    from_file = load_adapter_tensors(directory / "adapter_model.safetensors")
    assert set(from_dir) == set(from_file)


def test_identical_adapters_have_zero_drift(tmp_path):
    tensors = _base_tensors()
    _write_adapter(tmp_path / "a", tensors)
    _write_adapter(tmp_path / "b", dict(tensors))
    left = load_adapter_tensors(tmp_path / "a")
    right = load_adapter_tensors(tmp_path / "b")
    assert relative_drift(left, right) == 0.0


def test_drift_is_relative_and_positive(tmp_path):
    _write_adapter(tmp_path / "a", _base_tensors(1.0))
    _write_adapter(tmp_path / "b", _base_tensors(1.5))
    left = load_adapter_tensors(tmp_path / "a")
    right = load_adapter_tensors(tmp_path / "b")
    assert relative_drift(left, right) == pytest.approx(0.5, rel=1e-6)


def test_drift_rejects_incompatible_adapters(tmp_path):
    _write_adapter(tmp_path / "a", _base_tensors())
    _write_adapter(tmp_path / "b", {"unrelated.weight": torch.ones(2, 2)})
    left = load_adapter_tensors(tmp_path / "a")
    right = load_adapter_tensors(tmp_path / "b")
    with pytest.raises(ValueError, match="share no tensors"):
        relative_drift(left, right)


def test_verify_reference_frozen_detects_a_valid_run(tmp_path):
    sft = _write_adapter(tmp_path / "sft", _base_tensors(1.0))
    checkpoint = _write_adapter(tmp_path / "dpo", _base_tensors(1.1))
    _write_adapter(checkpoint / "ref", _base_tensors(1.0))
    check = verify_reference_frozen(sft, checkpoint)
    assert check.reference_is_frozen
    assert check.policy_moved
    assert check.is_valid_dpo_run


def test_verify_reference_frozen_flags_a_drifting_reference(tmp_path):
    """A reference that moved would make the KL anchor invalid."""

    sft = _write_adapter(tmp_path / "sft", _base_tensors(1.0))
    checkpoint = _write_adapter(tmp_path / "dpo", _base_tensors(1.1))
    _write_adapter(checkpoint / "ref", _base_tensors(1.05))
    check = verify_reference_frozen(sft, checkpoint)
    assert not check.reference_is_frozen
    assert not check.is_valid_dpo_run


def test_verify_reference_frozen_flags_an_untrained_policy(tmp_path):
    sft = _write_adapter(tmp_path / "sft", _base_tensors(1.0))
    checkpoint = _write_adapter(tmp_path / "dpo", _base_tensors(1.0))
    _write_adapter(checkpoint / "ref", _base_tensors(1.0))
    check = verify_reference_frozen(sft, checkpoint)
    assert check.reference_is_frozen
    assert not check.policy_moved
    assert not check.is_valid_dpo_run


def test_implicit_reward_is_beta_scaled_log_ratio():
    assert implicit_reward(-2.0, -3.0, 0.1) == pytest.approx(0.1)
    assert implicit_reward(-3.0, -2.0, 0.1) == pytest.approx(-0.1)


def test_implicit_reward_requires_positive_beta():
    with pytest.raises(ValueError, match="beta must be positive"):
        implicit_reward(-1.0, -2.0, 0.0)


def test_reward_margin_matches_reward_difference():
    margin = reward_margin(-1.0, -2.0, -4.0, -3.0, 0.5)
    assert margin == pytest.approx(implicit_reward(-1.0, -2.0, 0.5) - implicit_reward(-4.0, -3.0, 0.5))
    assert margin == pytest.approx(1.0)


def test_reward_consistency_accepts_matching_metrics():
    report = check_reward_consistency(
        {
            "eval_rewards/chosen": -0.16,
            "eval_rewards/rejected": -0.35,
            "eval_rewards/margins": -0.16 - (-0.35),
        }
    )
    assert report["is_consistent"]
    assert report["prefers_chosen"]


def test_reward_consistency_flags_swapped_chosen_and_rejected():
    report = check_reward_consistency(
        {
            "rewards/chosen": -0.35,
            "rewards/rejected": -0.16,
            "rewards/margins": 0.19,
        }
    )
    assert not report["is_consistent"]


def test_reward_consistency_requires_the_keys():
    with pytest.raises(KeyError, match="rewards/margins"):
        check_reward_consistency({"rewards/chosen": 0.1, "rewards/rejected": 0.0})


@pytest.mark.skipif(
    not (SFT_ADAPTER.is_dir() and (DPO_CHECKPOINT / "ref").is_dir()),
    reason="Requires the completed SFT and DPO artifacts",
)
def test_completed_dpo_run_used_the_sft_model_as_reference():
    """The shipped DPO run must anchor on the SFT policy, not the pre-SFT base."""

    check = verify_reference_frozen(SFT_ADAPTER, DPO_CHECKPOINT)
    assert check.reference_is_frozen, "DPO reference adapter drifted from the SFT adapter"
    assert check.policy_moved, "DPO policy adapter never moved"
    assert check.shared_tensors == 392
