"""Deterministic subsetting of preference pairs for data-selection ablations.

Comparing a filtered preference subset against the full set confounds two
variables: the filter changed the *quality* of the pairs, but it also changed
how *many* there are. A size-matched random control separates them — it holds
the count fixed while leaving the original quality mix intact, so a difference
between the filtered arm and the control is attributable to the filter alone.

Selection uses the same seeded SHA-256 scoring as the train/validation split, so
a subset is reproducible from its seed without carrying an index file around.
"""

from __future__ import annotations

import hashlib


def _score(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def sample_preference_subset(
    records: list[dict], size: int, seed: int, *, key: str = "example_id"
) -> list[dict]:
    """Pick `size` records deterministically, preserving the input ordering.

    The subset depends only on (seed, size, the set of keys) — not on the order
    records arrive in — so it can be regenerated from the seed alone.
    """

    if size <= 0:
        raise ValueError("size must be positive")
    if size > len(records):
        raise ValueError(f"Cannot sample {size} records from {len(records)}")
    keys = [str(record[key]) for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Records contain duplicate {key} values")

    selected = set(sorted(keys, key=lambda value: _score(seed, value))[:size])
    return [record for record in records if str(record[key]) in selected]
