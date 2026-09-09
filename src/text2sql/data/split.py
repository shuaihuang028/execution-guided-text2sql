from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from .bird import BirdExample


@dataclass(frozen=True, slots=True)
class DatabaseSplit:
    train: tuple[BirdExample, ...]
    validation: tuple[BirdExample, ...]
    train_db_ids: tuple[str, ...]
    validation_db_ids: tuple[str, ...]


def split_by_database(
    examples: Sequence[BirdExample], validation_fraction: float = 0.1, seed: int = 42
) -> DatabaseSplit:
    """Create a deterministic split with no database schemas shared across partitions."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    db_ids = sorted({example.db_id for example in examples})
    if len(db_ids) < 2:
        raise ValueError("At least two databases are required for a grouped split")

    random.Random(seed).shuffle(db_ids)
    target = max(1, round(len(examples) * validation_fraction))
    counts = {db_id: sum(example.db_id == db_id for example in examples) for db_id in db_ids}
    validation_ids: set[str] = set()
    validation_count = 0
    for db_id in db_ids:
        if validation_count >= target and validation_ids:
            break
        validation_ids.add(db_id)
        validation_count += counts[db_id]

    if len(validation_ids) == len(db_ids):
        validation_ids.remove(db_ids[-1])
    train_ids = set(db_ids) - validation_ids
    train = tuple(example for example in examples if example.db_id in train_ids)
    validation = tuple(example for example in examples if example.db_id in validation_ids)
    return DatabaseSplit(
        train=train,
        validation=validation,
        train_db_ids=tuple(sorted(train_ids)),
        validation_db_ids=tuple(sorted(validation_ids)),
    )

