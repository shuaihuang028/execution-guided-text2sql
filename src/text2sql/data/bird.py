from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class BirdExample:
    """Normalized representation of a BIRD Text-to-SQL example."""

    example_id: str
    db_id: str
    question: str
    sql: str
    evidence: str | None = None
    difficulty: str | None = None


def _records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"Expected an object at {path}:{line_number}")
                yield value
        return

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise TypeError(f"Expected a JSON array in {path}")
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            raise TypeError(f"Expected an object at {path}[{index}]")
        yield record


def iter_bird_examples(path: str | Path) -> Iterator[BirdExample]:
    """Yield BIRD examples from either the official JSON or filtered JSONL format."""

    source = Path(path)
    for index, record in enumerate(_records(source)):
        try:
            db_id = str(record["db_id"]).strip()
            question = str(record["question"]).strip()
            sql = str(record.get("SQL", record.get("sql", ""))).strip()
        except KeyError as exc:
            raise ValueError(f"Missing required field {exc.args[0]!r} in record {index}") from exc
        if not db_id or not question or not sql:
            raise ValueError(f"Blank db_id, question, or SQL in record {index}")

        raw_id = record.get("question_id", record.get("id", index))
        evidence = str(record.get("evidence", "")).strip() or None
        difficulty = str(record.get("difficulty", "")).strip() or None
        yield BirdExample(
            example_id=f"{db_id}:{raw_id}",
            db_id=db_id,
            question=question,
            sql=sql,
            evidence=evidence,
            difficulty=difficulty,
        )


def load_bird_examples(path: str | Path) -> list[BirdExample]:
    return list(iter_bird_examples(path))


def database_path(database_root: str | Path, db_id: str) -> Path:
    """Resolve the standard BIRD database layout and fail on ambiguous/missing files."""

    root = Path(database_root)
    candidates = [root / db_id / f"{db_id}.sqlite", root / f"{db_id}.sqlite"]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) != 1:
        rendered = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Expected exactly one SQLite database for {db_id!r}: {rendered}")
    return existing[0]
