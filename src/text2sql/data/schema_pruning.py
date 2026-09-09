from __future__ import annotations

import re

from .schema import DatabaseSchema, TableSchema

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> set[str]:
    expanded = text.replace("_", " ").casefold()
    return {token for token in _TOKEN_RE.findall(expanded) if len(token) > 1}


def _score_table(table: TableSchema, query_tokens: set[str]) -> int:
    score = 3 * len(_tokens(table.name) & query_tokens)
    for column in table.columns:
        score += 2 * len(_tokens(column.name) & query_tokens)
        if column.description:
            score += len(_tokens(column.description) & query_tokens)
    return score


def prune_schema(schema: DatabaseSchema, question: str, evidence: str | None, max_tables: int) -> DatabaseSchema:
    """Lexically rank tables, then keep FK-neighbors when space permits.

    This is an intentionally transparent baseline, not a learned schema linker.
    """

    if max_tables <= 0:
        raise ValueError("max_tables must be positive")
    if len(schema.tables) <= max_tables:
        return schema

    query_tokens = _tokens(f"{question} {evidence or ''}")
    ranked = sorted(schema.tables, key=lambda table: (-_score_table(table, query_tokens), table.name))
    selected_names = {table.name for table in ranked[:max_tables]}

    # Prefer replacing the lowest-ranked table with an FK neighbor of a strong match.
    by_name = {table.name: table for table in schema.tables}
    for table in ranked:
        if table.name not in selected_names:
            continue
        for foreign_key in table.foreign_keys:
            if foreign_key.target_table in by_name and foreign_key.target_table not in selected_names:
                weakest = next(
                    candidate for candidate in reversed(ranked) if candidate.name in selected_names
                )
                if weakest.name != table.name:
                    selected_names.remove(weakest.name)
                    selected_names.add(foreign_key.target_table)

    selected = tuple(table for table in schema.tables if table.name in selected_names)
    return DatabaseSchema(schema.db_id, selected)

