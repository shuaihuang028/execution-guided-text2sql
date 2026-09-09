from __future__ import annotations

import re

_FENCED_SQL = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sql(generation: str) -> str:
    """Normalize common model wrappers while leaving SQL semantics unchanged."""

    text = generation.strip()
    fenced = _FENCED_SQL.search(text)
    if fenced:
        text = fenced.group(1).strip()
    text = re.sub(r"^\s*(?:SQL|Answer)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()

