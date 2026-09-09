from __future__ import annotations

from dataclasses import dataclass

from .bird import BirdExample
from .schema import DatabaseSchema


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


@dataclass(frozen=True, slots=True)
class PromptBuilder:
    include_evidence: bool = True
    include_descriptions: bool = True
    max_description_chars: int | None = None

    def _description(self, description: str) -> str:
        if self.max_description_chars is None or len(description) <= self.max_description_chars:
            return description
        return description[: self.max_description_chars].rstrip() + "…"

    def serialize_schema(self, schema: DatabaseSchema) -> str:
        lines: list[str] = []
        for table in schema.tables:
            column_parts: list[str] = []
            for column in table.columns:
                suffixes: list[str] = []
                if column.primary_key_position:
                    suffixes.append("PK")
                if self.include_descriptions and column.description:
                    suffixes.append(self._description(column.description))
                suffix = f" -- {'; '.join(suffixes)}" if suffixes else ""
                column_parts.append(
                    f"  {_quote_identifier(column.name)} {column.data_type}{suffix}"
                )
            lines.append(
                f"TABLE {_quote_identifier(table.name)} (\n"
                + ",\n".join(column_parts)
                + "\n)"
            )
            for foreign_key in table.foreign_keys:
                target_column = (
                    _quote_identifier(foreign_key.target_column)
                    if foreign_key.target_column is not None
                    else "<PRIMARY KEY>"
                )
                lines.append(
                    f"FOREIGN KEY {_quote_identifier(table.name)}."
                    f"{_quote_identifier(foreign_key.source_column)} -> "
                    f"{_quote_identifier(foreign_key.target_table)}."
                    f"{target_column}"
                )
        return "\n".join(lines)

    def build(self, example: BirdExample, schema: DatabaseSchema) -> str:
        messages = self.build_messages(example, schema)
        return "\n\n".join(message["content"] for message in messages)

    def build_messages(self, example: BirdExample, schema: DatabaseSchema) -> list[dict[str, str]]:
        """Return chat-template-ready messages for inference and later SFT formatting."""

        user_sections = [
            f"Database schema:\n{self.serialize_schema(schema)}",
            f"Question:\n{example.question}",
        ]
        if self.include_evidence and example.evidence:
            user_sections.append(f"Evidence:\n{example.evidence}")
        return [
            {
                "role": "system",
                "content": "Generate one SQLite query that answers the question. "
                "Copy table and column names exactly as shown, including identifier quotes. "
                "Return SQL only, without Markdown or explanation.",
            },
            {"role": "user", "content": "\n\n".join(user_sections)},
        ]
