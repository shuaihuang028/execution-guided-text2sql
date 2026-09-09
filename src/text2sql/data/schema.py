from __future__ import annotations

import csv
import io
import json
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import quote


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    data_type: str
    nullable: bool
    primary_key_position: int = 0
    description: str | None = None


@dataclass(frozen=True, slots=True)
class ForeignKey:
    source_column: str
    target_table: str
    target_column: str | None


@dataclass(frozen=True, slots=True)
class TableSchema:
    name: str
    columns: tuple[Column, ...]
    foreign_keys: tuple[ForeignKey, ...] = ()


@dataclass(frozen=True, slots=True)
class DatabaseSchema:
    db_id: str
    tables: tuple[TableSchema, ...]


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _readonly_uri(path: Path) -> str:
    absolute = path.resolve().as_posix()
    return f"file:{quote(absolute, safe='/:')}?mode=ro"


def extract_sqlite_schema(path: str | Path, db_id: str | None = None) -> DatabaseSchema:
    """Extract tables, columns, primary keys, and foreign keys without mutating the DB."""

    database = Path(path)
    with sqlite3.connect(_readonly_uri(database), uri=True) as connection:
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        tables: list[TableSchema] = []
        for (table_name,) in table_rows:
            quoted = _quote_identifier(table_name)
            column_rows = connection.execute(f"PRAGMA table_info({quoted})").fetchall()
            columns = tuple(
                Column(
                    name=row[1],
                    data_type=row[2] or "UNKNOWN",
                    nullable=not bool(row[3]),
                    primary_key_position=int(row[5]),
                )
                for row in column_rows
            )
            foreign_key_rows = connection.execute(f"PRAGMA foreign_key_list({quoted})").fetchall()
            foreign_keys = tuple(
                ForeignKey(source_column=row[3], target_table=row[2], target_column=row[4])
                for row in foreign_key_rows
            )
            tables.append(TableSchema(table_name, columns, foreign_keys))
    return DatabaseSchema(db_id or database.stem, tuple(tables))


def load_column_descriptions(path: str | Path) -> dict[str, str]:
    """Load a BIRD database-description CSV using common column-name variants."""

    descriptions: dict[str, str] = {}
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Some official BIRD description files contain Windows smart punctuation.
        text = raw.decode("cp1252")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    for row in reader:
        original = (
            row.get("original_column_name")
            or row.get("column_name")
            or row.get("original column name")
            or ""
        ).strip()
        description = (
            row.get("column_description")
            or row.get("column description")
            or row.get("description")
            or ""
        ).strip()
        if original and description:
            descriptions[original.casefold()] = description
    return descriptions


def load_database_descriptions(database_directory: str | Path) -> dict[str, dict[str, str]]:
    """Load the per-table CSV files from a standard BIRD database directory."""

    description_root = Path(database_directory) / "database_description"
    if not description_root.is_dir():
        return {}
    return {
        csv_path.stem.casefold(): load_column_descriptions(csv_path)
        for csv_path in sorted(description_root.glob("*.csv"))
    }


def load_all_column_meanings_json(
    path: str | Path,
) -> dict[str, dict[str, dict[str, str]]]:
    """Load BIRD's compact JSON mapping into database/table/column levels."""

    with Path(path).open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise TypeError("Expected the column-meaning file to contain a JSON object")

    descriptions: dict[str, dict[str, dict[str, str]]] = {}
    for compound_key, description in values.items():
        parts = str(compound_key).split("|", maxsplit=2)
        if len(parts) != 3:
            continue
        db_name, table_name, column_name = (part.casefold() for part in parts)
        descriptions.setdefault(db_name, {}).setdefault(table_name, {})[column_name] = str(description)
    return descriptions


def load_column_meanings_json(
    path: str | Path, db_id: str
) -> dict[str, dict[str, str]]:
    """Load descriptions for one database from BIRD's compact JSON mapping."""

    return load_all_column_meanings_json(path).get(db_id.casefold(), {})


def attach_descriptions(
    schema: DatabaseSchema, descriptions_by_table: dict[str, dict[str, str]]
) -> DatabaseSchema:
    tables: list[TableSchema] = []
    for table in schema.tables:
        descriptions = descriptions_by_table.get(table.name.casefold(), {})
        columns = tuple(
            replace(column, description=descriptions.get(column.name.casefold()))
            for column in table.columns
        )
        tables.append(replace(table, columns=columns))
    return replace(schema, tables=tuple(tables))
