from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from text2sql.data.bird import database_path, load_bird_examples
from text2sql.data.schema import extract_sqlite_schema
from text2sql.execution.sandbox import ExecutionStatus, SQLiteSandbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate BIRD database coverage and gold SQL")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--gold-per-database", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=3.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_bird_examples(args.data_path)
    db_ids = sorted({example.db_id for example in examples})
    resolved: dict[str, Path] = {}
    missing: dict[str, str] = {}
    table_counts: dict[str, int] = {}
    for db_id in db_ids:
        try:
            resolved[db_id] = database_path(args.database_root, db_id)
            table_counts[db_id] = len(extract_sqlite_schema(resolved[db_id], db_id).tables)
        except (FileNotFoundError, OSError) as exc:
            missing[db_id] = str(exc)

    sandbox = SQLiteSandbox(timeout_seconds=args.timeout_seconds)
    selected_counts: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    for example in examples:
        if example.db_id not in resolved:
            continue
        if selected_counts[example.db_id] >= args.gold_per_database:
            continue
        selected_counts[example.db_id] += 1
        result = sandbox.execute(resolved[example.db_id], example.sql)
        statuses[result.status] += 1
        if result.status is not ExecutionStatus.SUCCESS:
            failures.append(
                {
                    "example_id": example.example_id,
                    "db_id": example.db_id,
                    "status": result.status,
                    "error": result.error or "",
                }
            )

    report = {
        "examples": len(examples),
        "database_ids": len(db_ids),
        "resolved_databases": len(resolved),
        "missing_databases": missing,
        "tables": {
            "total": sum(table_counts.values()),
            "min_per_database": min(table_counts.values()) if table_counts else 0,
            "max_per_database": max(table_counts.values()) if table_counts else 0,
        },
        "gold_smoke_test": {
            "per_database": args.gold_per_database,
            "executed": sum(statuses.values()),
            "status_counts": dict(statuses),
            "failures": failures,
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    if missing or failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
