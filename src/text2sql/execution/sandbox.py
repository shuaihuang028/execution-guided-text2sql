from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import quote


class ExecutionStatus(StrEnum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    FORBIDDEN = "forbidden"
    TOO_MANY_ROWS = "too_many_rows"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    status: ExecutionStatus
    rows: tuple[tuple[object, ...], ...] = ()
    columns: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is ExecutionStatus.SUCCESS


_DENIED_ACTION_NAMES = (
    "SQLITE_ALTER_TABLE",
    "SQLITE_ANALYZE",
    "SQLITE_ATTACH",
    "SQLITE_CREATE_INDEX",
    "SQLITE_CREATE_TABLE",
    "SQLITE_CREATE_TEMP_INDEX",
    "SQLITE_CREATE_TEMP_TABLE",
    "SQLITE_CREATE_TEMP_TRIGGER",
    "SQLITE_CREATE_TEMP_VIEW",
    "SQLITE_CREATE_TRIGGER",
    "SQLITE_CREATE_VIEW",
    "SQLITE_DELETE",
    "SQLITE_DETACH",
    "SQLITE_DROP_INDEX",
    "SQLITE_DROP_TABLE",
    "SQLITE_DROP_TEMP_INDEX",
    "SQLITE_DROP_TEMP_TABLE",
    "SQLITE_DROP_TEMP_TRIGGER",
    "SQLITE_DROP_TEMP_VIEW",
    "SQLITE_DROP_TRIGGER",
    "SQLITE_DROP_VIEW",
    "SQLITE_INSERT",
    "SQLITE_PRAGMA",
    "SQLITE_REINDEX",
    "SQLITE_TRANSACTION",
    "SQLITE_UPDATE",
)
_DENIED_ACTIONS = {getattr(sqlite3, name) for name in _DENIED_ACTION_NAMES if hasattr(sqlite3, name)}


class SQLiteSandbox:
    """Execute one SQLite statement through defense-in-depth read-only controls."""

    def __init__(
        self,
        timeout_seconds: float = 3.0,
        max_rows: int = 100_000,
        progress_steps: int = 1_000,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_rows <= 0 or progress_steps <= 0:
            raise ValueError("max_rows and progress_steps must be positive")
        self.timeout_seconds = timeout_seconds
        self.max_rows = max_rows
        self.progress_steps = progress_steps

    @staticmethod
    def _uri(path: Path) -> str:
        absolute = path.resolve().as_posix()
        return f"file:{quote(absolute, safe='/:')}?mode=ro&immutable=1"

    def execute(self, database_path: str | Path, sql: str) -> ExecutionResult:
        started = time.perf_counter()
        deadline = started + self.timeout_seconds
        timed_out = False
        forbidden = False

        def progress_handler() -> int:
            nonlocal timed_out
            timed_out = time.perf_counter() >= deadline
            return int(timed_out)

        def authorizer(action: int, _arg1: str | None, _arg2: str | None, _db: str | None, _source: str | None) -> int:
            nonlocal forbidden
            if action in _DENIED_ACTIONS:
                forbidden = True
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        try:
            with sqlite3.connect(self._uri(Path(database_path)), uri=True) as connection:
                connection.execute("PRAGMA query_only = ON")
                connection.set_authorizer(authorizer)
                connection.set_progress_handler(progress_handler, self.progress_steps)
                cursor = connection.execute(sql)
                rows = cursor.fetchmany(self.max_rows + 1)
                if len(rows) > self.max_rows:
                    return ExecutionResult(
                        status=ExecutionStatus.TOO_MANY_ROWS,
                        elapsed_seconds=time.perf_counter() - started,
                        error=f"Query returned more than {self.max_rows} rows",
                    )
                columns = tuple(item[0] for item in (cursor.description or ()))
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    rows=tuple(tuple(row) for row in rows),
                    columns=columns,
                    elapsed_seconds=time.perf_counter() - started,
                )
        except sqlite3.Error as exc:
            if timed_out:
                status = ExecutionStatus.TIMEOUT
            elif forbidden or "not authorized" in str(exc).lower():
                status = ExecutionStatus.FORBIDDEN
            else:
                status = ExecutionStatus.ERROR
            return ExecutionResult(
                status=status,
                elapsed_seconds=time.perf_counter() - started,
                error=str(exc),
            )

