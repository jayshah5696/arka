from __future__ import annotations

import fcntl
import json
import sqlite3
from pathlib import Path
from types import TracebackType
from typing import TypeVar

from pydantic import BaseModel

from arka.recovery.models import RecoveryReport
from arka.recovery.sources import digest

T = TypeVar("T", bound=BaseModel)


def write_json(path: Path, value: object) -> None:
    """Replace a JSON artifact atomically after its contents have been flushed."""
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class Journal:
    """One writer per operation; commit each remote result before subsequent work."""

    def __init__(self, directory: Path, identity: dict, *, resume: bool):
        self.directory = directory
        self.identity = identity
        self.identity_hash = digest(identity)
        self._lock = None
        self.connection: sqlite3.Connection | None = None
        if directory.exists() and not resume:
            raise ValueError(
                f"Operation already exists: {directory.name}; use --resume"
            )
        if resume and not directory.exists():
            raise ValueError("Cannot resume an operation that does not exist")
        directory.mkdir(parents=True, exist_ok=resume)
        self._lock = (directory / ".lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = directory / "operation.json"
            if resume:
                saved = json.loads(path.read_text())
                if saved != identity:
                    raise ValueError(
                        "Operation inputs or configuration changed; incompatible resume"
                    )
            else:
                write_json(path, identity)
            self.connection = sqlite3.connect(directory / "recovery.sqlite3")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            (directory / "recovery.sqlite3").chmod(0o600)
        except BaseException:
            self.close()
            raise

    def get(self, key: str, model: type[T]) -> T | None:
        row = self.connection.execute(
            "SELECT value FROM results WHERE key = ?", (key,)
        ).fetchone()
        return model.model_validate_json(row[0]) if row else None

    def save(self, key: str, value: BaseModel) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO results VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value.model_dump_json()),
            )

    def values(self, prefix: str, model: type[T]) -> list[T]:
        rows = self.connection.execute(
            "SELECT value FROM results WHERE substr(key, 1, ?) = ? ORDER BY rowid",
            (len(prefix), prefix),
        ).fetchall()
        return [model.model_validate_json(row[0]) for row in rows]

    def report(self, report: RecoveryReport, *, ancestry_runs: list[str]) -> None:
        write_json(
            self.directory / "report/recovery_report.json",
            report.model_dump(mode="json"),
        )
        write_json(
            self.directory / "manifest.json",
            {
                "run_id": report.run_id,
                "operation": report.operation,
                "status": report.status,
                "parent_run_id": report.parent_run_id,
                "parent_stage": report.parent_stage,
                "audit_run_id": report.audit_run_id,
                "stage_names": ["rebuild"] if report.operation == "rebuild" else [],
                "ancestry_runs": ancestry_runs,
                "final_count": report.final_count,
                "dataset_path": report.dataset_path,
            },
        )

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
        if self._lock is not None:
            self._lock.close()

    def __enter__(self) -> Journal:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
