from __future__ import annotations

import sqlite3
from pathlib import Path


class CheckpointManager:
    def __init__(self, sqlite_path: Path) -> None:
        self.sqlite_path = sqlite_path
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    config_hash TEXT NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS stage_runs (
                    run_id TEXT NOT NULL,
                    stage_name TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    count_in INTEGER NOT NULL,
                    count_out INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    PRIMARY KEY (run_id, stage_name)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS artifact_index (
                    run_id TEXT NOT NULL,
                    stage_name TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    PRIMARY KEY (run_id, stage_name, artifact_path)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS generator_runs (
                    run_id TEXT NOT NULL,
                    stage_name TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    responses_path TEXT NOT NULL,
                    response_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    PRIMARY KEY (run_id, stage_name)
                )
                """
            )

    def register_run(self, run_id: str, config_hash: str, status: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (run_id, config_hash, status)
                VALUES (?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    config_hash = excluded.config_hash,
                    status = excluded.status
                """,
                (run_id, config_hash, status),
            )

    def save_stage(
        self,
        run_id: str,
        stage_name: str,
        artifact_path: Path,
        count_in: int,
        count_out: int,
        status: str,
    ) -> None:
        artifact_path_str = str(artifact_path)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO stage_runs (run_id, stage_name, artifact_path, count_in, count_out, status)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, stage_name) DO UPDATE SET
                    artifact_path = excluded.artifact_path,
                    count_in = excluded.count_in,
                    count_out = excluded.count_out,
                    status = excluded.status
                """,
                (run_id, stage_name, artifact_path_str, count_in, count_out, status),
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO artifact_index (run_id, stage_name, artifact_path)
                VALUES (?, ?, ?)
                """,
                (run_id, stage_name, artifact_path_str),
            )

    def load_stage(self, run_id: str, stage_name: str) -> dict[str, str | int] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT run_id, stage_name, artifact_path, count_in, count_out, status
                FROM stage_runs
                WHERE run_id = ? AND stage_name = ?
                """,
                (run_id, stage_name),
            ).fetchone()
        if row is None:
            return None
        return {
            "run_id": str(row["run_id"]),
            "stage_name": str(row["stage_name"]),
            "artifact_path": str(row["artifact_path"]),
            "count_in": int(row["count_in"]),
            "count_out": int(row["count_out"]),
            "status": str(row["status"]),
        }

    def save_generator(
        self,
        run_id: str,
        stage_name: str,
        prompt_hash: str,
        responses_path: Path,
        response_count: int,
        status: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO generator_runs (
                    run_id, stage_name, prompt_hash, responses_path, response_count, status
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, stage_name) DO UPDATE SET
                    prompt_hash = excluded.prompt_hash,
                    responses_path = excluded.responses_path,
                    response_count = excluded.response_count,
                    status = excluded.status
                """,
                (
                    run_id,
                    stage_name,
                    prompt_hash,
                    str(responses_path),
                    response_count,
                    status,
                ),
            )

    def load_generator(
        self, run_id: str, stage_name: str
    ) -> dict[str, str | int] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT run_id, stage_name, prompt_hash, responses_path, response_count, status
                FROM generator_runs
                WHERE run_id = ? AND stage_name = ?
                """,
                (run_id, stage_name),
            ).fetchone()
        if row is None:
            return None
        return {
            "run_id": str(row["run_id"]),
            "stage_name": str(row["stage_name"]),
            "prompt_hash": str(row["prompt_hash"]),
            "responses_path": str(row["responses_path"]),
            "response_count": int(row["response_count"]),
            "status": str(row["status"]),
        }

    def update_run_status(self, run_id: str, status: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET status = ? WHERE run_id = ?",
                (status, run_id),
            )

    def list_stage_runs(self, run_id: str) -> list[dict[str, str | int]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id, stage_name, artifact_path, count_in, count_out, status FROM stage_runs WHERE run_id = ? ORDER BY stage_name",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_runs(self) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id, config_hash, status FROM runs ORDER BY run_id"
            ).fetchall()
        return [dict(row) for row in rows]
