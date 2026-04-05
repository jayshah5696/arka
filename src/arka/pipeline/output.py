from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from arka.records.models import Record, StageEvent, record_model_for_name


class OutputWriter:
    def write_jsonl(self, records: list[Record], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record.export_payload(), separators=(",", ":")) + "\n"
                )
        return path

    def write_parquet(self, records: list[Record], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pl.DataFrame(
            [self.storage_row_for_record(record) for record in records],
            schema=self._storage_schema(),
        )
        frame.write_parquet(path)
        return path

    def write_dropped_parquet(self, records: list[Record], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pl.DataFrame(
            [self._dropped_storage_row(record) for record in records],
            schema=self._dropped_storage_schema(),
        )
        frame.write_parquet(path)
        return path

    def read_parquet(self, path: Path) -> list[Record]:
        frame = pl.read_parquet(path)
        return [self._storage_row_to_record(record) for record in frame.to_dicts()]

    def storage_row_for_record(self, record: Record) -> dict[str, Any]:
        return {
            "record_type": record.record_type,
            "id": record.id,
            "content_hash": record.content_hash,
            "source_json": json.dumps(
                record.source.model_dump(mode="json"), separators=(",", ":")
            ),
            "lineage_json": json.dumps(
                record.lineage.model_dump(mode="json"), separators=(",", ":")
            ),
            "payload_json": json.dumps(record.export_payload(), separators=(",", ":")),
            "scores_json": json.dumps(
                record.scores.model_dump(mode="json"), separators=(",", ":")
            ),
            "stage_events_json": json.dumps(
                [event.model_dump(mode="json") for event in record.stage_events],
                separators=(",", ":"),
            ),
            "config_hash": record.config_hash,
            "created_at": record.created_at,
        }

    def _dropped_storage_row(self, record: Record) -> dict[str, Any]:
        last_event = self._last_stage_event(record)
        return {
            **self.storage_row_for_record(record),
            "drop_stage": last_event.stage if last_event is not None else None,
            "drop_reason": last_event.reason_code if last_event is not None else None,
            "drop_detail": last_event.details if last_event is not None else None,
        }

    def _last_stage_event(self, record: Record) -> StageEvent | None:
        if not record.stage_events:
            return None
        return record.stage_events[-1]

    def _storage_schema(self) -> dict[str, pl.DataType]:
        return {
            "record_type": pl.String,
            "id": pl.String,
            "content_hash": pl.String,
            "source_json": pl.String,
            "lineage_json": pl.String,
            "payload_json": pl.String,
            "scores_json": pl.String,
            "stage_events_json": pl.String,
            "config_hash": pl.String,
            "created_at": pl.String,
        }

    def _dropped_storage_schema(self) -> dict[str, pl.DataType]:
        return {
            **self._storage_schema(),
            "drop_stage": pl.String,
            "drop_reason": pl.String,
            "drop_detail": pl.String,
        }

    def _storage_row_to_record(self, row: dict[str, Any]) -> Record:
        record_model = record_model_for_name(str(row["record_type"]))
        payload = json.loads(str(row["payload_json"]))
        return record_model.model_validate(
            {
                "id": row["id"],
                "content_hash": row["content_hash"],
                "source": json.loads(str(row["source_json"])),
                "lineage": json.loads(str(row["lineage_json"])),
                "payload": payload,
                "scores": json.loads(str(row["scores_json"])),
                "stage_events": json.loads(str(row["stage_events_json"])),
                "config_hash": row["config_hash"],
                "created_at": row["created_at"],
            }
        )
