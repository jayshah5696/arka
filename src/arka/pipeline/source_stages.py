from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from arka.pipeline.models import StageContext
from arka.pipeline.stages import Stage
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


class SeedSourceStage(Stage):
    name = "01_source"
    stage_action = "sourced"

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def run(
        self, records: list[ConversationRecord], ctx: StageContext
    ) -> list[ConversationRecord]:
        if records:
            return records
        source_path = (
            self.project_root / ctx.config.data_source.path
            if ctx.config.data_source.path is not None
            else self.project_root
        )
        config_hash = self._config_hash(ctx)
        if source_path.suffix == ".jsonl":
            return self._read_jsonl(source_path=source_path, config_hash=config_hash)
        if source_path.suffix == ".csv":
            return self._read_csv(source_path=source_path, config_hash=config_hash)
        raise ValueError(f"Unsupported seed source format: {source_path.suffix}")

    def _read_jsonl(
        self, source_path: Path, config_hash: str
    ) -> list[ConversationRecord]:
        records: list[ConversationRecord] = []
        with source_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                records.append(
                    self._build_conversation_record(
                        instruction=str(row["instruction"]),
                        response=str(row["response"]),
                        source_path=source_path,
                        config_hash=config_hash,
                    )
                )
        return records

    def _read_csv(
        self, source_path: Path, config_hash: str
    ) -> list[ConversationRecord]:
        records: list[ConversationRecord] = []
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                records.append(
                    self._build_conversation_record(
                        instruction=str(row["instruction"]),
                        response=str(row["response"]),
                        source_path=source_path,
                        config_hash=config_hash,
                    )
                )
        return records

    def _build_conversation_record(
        self,
        instruction: str,
        response: str,
        source_path: Path,
        config_hash: str,
    ) -> ConversationRecord:
        normalized_instruction = instruction.strip()
        normalized_response = response.strip()
        payload = ConversationPayload(
            instruction=normalized_instruction,
            response=normalized_response,
        )
        content_hash = self._content_hash(payload)
        record_id = self._record_id(payload)
        return ConversationRecord(
            id=record_id,
            content_hash=content_hash,
            source=RecordSource(
                type="seed",
                seed_file_hash=self._file_hash(source_path),
            ),
            lineage=RecordLineage(
                root_id=record_id,
                parent_ids=[],
            ),
            payload=payload,
            scores=RecordScores(),
            config_hash=config_hash,
            created_at=datetime.now(UTC).isoformat(),
        )

    def _content_hash(self, payload: ConversationPayload) -> str:
        return hashlib.sha256(
            payload.model_dump_json(exclude_none=True).encode("utf-8")
        ).hexdigest()

    def _record_id(self, payload: ConversationPayload) -> str:
        identity_payload = {
            "payload": payload.model_dump(mode="json", exclude_none=True),
            "lineage": {
                "parent_ids": [],
                "operator": None,
                "round": None,
                "depth": None,
            },
        }
        return hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    def _file_hash(self, source_path: Path) -> str:
        return hashlib.sha256(source_path.read_bytes()).hexdigest()

    def _config_hash(self, ctx: StageContext) -> str:
        return hashlib.sha256(
            json.dumps(ctx.config.model_dump(mode="json"), sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()
