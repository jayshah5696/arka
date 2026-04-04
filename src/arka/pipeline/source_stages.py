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
        if source_path.suffix == ".jsonl":
            return self._read_jsonl(
                source_path=source_path, config_hash=self._config_hash(ctx)
            )
        if source_path.suffix == ".csv":
            return self._read_csv(
                source_path=source_path, config_hash=self._config_hash(ctx)
            )
        raise ValueError(f"Unsupported seed source format: {source_path.suffix}")

    def _read_jsonl(
        self, source_path: Path, config_hash: str
    ) -> list[ConversationRecord]:
        records: list[ConversationRecord] = []
        with source_path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                row = json.loads(line)
                records.append(
                    self._build_conversation_record(
                        instruction=str(row["instruction"]),
                        response=str(row["response"]),
                        row_index=index,
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
            for index, row in enumerate(reader, start=1):
                records.append(
                    self._build_conversation_record(
                        instruction=str(row["instruction"]),
                        response=str(row["response"]),
                        row_index=index,
                        source_path=source_path,
                        config_hash=config_hash,
                    )
                )
        return records

    def _build_conversation_record(
        self,
        instruction: str,
        response: str,
        row_index: int,
        source_path: Path,
        config_hash: str,
    ) -> ConversationRecord:
        normalized_instruction = instruction.strip()
        normalized_response = response.strip()
        payload = ConversationPayload(
            instruction=normalized_instruction,
            response=normalized_response,
        )
        content_hash = hashlib.sha256(
            f"{normalized_instruction}\n{normalized_response}".encode()
        ).hexdigest()[:16]
        return ConversationRecord(
            id=f"seed-{row_index}",
            content_hash=content_hash,
            source=RecordSource(
                type="seed",
                seed_file_hash=self._file_hash(source_path),
            ),
            lineage=RecordLineage(
                root_id=f"seed-{row_index}",
                parent_ids=[],
            ),
            payload=payload,
            scores=RecordScores(),
            config_hash=config_hash,
            created_at=datetime.now(UTC).isoformat(),
        )

    def _file_hash(self, source_path: Path) -> str:
        return hashlib.sha256(source_path.read_bytes()).hexdigest()[:16]

    def _config_hash(self, ctx: StageContext) -> str:
        return hashlib.sha256(
            json.dumps(ctx.config.model_dump(mode="json"), sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()[:16]
