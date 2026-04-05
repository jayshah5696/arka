from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from arka.pipeline.models import StageContext
from arka.pipeline.stages import Stage
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    GroundedChunkPayload,
    GroundedChunkRecord,
    Record,
    RecordLineage,
    RecordScores,
    RecordSource,
)


class SeedSourceStage(Stage):
    name = "01_source"
    stage_action = "sourced"

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def run(self, records: list[Record], ctx: StageContext) -> list[ConversationRecord]:
        if records:
            return [
                record for record in records if isinstance(record, ConversationRecord)
            ]
        if ctx.config.data_source.path is None:
            raise ValueError(
                "data_source.path is required when data_source.type='seeds'"
            )
        source_path = self.project_root / ctx.config.data_source.path
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
            lineage=RecordLineage(root_id=record_id, parent_ids=[]),
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


class PDFSourceStage(Stage):
    name = "01_source"
    stage_action = "sourced"

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def run(
        self,
        records: list[Record],
        ctx: StageContext,
    ) -> list[GroundedChunkRecord]:
        if records:
            return [
                record for record in records if isinstance(record, GroundedChunkRecord)
            ]
        if ctx.config.data_source.path is None:
            raise ValueError("PDF source requires data_source.path")
        source_path = self.project_root / ctx.config.data_source.path
        if not source_path.exists():
            raise ValueError(f"PDF source path does not exist: {source_path}")

        try:
            reader = PdfReader(str(source_path))
        except PdfReadError as exc:
            raise ValueError(
                "PDF extraction produced no text; scanned or empty PDFs are not supported"
            ) from exc
        page_texts: list[tuple[int, str]] = []
        for page_index, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            page_texts.append((page_index, page_text))

        extracted_text = "\n\n".join(text for _, text in page_texts if text)
        if not extracted_text.strip():
            raise ValueError(
                "PDF extraction produced no text; scanned or empty PDFs are not supported"
            )

        config_hash = self._config_hash(ctx)
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        doc_id = source_path.stem
        chunk_size = ctx.config.data_source.chunk_size_chars or 3000
        overlap = ctx.config.data_source.chunk_overlap_chars or 300
        step = chunk_size - overlap
        chunks: list[GroundedChunkRecord] = []

        for chunk_index, char_start in enumerate(range(0, len(extracted_text), step)):
            text = extracted_text[char_start : char_start + chunk_size].strip()
            if not text:
                continue
            char_end = char_start + len(text)
            page_start = 1
            page_end = len(page_texts)
            payload = GroundedChunkPayload(
                text=text,
                doc_id=doc_id,
                chunk_idx=chunk_index,
                page_start=page_start,
                page_end=page_end,
                char_start=char_start,
                char_end=char_end,
                word_count=len(text.split()),
                chunk_strategy=ctx.config.data_source.chunk_strategy or "fixed",
            )
            record_id = hashlib.sha256(
                payload.model_dump_json(exclude_none=True).encode("utf-8")
            ).hexdigest()
            chunks.append(
                GroundedChunkRecord(
                    id=record_id,
                    content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    source=RecordSource(
                        type="pdf_chunk",
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}:{chunk_index}",
                        page_start=page_start,
                        page_end=page_end,
                        char_start=char_start,
                        char_end=char_end,
                        source_hash=source_hash,
                    ),
                    lineage=RecordLineage(root_id=record_id, parent_ids=[]),
                    payload=payload,
                    scores=RecordScores(),
                    config_hash=config_hash,
                    created_at=datetime.now(UTC).isoformat(),
                )
            )
        return chunks

    def _config_hash(self, ctx: StageContext) -> str:
        return hashlib.sha256(
            json.dumps(ctx.config.model_dump(mode="json"), sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()
