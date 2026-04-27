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
from arka.records.identity import config_hash, content_hash, file_hash, record_id
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
        # SECURITY: Enforce max seed file size (50MB) to prevent OOM DOS
        if source_path.stat().st_size > 50 * 1024 * 1024:
            raise ValueError("Seed file exceeds maximum allowed size of 50MB")
        config_hash_value = config_hash(ctx.config)
        if source_path.suffix == ".jsonl":
            return self._read_jsonl(
                source_path=source_path, config_hash=config_hash_value
            )
        if source_path.suffix == ".csv":
            return self._read_csv(
                source_path=source_path, config_hash=config_hash_value
            )
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
        # SECURITY: Sanitize seed input by limiting length to mitigate prompt injection and context overflow
        normalized_instruction = instruction.strip()[:16384]
        normalized_response = response.strip()[:16384]
        payload = ConversationPayload(
            instruction=normalized_instruction,
            response=normalized_response,
        )
        rid = record_id(payload)
        return ConversationRecord(
            id=rid,
            content_hash=content_hash(payload),
            source=RecordSource(
                type="seed",
                seed_file_hash=file_hash(source_path),
            ),
            lineage=RecordLineage(root_id=rid, parent_ids=[]),
            payload=payload,
            scores=RecordScores(),
            config_hash=config_hash,
            created_at=datetime.now(UTC).isoformat(),
        )


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

        config_hash_value = config_hash(ctx.config)
        source_hash = file_hash(source_path)
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
            # NOTE: PDF chunk ids historically use sha256(payload_json) directly,
            # not the (payload, lineage) record_id helper. Preserved for stable
            # ids across runs.
            chunk_id_hash = hashlib.sha256(
                payload.model_dump_json(exclude_none=True).encode("utf-8")
            ).hexdigest()
            chunks.append(
                GroundedChunkRecord(
                    id=chunk_id_hash,
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
                    lineage=RecordLineage(root_id=chunk_id_hash, parent_ids=[]),
                    payload=payload,
                    scores=RecordScores(),
                    config_hash=config_hash_value,
                    created_at=datetime.now(UTC).isoformat(),
                )
            )
        return chunks
