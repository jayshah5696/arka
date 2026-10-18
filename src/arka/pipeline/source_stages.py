from __future__ import annotations

import csv
import hashlib
import json
from bisect import bisect_left, bisect_right
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from arka.common.pdf_boilerplate import strip_boilerplate
from arka.common.security import sanitize_for_prompt
from arka.config.models import PDFSourceConfig, SeedSourceConfig
from arka.pipeline.artifacts import StageArtifacts, StageReport
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

    def __init__(
        self, project_root: Path, config: SeedSourceConfig | None = None
    ) -> None:
        self.project_root = project_root
        self.config = config

    def run(self, records: list[Record], ctx: StageContext) -> list[ConversationRecord]:
        self.config = self.get_stage_config(
            ctx, SeedSourceConfig, "data_source", SeedSourceConfig(path="")
        )
        if records:
            return [
                record for record in records if isinstance(record, ConversationRecord)
            ]
        if not self.config.path:
            raise ValueError(
                "data_source.path is required when data_source.type='seeds'"
            )
        source_path = self.project_root / self.config.path
        # DX: Provide a clear error message with the full expected path when the seed file is missing
        if not source_path.exists():
            raise FileNotFoundError(
                f"Seed source file not found at expected path: {source_path}"
            )
        # SECURITY: Enforce max seed file size (50MB) to prevent OOM DOS
        if source_path.stat().st_size > 50 * 1024 * 1024:
            raise ValueError("Seed file exceeds maximum allowed size of 50MB")
        config_hash_value = config_hash(ctx.config)
        if source_path.suffix == ".jsonl":
            return ctx.protect_records(
                self._read_jsonl(source_path=source_path, config_hash=config_hash_value)
            )
        if source_path.suffix == ".csv":
            return ctx.protect_records(
                self._read_csv(source_path=source_path, config_hash=config_hash_value)
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
                        reasoning=row.get("reasoning"),
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
                        reasoning=row.get("reasoning"),
                    )
                )
        return records

    def _build_conversation_record(
        self,
        instruction: str,
        response: str,
        source_path: Path,
        config_hash: str,
        reasoning: str | None = None,
    ) -> ConversationRecord:
        # SECURITY: Sanitize seed input by limiting length to mitigate prompt injection and context overflow
        normalized_instruction = sanitize_for_prompt(instruction.strip()[:16384])
        normalized_response = sanitize_for_prompt(response.strip()[:16384])
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError(f"{source_path}: seed reasoning must be a string or null")
        payload = ConversationPayload(
            instruction=normalized_instruction,
            response=normalized_response,
            reasoning=sanitize_for_prompt(reasoning.strip()[:16384]) or None
            if reasoning is not None
            else None,
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

    def __init__(
        self, project_root: Path, config: PDFSourceConfig | None = None
    ) -> None:
        self.project_root = project_root
        self.config = config

    def run(
        self,
        records: list[Record],
        ctx: StageContext,
    ) -> list[GroundedChunkRecord]:
        self.config = self.get_stage_config(
            ctx, PDFSourceConfig, "data_source", PDFSourceConfig(path="")
        )
        if records:
            return [
                record for record in records if isinstance(record, GroundedChunkRecord)
            ]
        if not self.config.path:
            raise ValueError("PDF source requires data_source.path")
        source_path = self.project_root / self.config.path
        if not source_path.exists():
            raise FileNotFoundError(
                f"PDF source file not found at expected path: {source_path}"
            )

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

        spans = []
        stripped_empty = False
        if self.config.boilerplate_strip:
            page_texts, spans = strip_boilerplate(
                page_texts, self.config.min_block_bytes, self.config.min_doc_frequency
            )
            remaining_bytes = sum(len(text.encode("utf-8")) for _, text in page_texts)
            stripped_empty = (
                remaining_bytes < self.config.min_remaining_bytes or not remaining_bytes
            )
            ctx.work_dir.mkdir(parents=True, exist_ok=True)
            (ctx.work_dir / "boilerplate_spans.jsonl").write_text(
                "".join(json.dumps(asdict(span)) + "\n" for span in spans),
                encoding="utf-8",
            )
            extracted_text = "\n\n".join(text for _, text in page_texts if text)
            if stripped_empty:
                extracted_text = ""

        # Retain original page numbers in the normalized joined text.
        page_numbers: list[int] = []
        page_starts: list[int] = []
        page_ends: list[int] = []
        offset = 0
        for page_number, page_text in page_texts:
            if not page_text:
                continue
            if page_numbers:
                offset += 2  # The separator between non-empty pages.
            page_numbers.append(page_number)
            page_starts.append(offset)
            offset += len(page_text)
            page_ends.append(offset)

        config_hash_value = config_hash(ctx.config)
        source_hash = file_hash(source_path)
        doc_id = source_path.stem
        chunk_size = self.config.chunk_size_chars
        overlap = self.config.chunk_overlap_chars
        step = chunk_size - overlap
        chunks: list[GroundedChunkRecord] = []

        for chunk_index, window_start in enumerate(range(0, len(extracted_text), step)):
            window = extracted_text[window_start : window_start + chunk_size]
            text = window.strip()
            if not text:
                continue
            char_start = window_start + len(window) - len(window.lstrip())
            char_end = char_start + len(text)
            page_start = page_numbers[bisect_right(page_ends, char_start)]
            page_end = page_numbers[bisect_left(page_starts, char_end) - 1]
            payload = GroundedChunkPayload(
                text=text,
                doc_id=doc_id,
                chunk_idx=chunk_index,
                page_start=page_start,
                page_end=page_end,
                char_start=char_start,
                char_end=char_end,
                word_count=len(text.split()),
                chunk_strategy=self.config.chunk_strategy or "fixed",
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

        chunks = ctx.protect_records(chunks)
        report_extras: dict[str, Any] = {}
        if self.config.boilerplate_strip:
            report_extras.update(
                boilerplate_bytes_removed=sum(s.end - s.start for s in spans),
                boilerplate_blocks_cut=len(spans),
                pages_affected=len({s.page for s in spans}),
                docs_stripped_empty=int(stripped_empty),
            )

        dropped_chunks = None
        count_in = 0
        if self.config.decision_filter.enabled:
            from arka.pipeline.decision_checks import filter_pdf_chunks
            from arka.pipeline.decision_stages import decision_usage

            original_count = len(chunks)
            chunks, dropped, results = filter_pdf_chunks(
                chunks, ctx, self.config.decision_filter, self.name
            )
            count_in = original_count
            dropped_chunks = dropped
            report_extras.update(
                dropped_count=len(dropped),
                drop_reasons={"pdf_front_back_matter": len(dropped)} if dropped else {},
                decision_usage=decision_usage(results),
            )

        if self.config.boilerplate_strip or self.config.decision_filter.enabled:
            StageArtifacts(ctx).write(
                report=StageReport(
                    stage=self.name,
                    count_in=count_in,
                    count_out=len(chunks),
                    **report_extras,
                ),
                dropped=dropped_chunks,
            )

        return chunks
