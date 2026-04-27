from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from arka.common.models import StrictModel
from arka.llm.client import LLMClient
from arka.llm.models import TokenUsage
from arka.pipeline.artifacts import StageArtifacts, StageReport
from arka.pipeline.evol_instruct import (
    build_evol_messages,
    build_response_messages,
    contains_refusal,
    levenshtein_distance,
    normalized_instruction,
)
from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.pipeline.stages import Stage
from arka.records.identity import (
    config_hash as compute_config_hash,
)
from arka.records.identity import (
    content_hash as compute_content_hash,
)
from arka.records.identity import (
    record_id as compute_record_id,
)
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    Record,
    RecordLineage,
    RecordScores,
    RecordSource,
)

logger = logging.getLogger(__name__)


class EvolvedInstruction(StrictModel):
    instruction: str


class EvolvedResponse(StrictModel):
    response: str


class EvolRawRow(StrictModel):
    parent_id: str
    round: int
    operator: str
    instruction_text: str | None = None
    response_text: str | None = None
    dropped_reason: str | None = None
    latency_ms_instruction: int | None = None
    latency_ms_response: int | None = None
    usage_instruction: TokenUsage | None = None
    usage_response: TokenUsage | None = None


class EvolInstructRoundStage(Stage):
    stage_action = "generated"

    def __init__(
        self,
        *,
        round_number: int,
        llm_client: Any | None = None,
        output_writer: OutputWriter | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.round_number = round_number
        self.name = f"{round_number + 2:02d}_evol_round_{round_number:02d}"
        self._llm_client = llm_client
        self._output_writer = output_writer or OutputWriter()
        self._project_root = project_root

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        frontier = self._frontier_records(records)
        if not frontier:
            self._write_artifacts(
                ctx=ctx,
                raw_rows=[],
                generated_records=[],
                dropped_records=[],
                frontier_count=0,
                total_out_count=len(records),
                drop_reasons={},
            )
            return records

        llm_client = self._llm_client or LLMClient(config=ctx.config.llm)
        operators = ctx.config.generator.operators
        branching_factor = ctx.config.generator.branching_factor or 1
        config_hash = self._config_hash(ctx)

        generated_records: list[Record] = []
        dropped_records: list[Record] = []
        raw_rows: list[EvolRawRow] = []
        drop_reasons: dict[str, int] = {}

        for parent_index, parent in enumerate(frontier):
            for branch_index in range(branching_factor):
                operator = operators[
                    (parent_index * branching_factor + branch_index) % len(operators)
                ]
                try:
                    instruction_output = llm_client.complete_structured(
                        messages=build_evol_messages(parent, operator=operator),
                        schema=EvolvedInstruction,
                        temperature=ctx.config.generator.temperature,
                        max_tokens=ctx.config.generator.max_tokens,
                    )
                    evolved_instruction = self._parse_instruction_output(
                        instruction_output.parsed,
                        instruction_output.text,
                    )
                    rejection_reason = self._rejection_reason(
                        parent=parent,
                        candidate_instruction=evolved_instruction,
                        ctx=ctx,
                    )
                    if rejection_reason is not None:
                        raw_rows.append(
                            EvolRawRow(
                                parent_id=parent.id,
                                round=self.round_number,
                                operator=operator,
                                instruction_text=evolved_instruction,
                                dropped_reason=rejection_reason,
                                latency_ms_instruction=instruction_output.latency_ms,
                                usage_instruction=instruction_output.usage,
                            )
                        )
                        dropped_records.append(
                            parent.dropped_by(
                                self.name,
                                rejection_reason,
                                f"operator={operator}; round={self.round_number}; "
                                f"candidate_instruction={evolved_instruction}",
                            )
                        )
                        drop_reasons[rejection_reason] = (
                            drop_reasons.get(rejection_reason, 0) + 1
                        )
                        continue

                    response_output = llm_client.complete_structured(
                        messages=build_response_messages(evolved_instruction),
                        schema=EvolvedResponse,
                        temperature=ctx.config.generator.temperature,
                        max_tokens=ctx.config.generator.max_tokens,
                    )
                    evolved_response = self._parse_response_output(
                        response_output.parsed,
                        response_output.text,
                    )
                except (ValueError, ValidationError):
                    reason_code = "evol_parse_failure"
                    raw_rows.append(
                        EvolRawRow(
                            parent_id=parent.id,
                            round=self.round_number,
                            operator=operator,
                            dropped_reason=reason_code,
                        )
                    )
                    dropped_records.append(
                        parent.dropped_by(
                            self.name,
                            reason_code,
                            f"operator={operator}; round={self.round_number}",
                        )
                    )
                    drop_reasons[reason_code] = drop_reasons.get(reason_code, 0) + 1
                    logger.warning(
                        "Dropping evol candidate for parent_id=%s due to parse failure",
                        parent.id,
                    )
                    continue

                raw_rows.append(
                    EvolRawRow(
                        parent_id=parent.id,
                        round=self.round_number,
                        operator=operator,
                        instruction_text=evolved_instruction,
                        response_text=evolved_response,
                        latency_ms_instruction=instruction_output.latency_ms,
                        latency_ms_response=response_output.latency_ms,
                        usage_instruction=instruction_output.usage,
                        usage_response=response_output.usage,
                    )
                )
                generated_records.append(
                    self._build_evolved_record(
                        parent=parent,
                        operator=operator,
                        instruction=evolved_instruction,
                        response=evolved_response,
                        config_hash=config_hash,
                    )
                )

        self._write_artifacts(
            ctx=ctx,
            raw_rows=raw_rows,
            generated_records=generated_records,
            dropped_records=dropped_records,
            frontier_count=len(frontier),
            total_out_count=len(records) + len(generated_records),
            drop_reasons=drop_reasons,
        )
        return [*records, *generated_records]

    def _frontier_records(self, records: list[Record]) -> list[ConversationRecord]:
        frontier: list[ConversationRecord] = []
        for record in records:
            if not isinstance(record, ConversationRecord):
                continue
            if self.round_number == 1 and record.source.type != "evolved":
                frontier.append(record)
                continue
            if (
                self.round_number > 1
                and record.source.type == "evolved"
                and record.lineage.round == self.round_number - 1
            ):
                frontier.append(record)
        return frontier

    def _parse_instruction_output(
        self,
        parsed: Any | None,
        text: str | None,
    ) -> str:
        if isinstance(parsed, EvolvedInstruction):
            return parsed.instruction.strip()
        if text is None:
            raise ValueError("Missing evolved instruction output")
        payload = EvolvedInstruction.model_validate_json(text)
        return payload.instruction.strip()

    def _parse_response_output(
        self,
        parsed: Any | None,
        text: str | None,
    ) -> str:
        if isinstance(parsed, EvolvedResponse):
            return parsed.response.strip()
        if text is None:
            raise ValueError("Missing evolved response output")
        payload = EvolvedResponse.model_validate_json(text)
        return payload.response.strip()

    def _rejection_reason(
        self,
        *,
        parent: ConversationRecord,
        candidate_instruction: str,
        ctx: StageContext,
    ) -> str | None:
        if normalized_instruction(candidate_instruction) == normalized_instruction(
            parent.payload.instruction
        ):
            return "evol_identical_to_parent"
        if contains_refusal(
            candidate_instruction,
            ctx.config.generator.filter.refusal_keywords,
        ):
            return "evol_refusal"
        if (
            len(candidate_instruction)
            < ctx.config.generator.filter.min_instruction_chars
        ):
            return "evol_instruction_too_short"
        if (
            levenshtein_distance(candidate_instruction, parent.payload.instruction)
            < ctx.config.generator.filter.min_edit_distance_chars
        ):
            return "evol_edit_distance_too_small"
        return None

    def _build_evolved_record(
        self,
        *,
        parent: ConversationRecord,
        operator: str,
        instruction: str,
        response: str,
        config_hash: str,
    ) -> ConversationRecord:
        payload = ConversationPayload(
            instruction=instruction,
            response=response,
        )
        lineage = RecordLineage(
            root_id=parent.lineage.root_id,
            parent_ids=[parent.id],
            operator=operator,
            round=self.round_number,
            depth=self.round_number,
        )
        return ConversationRecord(
            id=compute_record_id(payload, lineage),
            content_hash=compute_content_hash(payload),
            source=RecordSource(
                type="evolved",
                doc_id=parent.source.doc_id,
                chunk_id=parent.source.chunk_id,
                page_start=parent.source.page_start,
                page_end=parent.source.page_end,
                char_start=parent.source.char_start,
                char_end=parent.source.char_end,
                source_hash=parent.source.source_hash or parent.content_hash,
                seed_file_hash=parent.source.seed_file_hash,
            ),
            lineage=lineage,
            payload=payload,
            scores=RecordScores(),
            config_hash=config_hash,
            created_at=datetime.now(UTC).isoformat(),
        )

    def _write_artifacts(
        self,
        *,
        ctx: StageContext,
        raw_rows: list[EvolRawRow],
        generated_records: list[Record],
        dropped_records: list[Record],
        frontier_count: int,
        total_out_count: int,
        drop_reasons: dict[str, int],
    ) -> None:
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        rows_payload = [
            row.model_dump(mode="json", exclude_none=True) for row in raw_rows
        ]
        (ctx.work_dir / "raw_responses.jsonl").write_text(
            "".join(
                json.dumps(row, separators=(",", ":")) + "\n" for row in rows_payload
            ),
            encoding="utf-8",
        )
        costs: list[float] = []
        for row in raw_rows:
            for usage in (row.usage_instruction, row.usage_response):
                if usage is not None and usage.cost_usd is not None:
                    costs.append(usage.cost_usd)
        report_kwargs: dict[str, Any] = {
            "stage": self.name,
            "count_in": frontier_count,
            "count_out": total_out_count,
            "dropped_count": len(dropped_records),
            "drop_reasons": drop_reasons,
            "generated_count": len(generated_records),
            "cost_usd": round(sum(costs), 6) if costs else None,
        }
        StageArtifacts(ctx, writer=self._output_writer).write(
            report=StageReport(**report_kwargs),
            dropped=dropped_records,
        )

    def _config_hash(self, ctx: StageContext) -> str:
        return compute_config_hash(ctx.config)
