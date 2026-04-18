from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import cycle, islice
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from arka.common.models import StrictModel
from arka.config.models import GeneratorConfig, LLMConfig, resolve_llm_override
from arka.llm.client import LLMClient
from arka.llm.models import LLMOutput, TokenUsage
from arka.pipeline.checkpoint import CheckpointManager
from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.pipeline.stages import Stage
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    GroundedChunkRecord,
    Record,
    RecordLineage,
    RecordScores,
    RecordSource,
    StageEvent,
)

__all__ = [
    "PromptBasedGeneratorStage",
    "TransformGeneratorStage",
    "compute_prompt_hash",
]

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_TEMPLATE = """You generate synthetic instruction-response pairs for supervised fine-tuning.
Create one new instruction and one strong response inspired by the seed example.
The new pair must be self-contained, specific, and meaningfully different from the seed.
Return only JSON with keys \"instruction\" and \"response\".

Seed instruction:
{seed_instruction}

Seed response:
{seed_response}
"""

_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class GeneratedConversation(StrictModel):
    instruction: str
    response: str


class RawGeneratorResponse(StrictModel):
    plan_index: int
    seed_id: str
    generated_text: str
    prompt_hash: str
    model: str
    latency_ms: int
    usage: TokenUsage


@dataclass(frozen=True)
class GenerationPlanItem:
    plan_index: int
    seed_record: ConversationRecord | GroundedChunkRecord


def compute_prompt_hash(generator: GeneratorConfig, llm: LLMConfig) -> str:
    prompt_identity = {
        "generator": generator.model_dump(mode="json", exclude_none=True),
        "llm": {
            "provider": llm.provider,
            "model": llm.model,
            "base_url": str(llm.base_url),
        },
    }
    payload = json.dumps(prompt_identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PromptBasedGeneratorStage(Stage):
    name = "02_generate"
    stage_action = "generated"

    def __init__(
        self,
        llm_client: Any | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        output_writer: OutputWriter | None = None,
        project_root: Path | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._checkpoint = checkpoint_manager
        self._output_writer = output_writer or OutputWriter()
        self._project_root = project_root

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        seed_records = [
            record
            for record in records
            if isinstance(record, ConversationRecord | GroundedChunkRecord)
        ]
        if not seed_records:
            return []

        generation_plan = self._generation_plan(seed_records, ctx.config.generator)
        if not generation_plan:
            return []

        prompt_hash = compute_prompt_hash(ctx.config.generator, ctx.config.llm)
        responses_path = ctx.work_dir / "raw_responses.jsonl"
        records_path = ctx.work_dir / "data.parquet"

        cached = self._checkpoint_manager(ctx).load_generator(ctx.run_id, self.name)
        if cached is not None and cached["prompt_hash"] == prompt_hash:
            if records_path.exists():
                return self._output_writer.read_parquet(records_path)
            if responses_path.exists():
                raw = self._load_raw_responses(responses_path)
                if len(raw) >= len(generation_plan):
                    return self._parse_responses(raw, generation_plan, ctx)

        return self._generate(generation_plan, ctx, prompt_hash, responses_path)

    def _generation_plan(
        self,
        seed_records: list[ConversationRecord | GroundedChunkRecord],
        config: GeneratorConfig,
    ) -> list[GenerationPlanItem]:
        requested_count = config.target_count * config.generation_multiplier
        if requested_count <= 0:
            return []
        planned_seeds = islice(cycle(seed_records), requested_count)
        return [
            GenerationPlanItem(plan_index=index, seed_record=seed_record)
            for index, seed_record in enumerate(planned_seeds)
        ]

    def _generate(
        self,
        plan: list[GenerationPlanItem],
        ctx: StageContext,
        prompt_hash: str,
        responses_path: Path,
    ) -> list[Record]:
        checkpoint = self._checkpoint_manager(ctx)
        existing_rows = self._existing_rows_for_prompt(responses_path, prompt_hash)
        if not existing_rows and responses_path.exists():
            responses_path.unlink()

        done_plan_indices = {row.plan_index for row in existing_rows}
        pending = [item for item in plan if item.plan_index not in done_plan_indices]

        checkpoint.save_generator(
            run_id=ctx.run_id,
            stage_name=self.name,
            prompt_hash=prompt_hash,
            responses_path=responses_path,
            response_count=len(existing_rows),
            status="running",
        )

        llm_client = self._llm_client or LLMClient(config=ctx.config.llm)
        responses_path.parent.mkdir(parents=True, exist_ok=True)

        written_count = len(existing_rows)
        with responses_path.open("a", encoding="utf-8") as handle:
            for item in pending:
                output = self._complete_raw(llm_client, item.seed_record, ctx)
                row = RawGeneratorResponse(
                    plan_index=item.plan_index,
                    seed_id=item.seed_record.id,
                    generated_text=self._generated_text_from_output(output),
                    prompt_hash=prompt_hash,
                    model=output.model,
                    latency_ms=output.latency_ms,
                    usage=output.usage,
                )
                handle.write(
                    json.dumps(row.model_dump(mode="json"), separators=(",", ":"))
                    + "\n"
                )
                handle.flush()
                written_count += 1
                checkpoint.save_generator(
                    run_id=ctx.run_id,
                    stage_name=self.name,
                    prompt_hash=prompt_hash,
                    responses_path=responses_path,
                    response_count=written_count,
                    status="running",
                )

        raw = self._load_raw_responses(responses_path)
        records = self._parse_responses(raw, plan, ctx)
        checkpoint.save_generator(
            run_id=ctx.run_id,
            stage_name=self.name,
            prompt_hash=prompt_hash,
            responses_path=responses_path,
            response_count=len(raw),
            status="completed",
        )
        return records

    def _complete_raw(
        self,
        llm_client: Any,
        seed_record: ConversationRecord | GroundedChunkRecord,
        ctx: StageContext,
    ) -> LLMOutput:
        messages = self._messages_for_seed(seed_record, ctx.config.generator)
        kwargs = {
            "messages": messages,
            "schema": GeneratedConversation,
            "temperature": ctx.config.generator.temperature,
            "max_tokens": ctx.config.generator.max_tokens,
        }
        try:
            return llm_client.complete_structured(**kwargs)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            kwargs.pop("temperature", None)
            kwargs.pop("max_tokens", None)
            return llm_client.complete_structured(**kwargs)

    def _messages_for_seed(
        self,
        seed_record: ConversationRecord | GroundedChunkRecord,
        config: GeneratorConfig,
    ) -> list[dict[str, str]]:
        if isinstance(seed_record, ConversationRecord):
            content = config.prompt_template.format(
                seed_instruction=seed_record.payload.instruction,
                seed_response=seed_record.payload.response,
            )
        else:
            content = (
                "You generate grounded instruction-response pairs from a document chunk.\n"
                "Create one self-contained instruction answerable from the chunk, and one grounded response.\n"
                "Avoid referring to 'the passage above'.\n"
                'Return only JSON with keys "instruction" and "response".\n\n'
                f"Document chunk:\n{seed_record.payload.text}\n"
            )
        return [{"role": "user", "content": content}]

    def _existing_rows_for_prompt(
        self, responses_path: Path, prompt_hash: str
    ) -> list[RawGeneratorResponse]:
        if not responses_path.exists():
            return []
        rows = self._load_raw_responses(responses_path)
        if not rows:
            return []
        if all(row.prompt_hash == prompt_hash for row in rows):
            return rows
        return []

    def _load_raw_responses(self, responses_path: Path) -> list[RawGeneratorResponse]:
        rows: list[RawGeneratorResponse] = []
        lines = responses_path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    continue
                raise
            rows.append(RawGeneratorResponse.model_validate(payload))
        rows.sort(key=lambda row: row.plan_index)
        return rows

    def _parse_responses(
        self,
        raw_rows: list[RawGeneratorResponse],
        plan: list[GenerationPlanItem],
        ctx: StageContext,
    ) -> list[Record]:
        plan_by_index = {item.plan_index: item for item in plan}
        config_hash = self._config_hash(ctx)
        generated_records: list[Record] = []
        dropped_records: list[Record] = []
        drop_reasons: dict[str, int] = {}

        for row in raw_rows:
            item = plan_by_index.get(row.plan_index)
            if item is None:
                continue
            try:
                payload = self._parse_generated_payload(row.generated_text)
            except ValueError as exc:
                reason_code = "generator_parse_failure"
                drop_reasons[reason_code] = drop_reasons.get(reason_code, 0) + 1
                details = (
                    f"plan_index={row.plan_index}; seed_id={row.seed_id}; "
                    f"error={exc}; generated_text={row.generated_text}"
                )
                logger.warning(
                    "Skipping malformed generator response for plan_index=%s: %s",
                    row.plan_index,
                    exc,
                )
                dropped_records.append(
                    self._drop_record(
                        record=item.seed_record,
                        reason_code=reason_code,
                        details=details,
                    )
                )
                continue
            generated_records.append(
                self._build_generated_record(
                    payload=payload,
                    parent=item.seed_record,
                    config_hash=config_hash,
                )
            )

        self._write_parse_artifacts(
            ctx=ctx,
            raw_rows=raw_rows,
            attempted_count=len(plan),
            generated_records=generated_records,
            dropped_records=dropped_records,
            drop_reasons=drop_reasons,
        )
        return generated_records

    def _parse_generated_payload(self, text: str) -> GeneratedConversation:
        try:
            payload = json.loads(self._extract_json_text(text))
            parsed = GeneratedConversation.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise ValueError(
                f"Generator output did not parse into GeneratedConversation: {exc}"
            ) from exc
        return GeneratedConversation(
            instruction=parsed.instruction.strip(),
            response=parsed.response.strip(),
        )

    def _extract_json_text(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            match = _JSON_FENCE_PATTERN.search(stripped)
            if match is None:
                raise ValueError("Could not extract JSON from code fence")
            return match.group(1)
        json_object_match = _JSON_OBJECT_PATTERN.search(stripped)
        if json_object_match is not None:
            return json_object_match.group(0)
        return stripped

    def _generated_text_from_output(self, output: LLMOutput) -> str:
        if output.text is not None:
            return output.text.strip()
        if output.parsed is not None:
            return output.parsed.model_dump_json()
        raise ValueError("Generator structured output returned neither text nor parsed")

    def _build_generated_record(
        self,
        payload: GeneratedConversation,
        parent: ConversationRecord | GroundedChunkRecord,
        config_hash: str,
    ) -> ConversationRecord:
        conversation_payload = ConversationPayload(
            instruction=payload.instruction,
            response=payload.response,
        )
        content_hash = self._content_hash(conversation_payload)
        lineage = RecordLineage(
            root_id=parent.lineage.root_id,
            parent_ids=[parent.id],
            operator="prompt_based",
            round=1,
            depth=1,
        )
        source = RecordSource(
            type="generated",
            seed_file_hash=parent.source.seed_file_hash,
            source_hash=parent.content_hash,
            doc_id=parent.source.doc_id,
            chunk_id=parent.source.chunk_id,
            page_start=parent.source.page_start,
            page_end=parent.source.page_end,
            char_start=parent.source.char_start,
            char_end=parent.source.char_end,
        )
        if isinstance(parent, GroundedChunkRecord):
            source.type = "generated"
        record_id = self._record_id(conversation_payload, lineage)
        return ConversationRecord(
            id=record_id,
            content_hash=content_hash,
            source=source,
            lineage=lineage,
            payload=conversation_payload,
            scores=RecordScores(),
            config_hash=config_hash,
            created_at=datetime.now(UTC).isoformat(),
        )

    def _drop_record(
        self,
        record: Record,
        reason_code: str,
        details: str,
    ) -> Record:
        return record.model_copy(
            update={
                "stage_events": [
                    *record.stage_events,
                    StageEvent(
                        stage=self.name,
                        action="dropped",
                        reason_code=reason_code,
                        details=details,
                        seq=len(record.stage_events) + 1,
                    ),
                ]
            }
        )

    def _write_parse_artifacts(
        self,
        *,
        ctx: StageContext,
        raw_rows: list[RawGeneratorResponse],
        attempted_count: int,
        generated_records: list[Record],
        dropped_records: list[Record],
        drop_reasons: dict[str, int],
    ) -> None:
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        self._output_writer.write_dropped_parquet(
            records=dropped_records,
            path=ctx.work_dir / "dropped.parquet",
        )
        costs = [
            row.usage.cost_usd for row in raw_rows if row.usage.cost_usd is not None
        ]
        total_cost = round(sum(costs), 6) if costs else None
        stats = {
            "stage": self.name,
            "count_in": attempted_count,
            "count_out": len(generated_records),
            "dropped_count": len(dropped_records),
            "drop_reasons": drop_reasons,
            "cost_usd": total_cost,
        }
        (ctx.work_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    def _content_hash(self, payload: ConversationPayload) -> str:
        return hashlib.sha256(
            payload.model_dump_json(exclude_none=True).encode("utf-8")
        ).hexdigest()

    def _record_id(self, payload: ConversationPayload, lineage: RecordLineage) -> str:
        identity_payload = {
            "payload": payload.model_dump(mode="json", exclude_none=True),
            "lineage": lineage.model_dump(mode="json", exclude_none=True),
        }
        return hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    def _config_hash(self, ctx: StageContext) -> str:
        return hashlib.sha256(
            json.dumps(ctx.config.model_dump(mode="json"), sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()

    def _checkpoint_manager(self, ctx: StageContext) -> CheckpointManager:
        if self._checkpoint is not None:
            return self._checkpoint
        if ctx.checkpoint_manager is not None:
            self._checkpoint = ctx.checkpoint_manager
            return self._checkpoint
        project_root = self._project_root or self._project_root_from_work_dir(
            ctx.work_dir
        )
        self._checkpoint = CheckpointManager(project_root / "state.db")
        return self._checkpoint

    def _project_root_from_work_dir(self, work_dir: Path) -> Path:
        for parent in work_dir.parents:
            if parent.name == "runs":
                return parent.parent
        raise ValueError(f"Could not determine project_root from work_dir: {work_dir}")


class TransformResponse(StrictModel):
    text: str


class TransformGeneratorStage(Stage):
    name = "02_transform_generate"
    stage_action = "generated"

    def __init__(
        self,
        llm_client: Any | None = None,
        output_writer: OutputWriter | None = None,
        project_root: Path | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._output_writer = output_writer or OutputWriter()
        self._project_root = project_root

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        transformable_records = [
            record for record in records if isinstance(record, ConversationRecord)
        ]
        if not transformable_records:
            self._write_artifacts(ctx=ctx, dropped_records=[], costs=[])
            return []

        effective_llm_config = resolve_llm_override(
            ctx.config.llm, ctx.config.generator.llm_override
        )
        llm_client = self._llm_client or LLMClient(config=effective_llm_config)
        transformed_records: list[Record] = []
        costs: list[float] = []

        for record in transformable_records:
            input_text = self._field_value(record, ctx.config.generator.input_field)
            output = llm_client.complete_structured(
                messages=self._messages_for_input(input_text, ctx.config.generator),
                schema=TransformResponse,
                temperature=ctx.config.generator.temperature,
                max_tokens=ctx.config.generator.max_tokens,
            )
            parsed = output.parsed
            if not isinstance(parsed, TransformResponse):
                raise ValueError("Transform output did not parse into TransformResponse")
            if output.usage.cost_usd is not None:
                costs.append(output.usage.cost_usd)
            transformed_records.append(
                self._build_transformed_record(
                    record=record,
                    transformed_text=parsed.text.strip(),
                    config_hash=self._config_hash(ctx),
                    generator_config=ctx.config.generator,
                )
            )

        self._write_artifacts(ctx=ctx, dropped_records=[], costs=costs)
        return transformed_records

    def _messages_for_input(
        self,
        input_text: str,
        generator_config: GeneratorConfig,
    ) -> list[dict[str, str]]:
        content = generator_config.prompt_template.format(input_text=input_text)
        return [{"role": "user", "content": content}]

    def _build_transformed_record(
        self,
        *,
        record: ConversationRecord,
        transformed_text: str,
        config_hash: str,
        generator_config: GeneratorConfig,
    ) -> ConversationRecord:
        payload = record.payload.model_copy(deep=True)
        original_text = self._field_value(record, generator_config.output_field)
        payload = self._set_payload_field(
            payload=payload,
            field_path=generator_config.output_field,
            value=transformed_text,
        )
        if generator_config.preserve_original:
            payload.system = json.dumps(
                {
                    "transform_original": {
                        "field": generator_config.output_field,
                        "text": original_text,
                    }
                },
                separators=(",", ":"),
            )

        lineage = RecordLineage(
            root_id=record.lineage.root_id,
            parent_ids=[record.id],
            operator="transform",
            round=(record.lineage.round or 0) + 1,
            depth=(record.lineage.depth or 0) + 1,
        )
        record_id = self._record_id(payload, lineage)
        return ConversationRecord(
            id=record_id,
            content_hash=self._content_hash(payload),
            source=record.source.model_copy(deep=True),
            lineage=lineage,
            payload=payload,
            scores=RecordScores(),
            config_hash=config_hash,
            created_at=datetime.now(UTC).isoformat(),
        )

    def _field_value(self, record: ConversationRecord, field_path: str | None) -> str:
        if field_path == "payload.instruction":
            return record.payload.instruction
        if field_path == "payload.response":
            return record.payload.response
        if field_path == "payload.system":
            return record.payload.system or ""
        raise ValueError(f"Unsupported transform field path: {field_path}")

    def _set_payload_field(
        self,
        *,
        payload: ConversationPayload,
        field_path: str | None,
        value: str,
    ) -> ConversationPayload:
        if field_path == "payload.instruction":
            return payload.model_copy(update={"instruction": value})
        if field_path == "payload.response":
            return payload.model_copy(update={"response": value})
        if field_path == "payload.system":
            return payload.model_copy(update={"system": value})
        raise ValueError(f"Unsupported transform field path: {field_path}")

    def _content_hash(self, payload: ConversationPayload) -> str:
        return hashlib.sha256(
            payload.model_dump_json(exclude_none=True).encode("utf-8")
        ).hexdigest()

    def _record_id(self, payload: ConversationPayload, lineage: RecordLineage) -> str:
        identity_payload = {
            "payload": payload.model_dump(mode="json", exclude_none=True),
            "lineage": lineage.model_dump(mode="json", exclude_none=True),
        }
        return hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    def _config_hash(self, ctx: StageContext) -> str:
        return hashlib.sha256(
            json.dumps(ctx.config.model_dump(mode="json"), sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()

    def _write_artifacts(
        self,
        *,
        ctx: StageContext,
        dropped_records: list[Record],
        costs: list[float],
    ) -> None:
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        self._output_writer.write_dropped_parquet(
            records=dropped_records,
            path=ctx.work_dir / "dropped.parquet",
        )
        total_cost = round(sum(costs), 6) if costs else None
        stats = {
            "stage": self.name,
            "count_in": len(costs) if costs else 0,
            "count_out": len(costs) if costs else 0,
            "dropped_count": len(dropped_records),
            "drop_reasons": {},
            "cost_usd": total_cost,
        }
        (ctx.work_dir / "stats.json").write_text(json.dumps(stats, indent=2))
