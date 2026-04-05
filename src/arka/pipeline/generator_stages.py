from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import cycle, islice
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from arka.common.models import StrictModel
from arka.config.models import GeneratorConfig, LLMConfig
from arka.llm.client import LLMClient
from arka.llm.models import LLMOutput, TokenUsage
from arka.pipeline.checkpoint import CheckpointManager
from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.pipeline.stages import Stage
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    Record,
    RecordLineage,
    RecordScores,
    RecordSource,
)

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
    seed_record: ConversationRecord


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
    ) -> None:
        self._llm_client = llm_client
        self._checkpoint = checkpoint_manager
        self._output_writer = output_writer or OutputWriter()

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        seed_records = [
            record for record in records if isinstance(record, ConversationRecord)
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
        seed_records: list[ConversationRecord],
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

        with responses_path.open("a", encoding="utf-8") as handle:
            for item in pending:
                output = self._complete_raw(llm_client, item.seed_record, ctx)
                row = RawGeneratorResponse(
                    plan_index=item.plan_index,
                    seed_id=item.seed_record.id,
                    generated_text=(output.text or "").strip(),
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
        seed_record: ConversationRecord,
        ctx: StageContext,
    ) -> LLMOutput:
        messages = self._messages_for_seed(seed_record, ctx.config.generator)
        if hasattr(llm_client, "complete"):
            return llm_client.complete(
                messages=messages,
                temperature=ctx.config.generator.temperature,
                max_tokens=ctx.config.generator.max_tokens,
            )
        return llm_client.complete_structured(
            messages=messages,
            schema=GeneratedConversation,
        )

    def _messages_for_seed(
        self,
        seed_record: ConversationRecord,
        config: GeneratorConfig,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": config.prompt_template.format(
                    seed_instruction=seed_record.payload.instruction,
                    seed_response=seed_record.payload.response,
                ),
            }
        ]

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

        for row in raw_rows:
            item = plan_by_index.get(row.plan_index)
            if item is None:
                continue
            payload = self._parse_generated_payload(row.generated_text)
            generated_records.append(
                self._build_generated_record(
                    payload=payload,
                    parent=item.seed_record,
                    config_hash=config_hash,
                )
            )

        if len(generated_records) != len(plan):
            raise ValueError(
                "Generator responses did not cover the full generation plan: "
                f"expected {len(plan)}, got {len(generated_records)}"
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

    def _build_generated_record(
        self,
        payload: GeneratedConversation,
        parent: ConversationRecord,
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
        record_id = self._record_id(conversation_payload, lineage)
        return ConversationRecord(
            id=record_id,
            content_hash=content_hash,
            source=RecordSource(
                type="generated",
                seed_file_hash=parent.source.seed_file_hash,
                source_hash=parent.content_hash,
            ),
            lineage=lineage,
            payload=conversation_payload,
            scores=RecordScores(),
            config_hash=config_hash,
            created_at=datetime.now(UTC).isoformat(),
        )

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
        project_root = ctx.work_dir.parents[3]
        self._checkpoint = CheckpointManager(project_root / "state.db")
        return self._checkpoint
