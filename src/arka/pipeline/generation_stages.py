from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from itertools import cycle, islice
from typing import Any

from arka.common.models import StrictModel
from arka.config.models import GeneratorConfig
from arka.llm.client import LLMClient
from arka.pipeline.models import StageContext
from arka.pipeline.stages import Stage
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    Record,
    RecordLineage,
    RecordScores,
    RecordSource,
)


class GeneratedConversation(StrictModel):
    instruction: str
    response: str


class PromptBasedGeneratorStage(Stage):
    name = "02_generate"
    stage_action = "generated"

    def __init__(self, llm_client: Any | None = None) -> None:
        self._llm_client = llm_client

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        seed_records = [
            record for record in records if isinstance(record, ConversationRecord)
        ]
        if not seed_records:
            return []

        generation_plan = list(
            self._generation_plan(seed_records, ctx.config.generator)
        )
        if not generation_plan:
            return []

        llm_client = self._llm_client or LLMClient(config=ctx.config.llm)
        config_hash = self._config_hash(ctx)
        generated_records: list[Record] = []

        for generation_index, seed_record in enumerate(generation_plan, start=1):
            generated_payload = self._generate_one(llm_client, seed_record)
            generated_records.append(
                self._build_generated_record(
                    payload=generated_payload,
                    parent=seed_record,
                    config_hash=config_hash,
                    generation_index=generation_index,
                )
            )

        return generated_records

    def _generation_plan(
        self,
        seed_records: list[ConversationRecord],
        config: GeneratorConfig,
    ) -> list[ConversationRecord]:
        requested_count = config.target_count * config.generation_multiplier
        if requested_count <= 0:
            return []
        return list(islice(cycle(seed_records), requested_count))

    def _generate_one(
        self,
        llm_client: Any,
        seed_record: ConversationRecord,
    ) -> GeneratedConversation:
        output = llm_client.complete_structured(
            messages=self._messages_for_seed(seed_record),
            schema=GeneratedConversation,
        )
        parsed = output.parsed
        if not isinstance(parsed, GeneratedConversation):
            raise ValueError(
                "Generator output did not parse into GeneratedConversation"
            )
        return GeneratedConversation(
            instruction=parsed.instruction.strip(),
            response=parsed.response.strip(),
        )

    def _messages_for_seed(
        self, seed_record: ConversationRecord
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You generate synthetic instruction-response pairs for supervised "
                    "fine-tuning. Create one new instruction and one strong response "
                    "inspired by the seed example. The new pair must be self-contained, "
                    "specific, and meaningfully different from the seed. Return only the "
                    "structured fields."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Seed instruction:\n"
                    f"{seed_record.payload.instruction}\n\n"
                    "Seed response:\n"
                    f"{seed_record.payload.response}\n\n"
                    "Generate one similar-but-different instruction and its response."
                ),
            },
        ]

    def _build_generated_record(
        self,
        payload: GeneratedConversation,
        parent: ConversationRecord,
        config_hash: str,
        generation_index: int,
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
