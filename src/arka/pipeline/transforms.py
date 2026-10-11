from __future__ import annotations

from arka.pipeline.models import StageContext
from arka.pipeline.stages import Stage
from arka.records.identity import content_hash
from arka.records.models import ConversationRecord, Record


class NormalizeConversationStage(Stage):
    name = "02_normalize"

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        normalized_records: list[Record] = []
        for record in records:
            if isinstance(record, ConversationRecord):
                payload = record.payload.model_copy(
                    update={
                        "instruction": record.payload.instruction.strip(),
                        "response": record.payload.response.strip(),
                        "reasoning": record.payload.reasoning.strip() or None
                        if record.payload.reasoning is not None
                        else None,
                    }
                )
                normalized_records.append(
                    record.model_copy(
                        update={
                            "payload": payload,
                            "content_hash": content_hash(payload),
                        }
                    )
                )
                continue
            normalized_records.append(record)
        return normalized_records
