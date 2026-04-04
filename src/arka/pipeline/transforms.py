from __future__ import annotations

from arka.pipeline.models import StageContext
from arka.pipeline.stages import Stage
from arka.records.models import ConversationPayload, ConversationRecord, Record


class NormalizeConversationStage(Stage):
    name = "02_normalize"

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        normalized_records: list[Record] = []
        for record in records:
            if isinstance(record, ConversationRecord):
                normalized_records.append(
                    record.model_copy(
                        update={
                            "payload": ConversationPayload(
                                instruction=record.payload.instruction.strip(),
                                response=record.payload.response.strip(),
                                system=record.payload.system,
                                turns=record.payload.turns,
                            )
                        }
                    )
                )
                continue
            normalized_records.append(record)
        return normalized_records
