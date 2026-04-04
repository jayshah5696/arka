from __future__ import annotations

from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


def test_conversation_record_preserves_typed_payload() -> None:
    record = ConversationRecord(
        id="rec-1",
        content_hash="hash-1",
        source=RecordSource(type="generated"),
        lineage=RecordLineage(root_id="root-1", parent_ids=[]),
        payload=ConversationPayload(
            instruction="Say hello",
            response="Hello",
        ),
        scores=RecordScores(),
        config_hash="cfg-1",
        created_at="2026-04-04T00:00:00Z",
    )

    assert record.payload.instruction == "Say hello"
    assert record.payload.response == "Hello"
    assert record.export_payload() == {
        "instruction": "Say hello",
        "response": "Hello",
        "system": None,
        "turns": None,
    }
