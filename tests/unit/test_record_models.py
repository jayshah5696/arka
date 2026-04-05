from __future__ import annotations

from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    GroundedChunkPayload,
    GroundedChunkRecord,
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


def test_grounded_chunk_record_preserves_typed_payload() -> None:
    record = GroundedChunkRecord(
        id="chunk-1",
        content_hash="hash-chunk-1",
        source=RecordSource(
            type="pdf_chunk",
            doc_id="doc-1",
            chunk_id="doc-1:0",
            page_start=1,
            page_end=1,
            char_start=0,
            char_end=42,
            source_hash="source-hash",
        ),
        lineage=RecordLineage(root_id="chunk-1", parent_ids=[]),
        payload=GroundedChunkPayload(
            text="Hello PDF world",
            doc_id="doc-1",
            chunk_idx=0,
            page_start=1,
            page_end=1,
            char_start=0,
            char_end=42,
            word_count=3,
            chunk_strategy="fixed",
        ),
        scores=RecordScores(),
        config_hash="cfg-1",
        created_at="2026-04-05T00:00:00Z",
    )

    assert record.payload.text == "Hello PDF world"
    assert record.export_payload() == {
        "text": "Hello PDF world",
        "doc_id": "doc-1",
        "chunk_idx": 0,
        "page_start": 1,
        "page_end": 1,
        "char_start": 0,
        "char_end": 42,
        "word_count": 3,
        "chunk_strategy": "fixed",
    }


def test_record_text_for_diversity_defaults_to_none() -> None:
    from arka.records.models import Record

    record = Record(
        id="rec-1",
        content_hash="hash-1",
        source=RecordSource(type="generated"),
        lineage=RecordLineage(root_id="root-1", parent_ids=[]),
        payload={"value": "alpha"},
        scores=RecordScores(),
        config_hash="cfg-1",
        created_at="2026-04-04T00:00:00Z",
    )

    assert record.text_for_diversity() is None


def test_conversation_and_grounded_chunk_records_expose_diversity_text() -> None:
    conversation = ConversationRecord(
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
    chunk = GroundedChunkRecord(
        id="chunk-1",
        content_hash="hash-chunk-1",
        source=RecordSource(type="pdf_chunk", doc_id="doc-1", chunk_id="doc-1:0"),
        lineage=RecordLineage(root_id="chunk-1", parent_ids=[]),
        payload=GroundedChunkPayload(
            text="Hello PDF world",
            doc_id="doc-1",
            chunk_idx=0,
            page_start=1,
            page_end=1,
            char_start=0,
            char_end=42,
            word_count=3,
            chunk_strategy="fixed",
        ),
        scores=RecordScores(),
        config_hash="cfg-1",
        created_at="2026-04-05T00:00:00Z",
    )

    assert conversation.text_for_diversity() == "Say hello"
    assert chunk.text_for_diversity() == "Hello PDF world"
