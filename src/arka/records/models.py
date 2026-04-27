from __future__ import annotations

from typing import Any

from pydantic import Field

from arka.common.models import StrictModel


class RecordSource(StrictModel):
    type: str
    doc_id: str | None = None
    chunk_id: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    source_hash: str | None = None
    seed_file_hash: str | None = None


class RecordLineage(StrictModel):
    root_id: str
    parent_ids: list[str]
    operator: str | None = None
    round: int | None = None
    depth: int | None = None


class RecordScores(StrictModel):
    quality: float | None = None
    quality_per_dim: dict[str, Any] = Field(default_factory=dict)
    rubric_hash: str | None = None
    rubric_version: str | None = None
    judge_model: str | None = None
    judge_prompt_hash: str | None = None
    ifd: float | None = None
    reward_model: float | None = None
    humanness: float | None = None
    humanness_per_dim: dict[str, float] | None = None
    humanness_checklist: dict[str, bool] | None = None
    humanness_reasoning: str | None = None


class StageEvent(StrictModel):
    stage: str
    action: str
    reason_code: str | None = None
    details: str | None = None
    seq: int


class Record(StrictModel):
    id: str
    content_hash: str
    source: RecordSource
    lineage: RecordLineage
    payload: dict[str, Any]
    scores: RecordScores
    stage_events: list[StageEvent] = Field(default_factory=list)
    config_hash: str
    created_at: str

    @property
    def record_type(self) -> str:
        return self.__class__.__name__

    def export_payload(self) -> dict[str, Any]:
        return self.payload

    def text_for_diversity(self) -> str | None:
        return None

    def with_event(
        self,
        *,
        stage: str,
        action: str,
        reason_code: str | None = None,
        details: str | None = None,
    ) -> Record:
        """Return a copy of this Record with one StageEvent appended.

        Centralises the StageEvent invariants (monotonic ``seq``, immutable
        copy, ``action`` vocabulary) so individual stages do not re-implement
        them. See ``dropped_by`` for the canonical drop helper.
        """
        event = StageEvent(
            stage=stage,
            action=action,
            reason_code=reason_code,
            details=details,
            seq=len(self.stage_events) + 1,
        )
        return self.model_copy(update={"stage_events": [*self.stage_events, event]})

    def dropped_by(
        self,
        stage: str,
        reason_code: str,
        details: str | None = None,
    ) -> Record:
        """Return a copy of this Record marked as dropped by ``stage``.

        Equivalent to ``with_event(stage=stage, action="dropped", ...)`` --
        the canonical way a Stage records a Drop Reason on a Record.
        """
        return self.with_event(
            stage=stage,
            action="dropped",
            reason_code=reason_code,
            details=details,
        )


class ConversationPayload(StrictModel):
    instruction: str
    response: str
    system: str | None = None
    turns: list[dict[str, Any]] | None = None


class ConversationRecord(Record):
    payload: ConversationPayload

    def export_payload(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json")

    def text_for_diversity(self) -> str | None:
        return self.payload.instruction


class GroundedChunkPayload(StrictModel):
    text: str
    doc_id: str
    chunk_idx: int
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    word_count: int
    chunk_strategy: str


class GroundedChunkRecord(Record):
    payload: GroundedChunkPayload

    def export_payload(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json")

    def text_for_diversity(self) -> str | None:
        return self.payload.text


RECORD_TYPE_REGISTRY: dict[str, type[Record]] = {
    "Record": Record,
    "ConversationRecord": ConversationRecord,
    "GroundedChunkRecord": GroundedChunkRecord,
}


def record_model_for_name(name: str) -> type[Record]:
    if name in RECORD_TYPE_REGISTRY:
        return RECORD_TYPE_REGISTRY[name]
    return Record
