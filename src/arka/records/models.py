from __future__ import annotations

from typing import Any

from pydantic import Field

from arka.common.models import StrictModel
from arka.decisions.models import DecisionAnswer


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


class DecontaminationMatch(StrictModel):
    benchmark: str
    eval_id: int
    score: float = Field(ge=0, le=1)
    field: str
    window_start: int
    window_end: int


class EscalationEntry(StrictModel):
    record_id: str
    stage: str
    reason_code: str
    details: str | None = None
    question: str | None = None
    probability: float | None = None
    confidence: float | None = None
    resolved: bool = False


class RecordAnnotations(StrictModel):
    taxonomy_nodes: dict[str, list[str]] = Field(default_factory=dict)
    taxonomy_nodes_requested: dict[str, list[str]] = Field(default_factory=dict)
    seed_domain: str | None = None
    double_critic: dict[str, Any] | None = None
    pair_preference: dict[str, Any] | None = None
    escalations: list[dict[str, Any]] = Field(default_factory=list)


class RidgePrediction(StrictModel):
    model_sha256: str
    input_hash: str
    target: str
    rubric_hash: str


class RecordScores(StrictModel):
    distilled_ridge: float | None = Field(default=None, allow_inf_nan=False)
    decisions: dict[str, DecisionAnswer] = Field(default_factory=dict)
    quality: float | None = None
    distilled_quality: float | None = None
    quality_per_dim: dict[str, Any] = Field(default_factory=dict)
    annotations: dict[str, Any] = Field(default_factory=dict)
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
    decon_match: DecontaminationMatch | None = None

    @property
    def distilled_ridge_prediction(self) -> RidgePrediction | None:
        value = self.annotations.get("distilled_ridge")
        return RidgePrediction.model_validate(value) if value is not None else None

    def get_annotation(self, key: str, default: Any = None) -> Any:
        if key in self.annotations:
            return self.annotations[key]
        return self.quality_per_dim.get(key, default)

    def with_annotation(self, key: str, value: Any) -> RecordScores:
        new_annotations = dict(self.annotations)
        new_annotations[key] = value
        new_qpd = dict(self.quality_per_dim)
        new_qpd[key] = value
        return self.model_copy(
            update={
                "annotations": new_annotations,
                "quality_per_dim": new_qpd,
            }
        )


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

    @property
    def annotations(self) -> dict[str, Any]:
        return {**self.scores.quality_per_dim, **self.scores.annotations}

    def with_annotation(self, key: str, value: Any) -> Record:
        return self.model_copy(
            update={"scores": self.scores.with_annotation(key, value)}
        )

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

    def with_escalation(
        self,
        *,
        stage: str,
        reason_code: str,
        details: str | None = None,
        question: str | None = None,
        probability: float | None = None,
        confidence: float | None = None,
    ) -> Record:
        """Return a copy marked as escalated with structured escalation entry."""
        entry = {
            "record_id": self.id,
            "stage": stage,
            "reason_code": reason_code,
            "details": details,
            "question": question,
            "probability": probability,
            "confidence": confidence,
            "resolved": False,
        }
        current_escalations = list(self.scores.annotations.get("escalations", []))
        current_escalations.append(entry)
        updated_scores = self.scores.model_copy(
            update={
                "annotations": {
                    **self.scores.annotations,
                    "escalations": current_escalations,
                }
            }
        )
        record = self.model_copy(update={"scores": updated_scores})
        return record.with_event(
            stage=stage,
            action="escalated",
            reason_code=reason_code,
            details=details,
        )

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

    def is_escalated(self, stage: str | None = None) -> bool:
        return any(
            event.action == "escalated" and (stage is None or event.stage == stage)
            for event in self.stage_events
        )


class ConversationPayload(StrictModel):
    instruction: str
    response: str
    system: str | None = None
    turns: list[dict[str, Any]] | None = None
    reasoning: str | None = Field(default=None, exclude_if=lambda value: value is None)


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
