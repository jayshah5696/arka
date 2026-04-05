from __future__ import annotations

from arka.common.models import StrictModel


class LabelResult(StrictModel):
    scores: dict[str, int]
    overall: float
    reasoning: str
    rubric_hash: str
    rubric_version: str
    judge_model: str
    judge_prompt_hash: str
    provider: str
    latency_ms: int
