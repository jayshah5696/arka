from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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
