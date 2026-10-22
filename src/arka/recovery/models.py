from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from arka.common.models import StrictModel
from arka.config.loader import ConfigLoader
from arka.config.models import LLMConfig
from arka.decisions.models import DecisionConfig
from arka.llm.models import TokenUsage
from arka.records.models import ConversationRecord


class RecoveryJudgeConfig(StrictModel):
    backend: Literal["llm", "decision"]
    rubric_path: str = Field(min_length=1)
    min_overall_score: float = Field(allow_inf_nan=False)
    llm: LLMConfig | None = None
    decision: DecisionConfig | None = None

    @model_validator(mode="after")
    def validate_backend(self):
        if getattr(self, self.backend) is None:
            raise ValueError(f"judge.{self.backend} configuration is required")
        if self.backend == "llm" and self.decision is not None:
            raise ValueError("judge.decision is only valid for backend decision")
        if self.backend == "decision" and self.llm is not None:
            raise ValueError("judge.llm is only valid for backend llm")
        return self


class ReplacementConfig(StrictModel):
    llm: LLMConfig
    max_attempts: int = Field(ge=1)
    temperature: float = Field(default=0.7, ge=0, le=2, allow_inf_nan=False)
    max_tokens: int | None = Field(default=None, ge=1)
    exact_dedup: bool = True


class RecoveryConfig(StrictModel):
    judge: RecoveryJudgeConfig
    replacement: ReplacementConfig | None = None
    sample_size: int = Field(default=200, ge=1)
    sample_seed: int = Field(default=0, ge=0)

    @classmethod
    def load(cls, path: Path) -> RecoveryConfig:
        text = ConfigLoader()._resolve_env_vars(path.read_text(encoding="utf-8"))
        return cls.model_validate(yaml.safe_load(text))


class Assessment(StrictModel):
    score: float = Field(allow_inf_nan=False)
    scores: dict[str, float]
    reasoning: str | None = None
    passed: bool
    model: str
    provider: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    requests: int = 1
    prompt_hash: str


class Candidate(StrictModel):
    response: str
    reasoning: str | None


class SavedCandidate(StrictModel):
    record: ConversationRecord
    usage: TokenUsage
    model: str
    provider: str
    prompt_hash: str


class CandidateOutcome(StrictModel):
    accepted: bool
    reason: Literal[
        "accepted",
        "quality_failed",
        "duplicate",
        "empty_response",
        "unchanged_response",
    ]


class SourceSummary(StrictModel):
    key: str
    total_count: int
    sampled_count: int
    failed_count: int
    failure_rate: float | None
    sample_ids: list[str]
    selected: bool = False
    assessed_count: int = 0
    full_failed_count: int = 0
    attempt_count: int = 0
    accepted_replacements: int = 0
    exhausted_count: int = 0


class RecoveryReport(StrictModel):
    run_id: str
    operation: Literal["audit", "rebuild"]
    status: Literal["completed", "failed"] = "completed"
    parent_run_id: str
    parent_stage: str
    audit_run_id: str | None = None
    selected_sources: list[str] = Field(default_factory=list)
    sources: list[SourceSummary] = Field(default_factory=list)
    total_count: int
    sampled_count: int = 0
    assessed_count: int = 0
    failed_count: int = 0
    accepted_replacements: int = 0
    exhausted_count: int = 0
    unresolved_count: int = 0
    attempt_count: int = 0
    duplicate_count: int = 0
    final_count: int = 0
    usage: TokenUsage = Field(default_factory=TokenUsage)
    requests: int = 0
    request_count_scope: str = (
        "Completed client calls; transport retries and failed requests are not counted"
    )
    unknown_cost_calls: int = 0
    cost_complete: bool = True
    dataset_path: str | None = None
    error: str | None = None
