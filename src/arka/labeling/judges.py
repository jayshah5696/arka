from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, ConfigDict

from arka.labeling.models import LabelResult
from arka.labeling.prompting import build_single_judge_messages
from arka.labeling.rubric import Rubric


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JudgeResponse(StrictModel):
    scores: dict[str, int]
    reasoning: str


class SingleLLMJudge:
    def __init__(self, llm_client: Any) -> None:
        self.llm_client = llm_client

    def label(self, instruction: str, response: str, rubric: Rubric) -> LabelResult:
        messages = build_single_judge_messages(
            instruction=instruction,
            response=response,
            rubric=rubric,
        )
        prompt_hash = hashlib.sha256(
            messages[0]["content"].encode("utf-8")
        ).hexdigest()[:16]
        llm_output = self.llm_client.complete_structured(
            messages=messages,
            schema=JudgeResponse,
        )
        parsed = llm_output.parsed
        if not isinstance(parsed, JudgeResponse):
            raise ValueError("Judge output did not parse into JudgeResponse")
        overall = self._compute_overall(
            scores=parsed.scores,
            weights=rubric.overall_weights,
        )
        return LabelResult(
            scores=parsed.scores,
            overall=overall,
            reasoning=parsed.reasoning,
            rubric_hash=rubric.hash,
            rubric_version=rubric.version,
            judge_model=llm_output.model,
            judge_prompt_hash=prompt_hash,
            provider=llm_output.provider,
            latency_ms=llm_output.latency_ms,
        )

    def _compute_overall(
        self, scores: dict[str, int], weights: dict[str, float]
    ) -> float:
        total = sum(scores[name] * weights[name] for name in weights)
        return round(total, 4)
