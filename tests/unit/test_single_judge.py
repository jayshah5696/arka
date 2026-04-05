from __future__ import annotations

from pydantic import BaseModel

from arka.labeling.judges import JudgeResponse, SingleLLMJudge
from arka.labeling.models import LabelResult
from arka.labeling.rubric import Rubric
from arka.llm.models import LLMOutput, TokenUsage


class FakeLLMClient:
    def __init__(self, response: JudgeResponse) -> None:
        self.response = response
        self.last_messages: list[dict[str, str]] | None = None

    def complete_structured(self, messages, schema: type[BaseModel]) -> LLMOutput:
        self.last_messages = list(messages)
        return LLMOutput(
            text='{"scores":{"instruction_clarity":5,"response_quality":4},"reasoning":"instruction_clarity and response_quality are both strong"}',
            parsed=self.response,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="stop",
            model="gpt-4o-mini",
            provider="openai",
            request_id="req_1",
            latency_ms=12,
            error=None,
        )


def build_rubric() -> Rubric:
    return Rubric.model_validate(
        {
            "version": "1.0",
            "description": "SFT quality rubric",
            "dimensions": [
                {
                    "name": "instruction_clarity",
                    "description": "Clear instruction",
                    "scale_min": 1,
                    "scale_max": 5,
                    "criteria": {1: "bad", 5: "good"},
                },
                {
                    "name": "response_quality",
                    "description": "Good response",
                    "scale_min": 1,
                    "scale_max": 5,
                    "criteria": {1: "bad", 5: "good"},
                },
            ],
            "overall_weights": {
                "instruction_clarity": 0.4,
                "response_quality": 0.6,
            },
            "few_shot": [
                {
                    "instruction": "What is 2+2?",
                    "response": "4",
                    "scores": {"instruction_clarity": 5, "response_quality": 5},
                    "reasoning": "Clear and correct.",
                    "expected_verdict": "pass",
                }
            ],
        }
    )


def test_single_judge_returns_label_result_with_metadata() -> None:
    rubric = build_rubric()
    client = FakeLLMClient(
        JudgeResponse(
            scores={"instruction_clarity": 5, "response_quality": 4},
            reasoning="instruction_clarity and response_quality are both strong",
        )
    )

    result = SingleLLMJudge(llm_client=client).label(
        instruction="Explain gravity",
        response="Gravity is the attractive force between masses.",
        rubric=rubric,
    )

    assert isinstance(result, LabelResult)
    assert result.overall == 4.4
    assert result.rubric_hash == rubric.hash
    assert result.rubric_version == "1.0"
    assert result.judge_model == "gpt-4o-mini"
    assert result.provider == "openai"
    assert result.reasoning.startswith("instruction_clarity")
    assert result.judge_prompt_hash


def test_single_judge_builds_prompt_with_rubric_details_and_few_shot() -> None:
    rubric = build_rubric()
    client = FakeLLMClient(
        JudgeResponse(
            scores={"instruction_clarity": 5, "response_quality": 4},
            reasoning="Looks good",
        )
    )

    SingleLLMJudge(llm_client=client).label(
        instruction="Explain gravity",
        response="Gravity is the attractive force between masses.",
        rubric=rubric,
    )

    assert client.last_messages is not None
    assert "instruction_clarity" in client.last_messages[0]["content"]
    assert "What is 2+2?" in client.last_messages[0]["content"]
    assert "Explain gravity" in client.last_messages[1]["content"]
