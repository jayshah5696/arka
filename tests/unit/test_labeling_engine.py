from __future__ import annotations

import warnings

from pydantic import BaseModel

from arka.labeling.engine import LabelingEngine
from arka.labeling.judges import JudgeResponse
from arka.labeling.rubric import Rubric
from arka.llm.models import LLMOutput, TokenUsage


class SequentialFakeLLMClient:
    def __init__(self, responses: list[JudgeResponse]) -> None:
        self.responses = responses
        self.calls = 0

    def complete_structured(self, messages, schema: type[BaseModel]) -> LLMOutput:
        response = self.responses[self.calls]
        self.calls += 1
        return LLMOutput(
            text="{}",
            parsed=response,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="stop",
            model="gpt-4o-mini",
            provider="openai",
            request_id=f"req_{self.calls}",
            latency_ms=10,
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
                },
                {
                    "instruction": "Tell me stuff",
                    "response": "Here is some stuff.",
                    "scores": {"instruction_clarity": 1, "response_quality": 1},
                    "reasoning": "Vague and weak.",
                },
            ],
        }
    )


def test_labeling_engine_label_batch_returns_results_for_inputs_only() -> None:
    rubric = build_rubric()
    client = SequentialFakeLLMClient(
        [
            JudgeResponse(
                scores={"instruction_clarity": 5, "response_quality": 4},
                reasoning="strong",
            ),
            JudgeResponse(
                scores={"instruction_clarity": 5, "response_quality": 5},
                reasoning="good canary",
            ),
            JudgeResponse(
                scores={"instruction_clarity": 1, "response_quality": 1},
                reasoning="bad canary",
            ),
        ]
    )
    engine = LabelingEngine(llm_client=client)

    results = engine.label_batch(
        pairs=[("Explain gravity", "Gravity attracts masses.")],
        rubric=rubric,
        max_workers=1,
    )

    assert len(results) == 1
    assert results[0].overall == 4.4


def test_labeling_engine_warns_when_known_bad_canary_scores_too_high() -> None:
    rubric = build_rubric()
    client = SequentialFakeLLMClient(
        [
            JudgeResponse(
                scores={"instruction_clarity": 5, "response_quality": 4},
                reasoning="strong",
            ),
            JudgeResponse(
                scores={"instruction_clarity": 5, "response_quality": 5},
                reasoning="good canary",
            ),
            JudgeResponse(
                scores={"instruction_clarity": 5, "response_quality": 5},
                reasoning="bad canary scored too high",
            ),
        ]
    )
    engine = LabelingEngine(llm_client=client)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        engine.label_batch(
            pairs=[("Explain gravity", "Gravity attracts masses.")],
            rubric=rubric,
            max_workers=1,
        )

    assert any("known-bad" in str(warning.message) for warning in captured)
