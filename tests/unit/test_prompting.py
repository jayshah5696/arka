from __future__ import annotations

from arka.labeling.prompting import build_single_judge_messages
from arka.labeling.rubric import Rubric


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
            "few_shot": [],
        }
    )


def test_prompting_requests_json_only_output() -> None:
    messages = build_single_judge_messages(
        instruction="Explain gravity",
        response="Gravity attracts masses.",
        rubric=build_rubric(),
    )

    assert "Return valid JSON only" in messages[0]["content"]
    assert '"scores"' in messages[0]["content"]
    assert '"reasoning"' in messages[0]["content"]
