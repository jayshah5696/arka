from __future__ import annotations

from pathlib import Path

import pytest

from arka.labeling.rubric import RubricLoader, RubricValidationError

RUBRIC_YAML = """
version: "1.0"
description: "SFT instruction-response quality rubric"
dimensions:
  - name: instruction_clarity
    description: Is the instruction unambiguous and well-scoped?
    scale_min: 1
    scale_max: 5
    criteria:
      1: Unclear, multiple valid interpretations
      3: Mostly clear, minor ambiguity
      5: Perfectly clear, single interpretation
  - name: response_quality
    description: Accurate, complete, on-topic, appropriately concise?
    scale_min: 1
    scale_max: 5
    criteria:
      1: Wrong, incomplete, or off-topic
      3: Mostly correct, minor gaps
      5: Accurate, complete, well-calibrated length
overall_weights:
  instruction_clarity: 0.4
  response_quality: 0.6
few_shot:
  - instruction: "What is 2+2?"
    response: "4"
    scores: {instruction_clarity: 5, response_quality: 5}
    reasoning: "Clear simple question, correct answer."
    expected_verdict: pass
  - instruction: "Tell me stuff"
    response: "Here is some stuff."
    scores: {instruction_clarity: 1, response_quality: 1}
    reasoning: "Vague instruction, non-answer response."
    expected_verdict: fail
"""


def test_rubric_loader_reads_yaml_and_hash_is_stable(tmp_path: Path) -> None:
    rubric_path = tmp_path / "rubric.yaml"
    rubric_path.write_text(RUBRIC_YAML)

    loader = RubricLoader()
    first = loader.load(rubric_path)
    second = loader.load(rubric_path)

    assert first.hash == second.hash
    assert first.overall_weights == {
        "instruction_clarity": 0.4,
        "response_quality": 0.6,
    }
    assert len(first.few_shot) == 2


def test_rubric_loader_rejects_unknown_weight_dimension(tmp_path: Path) -> None:
    rubric_path = tmp_path / "rubric.yaml"
    rubric_path.write_text(
        RUBRIC_YAML.replace(
            "response_quality: 0.6",
            "hallucination: 0.6",
        )
    )

    with pytest.raises(RubricValidationError, match="overall_weights"):
        RubricLoader().load(rubric_path)


def test_rubric_loader_requires_expected_verdicts_for_few_shot_examples(
    tmp_path: Path,
) -> None:
    rubric_path = tmp_path / "rubric.yaml"
    rubric_path.write_text(RUBRIC_YAML.replace("    expected_verdict: pass\n", ""))

    with pytest.raises(RubricValidationError, match="expected_verdict"):
        RubricLoader().load(rubric_path)
