from __future__ import annotations

from arka.labeling.rubric import Rubric


def build_single_judge_messages(
    instruction: str,
    response: str,
    rubric: Rubric,
) -> list[dict[str, str]]:
    dimensions = "\n".join(
        f"- {dimension.name}: {dimension.description}; criteria={dimension.criteria}"
        for dimension in rubric.dimensions
    )
    few_shot = "\n\n".join(
        (
            f"Instruction: {example.instruction}\n"
            f"Response: {example.response}\n"
            f"Scores: {example.scores}\n"
            f"Reasoning: {example.reasoning}"
        )
        for example in rubric.few_shot
    )
    system_prompt = (
        f"Rubric version: {rubric.version}\n"
        f"Description: {rubric.description}\n"
        f"Dimensions:\n{dimensions}\n\n"
        f"Few-shot examples:\n{few_shot}"
    )
    user_prompt = f"Instruction: {instruction}\nResponse: {response}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
