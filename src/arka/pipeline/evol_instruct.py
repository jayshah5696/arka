from __future__ import annotations

from collections.abc import Sequence

from arka.records.models import ConversationRecord

SUPPORTED_EVOL_OPERATORS: tuple[str, ...] = (
    "add_constraints",
    "deepen",
    "increase_reasoning_steps",
    "breadth_mutation",
    "complexify",
)

_OPERATOR_PROMPTS: dict[str, str] = {
    "add_constraints": (
        "Rewrite the instruction so the answer must satisfy one or two concrete "
        "constraints. Keep it self-contained and meaningfully harder."
    ),
    "deepen": (
        "Rewrite the instruction so it requires deeper domain knowledge, more "
        "specificity, or stronger technical judgment."
    ),
    "increase_reasoning_steps": (
        "Rewrite the instruction so answering it requires multiple explicit "
        "reasoning steps, tradeoffs, or comparisons."
    ),
    "breadth_mutation": (
        "Create a new instruction on a related but different angle of the same "
        "general topic. Keep it self-contained and substantially different."
    ),
    # Simula §2.2 'complexify': preserve the task type and output format, push
    # the difficulty up along ONE concrete axis. Distinct from breadth_mutation,
    # which intentionally drifts to a sibling topic.
    "complexify": (
        "Rewrite the instruction so it is meaningfully more complex while "
        "PRESERVING the original task type, the original topic, and the output "
        "format. Pick exactly ONE complexification axis from this list and apply "
        "it explicitly: (a) add one realistic constraint that makes the task "
        "harder, (b) require an edge case or boundary condition to be handled, "
        "(c) increase the depth of domain knowledge needed by one level, or "
        "(d) add one extra reasoning step the answer must walk through. Do NOT "
        "switch to a different topic; that is the job of breadth_mutation."
    ),
}


def build_evol_messages(
    parent: ConversationRecord,
    *,
    operator: str,
) -> list[dict[str, str]]:
    if operator not in _OPERATOR_PROMPTS:
        raise ValueError(f"Unsupported evol operator: {operator!r}")
    return [
        {
            "role": "user",
            "content": (
                "You rewrite an instruction for synthetic data generation.\n"
                f"Operator: {operator}\n"
                f"Goal: {_OPERATOR_PROMPTS[operator]}\n"
                'Return only JSON with key "instruction".\n\n'
                f"Parent instruction:\n{parent.payload.instruction}\n\n"
                f"Parent response:\n{parent.payload.response}\n"
            ),
        }
    ]


def build_response_messages(instruction: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "You write a strong response for supervised fine-tuning.\n"
                "Answer the instruction directly, clearly, and helpfully.\n"
                'Return only JSON with key "response".\n\n'
                f"Instruction:\n{instruction}\n"
            ),
        }
    ]


def normalized_instruction(text: str) -> str:
    return " ".join(text.lower().split())


def contains_refusal(text: str, refusal_keywords: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in refusal_keywords)


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    if len(left) < len(right):
        left, right = right, left

    previous_row = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current_row = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insert_cost = current_row[right_index - 1] + 1
            delete_cost = previous_row[right_index] + 1
            replace_cost = previous_row[right_index - 1] + (left_char != right_char)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row
    return previous_row[-1]
