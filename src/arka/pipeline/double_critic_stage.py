"""Slice 1 — Simula-style double-critic filter.

For each (instruction, response) pair we run TWO independent critic LLM calls:

  1. yes-critic: "Given this instruction, is the response **correct**? yes/no + reason."
  2. no-critic:  "Given this instruction, is the response **incorrect**? yes/no + reason."

ACCEPT iff yes_verdict == "yes" AND no_verdict == "no". Anything else is dropped
with reason ``double_critic_disagreement``. The full audit trail
``{yes_verdict, no_verdict, yes_reason, no_reason}`` is attached to
``record.scores.quality_per_dim["double_critic"]``.

Why two calls instead of one? Single positively-framed judges are biased toward
affirmation (Sharma et al. 2024 — sycophancy). Asking the inverse question
independently catches errors a single positive critic would let through. See
Davidson et al. 2026 (Simula, TMLR), §2.2 and §3.1.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from pydantic import BaseModel, Field

from arka.llm.client import LLMClient, LLMClientError
from arka.pipeline.filter_stages import _drop_record, _write_filter_artifacts
from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.pipeline.stages import Stage
from arka.records.models import (
    ConversationRecord,
    Record,
)

# --- Critic response schema ---------------------------------------------------


class CriticVerdict(BaseModel):
    """Structured output for a single critic call.

    The LLM is asked one yes/no question; we capture the verdict plus a one-line
    reason for the audit trail (debugging + possible future reward signal).
    """

    verdict: Literal["yes", "no"] = Field(
        ...,
        description="One-word answer: 'yes' or 'no'. Lower-case.",
    )
    reason: str = Field(
        ...,
        description="One short sentence explaining the verdict (for audit only).",
    )


# --- Prompt templates ---------------------------------------------------------

# We keep these as constants (not user-configurable yet) so the inverse-question
# property is preserved by construction. Slice 1 is intentionally rigid; making
# the templates configurable can wait until we see the metric move.

_YES_SYSTEM = (
    "You are an exacting evaluator. Read the instruction and response carefully, "
    "then answer ONE question: is the response a CORRECT and faithful answer to "
    "the instruction? Reply with verdict='yes' or verdict='no' and a one-sentence "
    "reason. Do not be lenient."
)

_NO_SYSTEM = (
    "You are an exacting evaluator. Read the instruction and response carefully, "
    "then answer ONE question: is the response INCORRECT, off-topic, or a "
    "non-answer to the instruction? Reply with verdict='yes' (it IS incorrect) "
    "or verdict='no' (it is NOT incorrect) and a one-sentence reason. Do not be "
    "lenient. Note: this question is the INVERSE of asking whether the response "
    "is correct; answer it independently."
)


def _build_messages(
    system: str, instruction: str, response: str
) -> list[dict[str, str]]:
    user = (
        f"INSTRUCTION:\n{instruction}\n\n"
        f"RESPONSE:\n{response}\n\n"
        'Reply with JSON {"verdict": "yes" | "no", "reason": "..."}.'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# --- Stage --------------------------------------------------------------------


class DoubleCriticFilterStage(Stage):
    """Drop records whose two independent critic calls disagree on correctness."""

    name = "03_double_critic"
    stage_action = "filtered"

    def __init__(self, llm_client: Any | None = None) -> None:
        self._llm_client = llm_client
        self._output_writer = OutputWriter()

    # NOTE: filter_config is currently a no-op (no tunable knobs). The config
    # entry exists so users can opt the stage in/out via YAML; future knobs
    # (e.g. allow majority-of-N, alternate prompt templates) land here.
    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        if ctx.config.filters.get_stage_config("double_critic") is None:
            # Stage was constructed but config absent → no-op (defensive).
            return records

        conversation_records: list[ConversationRecord] = [
            r for r in records if isinstance(r, ConversationRecord)
        ]
        non_conv = [r for r in records if not isinstance(r, ConversationRecord)]

        if not conversation_records:
            _write_filter_artifacts(
                self._output_writer,
                ctx,
                self.name,
                count_in=len(records),
                count_out=len(records),
                dropped=[],
                drop_reasons={},
            )
            return list(records)

        client = self._llm_client or LLMClient(config=ctx.config.llm)

        # Run all 2N critic calls concurrently with the same worker budget the
        # rest of the pipeline uses. Each record's two calls are independent; we
        # do not chain them, to keep the inverse-judgment property intact.
        worker_count = max(1, min(ctx.max_workers, 2 * len(conversation_records)))

        def _call_one(
            system: str, record: ConversationRecord
        ) -> CriticVerdict | LLMClientError:
            try:
                out = client.complete_structured(
                    messages=_build_messages(
                        system, record.payload.instruction, record.payload.response
                    ),
                    schema=CriticVerdict,
                )
            except LLMClientError as exc:
                return exc
            parsed = out.parsed
            if not isinstance(parsed, CriticVerdict):
                return LLMClientError(
                    "invalid_structured_response",
                    "Critic LLM did not return a CriticVerdict",
                )
            return parsed

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            yes_futs = [
                pool.submit(_call_one, _YES_SYSTEM, r) for r in conversation_records
            ]
            no_futs = [
                pool.submit(_call_one, _NO_SYSTEM, r) for r in conversation_records
            ]
            yes_results = [f.result() for f in yes_futs]
            no_results = [f.result() for f in no_futs]

        kept: list[Record] = list(non_conv)
        dropped: list[Record] = []
        drop_reasons: dict[str, int] = {}

        for record, yes_r, no_r in zip(
            conversation_records, yes_results, no_results, strict=True
        ):
            # Treat any LLM failure as a drop with a distinct reason — we don't
            # silently accept; we don't crash the pipeline on a flaky judge.
            if isinstance(yes_r, LLMClientError) or isinstance(no_r, LLMClientError):
                reason = "double_critic_llm_error"
                details = str(yes_r if isinstance(yes_r, LLMClientError) else no_r)
                dropped.append(_drop_record(record, self.name, reason, details))
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                continue

            audit = {
                "yes_verdict": yes_r.verdict,
                "no_verdict": no_r.verdict,
                "yes_reason": yes_r.reason,
                "no_reason": no_r.reason,
            }
            updated = record.model_copy(
                update={
                    "scores": record.scores.model_copy(
                        update={
                            "quality_per_dim": {
                                **record.scores.quality_per_dim,
                                "double_critic": audit,
                            },
                        }
                    )
                }
            )

            accept = yes_r.verdict == "yes" and no_r.verdict == "no"
            if accept:
                kept.append(updated)
            else:
                reason = "double_critic_disagreement"
                dropped.append(
                    _drop_record(
                        updated,
                        self.name,
                        reason,
                        f"yes={yes_r.verdict} no={no_r.verdict}",
                    )
                )
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

        _write_filter_artifacts(
            self._output_writer,
            ctx,
            self.name,
            count_in=len(records),
            count_out=len(kept),
            dropped=dropped,
            drop_reasons=drop_reasons,
        )
        return kept
