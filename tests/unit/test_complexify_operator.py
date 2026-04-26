"""Slice 2 — Simula `complexify` operator inside the Evol-Instruct registry.

Simula §2.2 "Optimizing Local Diversity and Complexity": for a fraction `c` of
generated samples, ask the model to rewrite the meta prompt + output to be
HARDER while preserving the original requirements.

We piggy-back on arka's existing Evol-Instruct operator registry. Adding the
operator is sufficient because the dispatch loop, lineage tracking, dedup
against parent, and refusal filtering already exist.

Tests:
1. `complexify` is registered in SUPPORTED_EVOL_OPERATORS
2. `build_evol_messages(parent, operator='complexify')` returns a prompt that
   asks the model to (a) preserve the task type and (b) increase complexity
3. The prompt explicitly forbids changing the topic / output format
4. Config validation accepts `operators: ['complexify']`
5. End-to-end: an EvolInstructRoundStage with `operators=['complexify']`
   produces a child record with `lineage.operator == 'complexify'`
"""

from __future__ import annotations

import json
from pathlib import Path

from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


def _parent_record() -> ConversationRecord:
    return ConversationRecord(
        id="seed-1",
        content_hash="hash-1",
        source=RecordSource(type="seed"),
        lineage=RecordLineage(root_id="root-seed-1", parent_ids=[]),
        payload=ConversationPayload(
            instruction="Explain gravity in two sentences.",
            response="Gravity is a force that attracts massive objects toward each other.",
        ),
        scores=RecordScores(),
        config_hash="cfg-1",
        created_at="2026-04-25T00:00:00Z",
    )


# --- Test 1: registered ------------------------------------------------------


def test_complexify_is_in_supported_operators() -> None:
    from arka.pipeline.evol_instruct import SUPPORTED_EVOL_OPERATORS

    assert "complexify" in SUPPORTED_EVOL_OPERATORS, (
        "complexify must be registered so generator config validation accepts it"
    )


# --- Test 2: prompt asks for harder, not different --------------------------


def test_complexify_prompt_asks_for_harder_version() -> None:
    from arka.pipeline.evol_instruct import build_evol_messages

    messages = build_evol_messages(_parent_record(), operator="complexify")
    prompt = " ".join(m["content"] for m in messages).lower()

    # The instruction must clearly ask for an increase in difficulty
    assert any(
        word in prompt
        for word in ("complex", "harder", "more difficult", "deeper", "edge case")
    ), f"complexify prompt must ask for higher complexity, got: {prompt[:300]}"


def test_complexify_prompt_preserves_task_and_format() -> None:
    """Critical: complexify must NOT pivot to a new topic (that's breadth_mutation's job).
    It must keep the same task type and same output format."""
    from arka.pipeline.evol_instruct import build_evol_messages

    messages = build_evol_messages(_parent_record(), operator="complexify")
    prompt = " ".join(m["content"] for m in messages).lower()

    assert "preserv" in prompt or "keep" in prompt or "same" in prompt, (
        "complexify must explicitly preserve task/format; otherwise it drifts to "
        "breadth_mutation behavior"
    )
    # Must explicitly forbid topic-switching (it's the boundary vs breadth_mutation).
    # The prompt is allowed (and expected) to mention 'topic' — but only in the
    # form of an explicit prohibition.
    assert (
        "do not switch" in prompt
        or "do not change" in prompt
        or "not pivot" in prompt
        or "same topic" in prompt
    ), (
        "complexify must explicitly prohibit topic switching; otherwise it overlaps "
        "with breadth_mutation"
    )


# --- Test 3: returned shape unchanged ---------------------------------------


def test_complexify_messages_have_user_role_and_json_instruction_key() -> None:
    """Mirror existing operator contract: returns one user message asking for
    JSON with key 'instruction'. Otherwise the EvolInstructRoundStage parser breaks.
    """
    from arka.pipeline.evol_instruct import build_evol_messages

    messages = build_evol_messages(_parent_record(), operator="complexify")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert '"instruction"' in messages[0]["content"]


# --- Test 4: config layer accepts complexify --------------------------------


def test_config_accepts_complexify_in_evol_operators() -> None:
    from arka.config.loader import ConfigLoader

    cfg_dict = {
        "version": "1",
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "k",
            "base_url": "https://api.openai.com/v1",
        },
        "executor": {"mode": "threadpool", "max_workers": 1},
        "data_source": {"type": "seeds", "path": "./seeds.jsonl"},
        "generator": {
            "type": "evol_instruct",
            "target_count": 2,
            "generation_multiplier": 1,
            "rounds": 1,
            "branching_factor": 1,
            "operators": ["complexify"],
        },
        "filters": {"target_count": 2, "stages": []},
        "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
    }
    config = ConfigLoader().load_dict(cfg_dict)
    assert config.generator.operators == ["complexify"]


# --- Test 5: end-to-end through EvolInstructRoundStage ---------------------


def test_evol_round_with_complexify_records_lineage(tmp_path: Path) -> None:
    """Verifies the child record carries operator='complexify' in its lineage,
    proving the operator is dispatchable through the existing stage with no
    code changes beyond the registry entry.
    """
    from arka.config.loader import ConfigLoader
    from arka.llm.models import LLMOutput, TokenUsage
    from arka.pipeline.evol_generator_stage import EvolInstructRoundStage
    from arka.pipeline.models import StageContext

    cfg_dict = {
        "version": "1",
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "k",
            "base_url": "https://api.openai.com/v1",
        },
        "executor": {"mode": "threadpool", "max_workers": 1},
        "data_source": {"type": "seeds", "path": "./seeds.jsonl"},
        "generator": {
            "type": "evol_instruct",
            "target_count": 2,
            "generation_multiplier": 1,
            "rounds": 1,
            "branching_factor": 1,
            "operators": ["complexify"],
        },
        "filters": {"target_count": 2, "stages": []},
        "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
    }
    config = ConfigLoader().load_dict(cfg_dict)

    ctx = StageContext(
        run_id="run-1",
        stage_name="03_evol_round_01",
        work_dir=tmp_path / "stages" / "03_evol_round_01",
        config=config,
        executor_mode=config.executor.mode,
        max_workers=config.executor.max_workers,
    )
    ctx.work_dir.mkdir(parents=True, exist_ok=True)

    # Sequential fake LLM: first call returns the complexified instruction;
    # second call returns the response for it.
    class _FakeLLM:
        def __init__(self, payloads: list[dict]) -> None:
            self._payloads = payloads
            self.calls = 0

        def complete_structured(self, messages, schema, **kwargs):
            # Tolerate temperature/max_tokens kwargs the real LLMClient passes.
            payload = self._payloads[self.calls]
            self.calls += 1
            text = json.dumps(payload)
            parsed = schema(**payload)
            return LLMOutput(
                text=text,
                parsed=parsed,
                usage=TokenUsage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
                finish_reason="stop",
                model="gpt-4o-mini",
                provider="openai",
                request_id=f"req-{self.calls}",
                latency_ms=5,
                error=None,
            )

    stage = EvolInstructRoundStage(
        round_number=1,
        llm_client=_FakeLLM(
            [
                {
                    "instruction": (
                        "Explain gravity in two sentences while accounting for both "
                        "the curvature of spacetime and the equivalence principle."
                    )
                },
                {
                    "response": (
                        "In Einstein's general relativity, gravity is the curvature of "
                        "spacetime caused by mass-energy. The equivalence principle "
                        "asserts inertial and gravitational mass are identical."
                    )
                },
            ]
        ),
    )

    out = stage.run([_parent_record()], ctx)

    assert len(out) == 2
    child = out[-1]
    assert child.lineage.operator == "complexify"
    assert child.lineage.parent_ids == ["seed-1"]
    assert child.source.type == "evolved"
