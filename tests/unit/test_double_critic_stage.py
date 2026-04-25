"""Slice 1 — Double-critic filter stage (Simula §2.2).

For each (instruction, response) pair, run TWO independent critic calls:
  1. yes-critic: "Is this response correct?"      (yes/no)
  2. no-critic:  "Is this response incorrect?"    (yes/no)

ACCEPT iff yes_verdict=YES AND no_verdict=NO. Anything else → drop with reason
`double_critic_disagreement`.

Rationale (paper §3.1, Sharma et al. 2024 sycophancy): a single positively-framed
judge under-rejects because LLMs are biased toward affirmation. The negatively-framed
inverse critic catches errors the positive critic would let through.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from arka.config.loader import ConfigLoader
from arka.llm.models import LLMOutput, TokenUsage
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)

# --- Fixtures / helpers --------------------------------------------------


def _make_record(record_id: str, instruction: str, response: str) -> ConversationRecord:
    return ConversationRecord(
        id=record_id,
        content_hash=f"hash-{record_id}",
        source=RecordSource(type="generated"),
        lineage=RecordLineage(root_id=f"root-{record_id}", parent_ids=[]),
        payload=ConversationPayload(instruction=instruction, response=response),
        scores=RecordScores(),
        config_hash="cfg-1",
        created_at="2026-04-25T00:00:00Z",
    )


def _base_config_dict(model: str = "gpt-4o-mini") -> dict[str, Any]:
    return {
        "version": "1",
        "llm": {
            "provider": "openai",
            "model": model,
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
        },
        "executor": {"mode": "threadpool", "max_workers": 1},
        "data_source": {"type": "seeds", "path": "./seeds.jsonl"},
        "generator": {
            "type": "prompt_based",
            "target_count": 2,
            "generation_multiplier": 1,
        },
        "filters": {
            "target_count": 2,
            "stages": [{"type": "double_critic"}],
        },
        "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
    }


def _ctx(config, tmp_path: Path):
    from arka.pipeline.models import StageContext

    work_dir = tmp_path / "double_critic"
    work_dir.mkdir(parents=True, exist_ok=True)
    return StageContext(
        run_id="run-1",
        stage_name="03_double_critic",
        work_dir=work_dir,
        config=config,
        executor_mode=config.executor.mode,
        max_workers=config.executor.max_workers,
    )


class FakeDoubleCriticClient:
    """Returns a sequence of (yes_verdict, no_verdict, reason) tuples.

    Each record pulls TWO entries from the queue: yes-critic call then no-critic call.
    Verdict values are 'yes' / 'no' (lower-case).
    """

    def __init__(self, sequence: list[tuple[str, str]]) -> None:
        # Flatten: each record consumes (yes_call_result, no_call_result).
        # `sequence` is per-record [(yes_verdict, no_verdict), ...].
        self._queue: list[str] = []
        for yes_v, no_v in sequence:
            self._queue.append(yes_v)
            self._queue.append(no_v)
        self.calls = 0
        self.received_messages: list[list[dict]] = []

    def complete_structured(self, messages, schema: type[BaseModel]) -> LLMOutput:
        verdict = self._queue[self.calls]
        self.calls += 1
        self.received_messages.append(list(messages))
        parsed = schema(verdict=verdict, reason=f"reason-{self.calls}")
        return LLMOutput(
            text=json.dumps({"verdict": verdict, "reason": f"reason-{self.calls}"}),
            parsed=parsed,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="stop",
            model="gpt-4o-mini",
            provider="openai",
            request_id=f"req-{self.calls}",
            latency_ms=5,
            error=None,
        )


# --- Tests ---------------------------------------------------------------


def test_double_critic_accepts_when_both_agree(tmp_path: Path) -> None:
    """yes=YES + no=NO → record kept, scores attached for audit."""
    from arka.pipeline.double_critic_stage import DoubleCriticFilterStage

    config = ConfigLoader().load_dict(_base_config_dict())
    fake = FakeDoubleCriticClient([("yes", "no")])
    stage = DoubleCriticFilterStage(llm_client=fake)
    records = [_make_record("r1", "What is 2+2?", "4")]

    kept = stage.run(records, _ctx(config, tmp_path))

    assert len(kept) == 1
    assert fake.calls == 2  # one yes-call, one no-call per record
    assert kept[0].id == "r1"
    # Audit trail attached to scores
    audit = kept[0].scores.quality_per_dim.get("double_critic")
    assert audit is not None, "double_critic audit must be attached to scores"
    assert audit["yes_verdict"] == "yes"
    assert audit["no_verdict"] == "no"
    assert "yes_reason" in audit and "no_reason" in audit


def test_double_critic_rejects_when_yes_says_incorrect(tmp_path: Path) -> None:
    """yes=NO (not correct) → record dropped, even if no-critic disagrees."""
    from arka.pipeline.double_critic_stage import DoubleCriticFilterStage

    config = ConfigLoader().load_dict(_base_config_dict())
    fake = FakeDoubleCriticClient([("no", "no")])
    stage = DoubleCriticFilterStage(llm_client=fake)
    records = [_make_record("r1", "What is 2+2?", "5")]

    kept = stage.run(records, _ctx(config, tmp_path))

    assert kept == []


def test_double_critic_catches_sycophancy(tmp_path: Path) -> None:
    """The headline case: yes-critic SAYS correct (sycophancy), but the inverse
    no-critic correctly identifies the answer as incorrect → drop.

    Single-critic systems would have accepted this; double-critic catches it.
    """
    from arka.pipeline.double_critic_stage import DoubleCriticFilterStage

    config = ConfigLoader().load_dict(_base_config_dict())
    fake = FakeDoubleCriticClient([("yes", "yes")])  # both fire YES
    stage = DoubleCriticFilterStage(llm_client=fake)
    records = [_make_record("r1", "What is 2+2?", "5")]  # obviously wrong

    kept = stage.run(records, _ctx(config, tmp_path))

    assert kept == [], (
        "sycophancy case must be rejected (yes-critic agreed but no-critic flagged)"
    )


def test_double_critic_writes_drop_reason_in_stats(tmp_path: Path) -> None:
    """stats.json must record the disagreement count under drop_reasons."""
    from arka.pipeline.double_critic_stage import DoubleCriticFilterStage

    config = ConfigLoader().load_dict(_base_config_dict())
    fake = FakeDoubleCriticClient(
        [
            ("yes", "no"),  # accept
            ("yes", "yes"),  # sycophancy → drop
            ("no", "no"),  # yes-critic disagrees → drop
        ]
    )
    stage = DoubleCriticFilterStage(llm_client=fake)
    records = [
        _make_record("r1", "Q", "A1"),
        _make_record("r2", "Q", "A2"),
        _make_record("r3", "Q", "A3"),
    ]

    ctx = _ctx(config, tmp_path)
    kept = stage.run(records, ctx)

    assert len(kept) == 1
    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["count_in"] == 3
    assert stats["count_out"] == 1
    assert stats["dropped_count"] == 2
    assert stats["drop_reasons"]["double_critic_disagreement"] == 2


def test_double_critic_passes_through_non_conversation_records(tmp_path: Path) -> None:
    """Non-conversation records (e.g. PreferenceRecord) must pass through untouched."""
    from arka.pipeline.double_critic_stage import DoubleCriticFilterStage

    config = ConfigLoader().load_dict(_base_config_dict())
    fake = FakeDoubleCriticClient([])
    stage = DoubleCriticFilterStage(llm_client=fake)

    # No conversation records → no LLM calls, no drops
    kept = stage.run([], _ctx(config, tmp_path))
    assert kept == []
    assert fake.calls == 0


def test_double_critic_yes_and_no_prompts_are_distinct(tmp_path: Path) -> None:
    """The two critic calls must use semantically inverse prompts. We assert that
    the yes-call mentions 'correct' positively and the no-call mentions 'incorrect',
    so they elicit independent judgments rather than echo each other."""
    from arka.pipeline.double_critic_stage import DoubleCriticFilterStage

    config = ConfigLoader().load_dict(_base_config_dict())
    fake = FakeDoubleCriticClient([("yes", "no")])
    stage = DoubleCriticFilterStage(llm_client=fake)
    records = [_make_record("r1", "What is 2+2?", "4")]

    stage.run(records, _ctx(config, tmp_path))

    assert len(fake.received_messages) == 2
    yes_prompt = " ".join(m["content"] for m in fake.received_messages[0]).lower()
    no_prompt = " ".join(m["content"] for m in fake.received_messages[1]).lower()
    assert "correct" in yes_prompt
    assert "incorrect" in no_prompt
    assert yes_prompt != no_prompt
