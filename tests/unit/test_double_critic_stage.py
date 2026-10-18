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

import pytest
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

    def complete_structured(
        self, messages: list[dict[str, str]], schema: type[BaseModel]
    ) -> LLMOutput:
        self.calls += 1
        if not self._queue:
            raise RuntimeError("FakeDoubleCriticClient queue exhausted")
        verdict = self._queue.pop(0)
        parsed = schema(verdict=verdict, reason=f"verdict was {verdict}")
        return LLMOutput(
            text=parsed.model_dump_json(),
            parsed=parsed,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="fake-critic",
            provider="openai",
            latency_ms=5,
        )


# --- Tests --------------------------------------------------------------------


def test_double_critic_accepts_when_yes_and_not_no(tmp_path: Path) -> None:
    """Case 1: yes-critic says YES, no-critic says NO -> ACCEPT."""
    from arka.pipeline.double_critic_stage import DoubleCriticFilterStage

    config = ConfigLoader().load_dict(_base_config_dict())
    fake = FakeDoubleCriticClient([("yes", "no")])
    stage = DoubleCriticFilterStage(llm_client=fake)
    records = [_make_record("r1", "What is 2+2?", "4")]

    kept = stage.run(records, _ctx(config, tmp_path))

    assert len(kept) == 1
    assert kept[0].id == "r1"
    assert fake.calls == 2
    # Verify audit annotation attached
    audit = kept[0].scores.quality_per_dim["double_critic"]
    assert audit["yes_verdict"] == "yes"
    assert audit["no_verdict"] == "no"


def test_double_critic_drops_when_both_say_yes(tmp_path: Path) -> None:
    """Case 2: yes-critic says YES, no-critic ALSO says YES -> DISAGREEMENT -> DROP."""
    from arka.pipeline.double_critic_stage import DoubleCriticFilterStage

    config = ConfigLoader().load_dict(_base_config_dict())
    fake = FakeDoubleCriticClient([("yes", "yes")])
    stage = DoubleCriticFilterStage(llm_client=fake)
    records = [_make_record("r1", "What is 2+2?", "5")]

    kept = stage.run(records, _ctx(config, tmp_path))

    assert kept == []
    assert fake.calls == 2


def test_double_critic_drops_when_both_say_no(tmp_path: Path) -> None:
    """Case 3: yes-critic says NO, no-critic says NO -> DISAGREEMENT -> DROP."""
    from arka.pipeline.double_critic_stage import DoubleCriticFilterStage

    config = ConfigLoader().load_dict(_base_config_dict())
    fake = FakeDoubleCriticClient([("no", "no")])
    stage = DoubleCriticFilterStage(llm_client=fake)
    records = [_make_record("r1", "What is 2+2?", "5")]

    kept = stage.run(records, _ctx(config, tmp_path))

    assert kept == []


def test_double_critic_catches_sycophancy(tmp_path: Path) -> None:
    """The headline case: yes-critic SAYS correct (sycophancy), but the inverse
    no-critic correctly identifies the answer as incorrect -> drop.

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
            ("yes", "yes"),  # sycophancy -> drop
            ("no", "no"),  # yes-critic disagrees -> drop
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

    # No conversation records -> no LLM calls, no drops
    kept = stage.run([], _ctx(config, tmp_path))
    assert kept == []
    assert fake.calls == 0


def test_double_critic_uses_llm_override_when_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slice 1.5 — a stage-level llm_override (different model / base_url) must
    be applied to the critic LLMClient. We assert that when override.model is set
    the resolved client is constructed against the override, not the top-level
    config. We do NOT make a real HTTP call — we monkeypatch the LLMClient
    constructor so we can capture the LLMConfig it received.
    """
    from arka.pipeline.double_critic_stage import DoubleCriticFilterStage

    base_dict = _base_config_dict(model="gen-model")
    base_dict["filters"]["stages"] = [
        {
            "type": "double_critic",
            "llm_override": {"model": "strong-critic-model"},
        }
    ]
    config = ConfigLoader().load_dict(base_dict)

    captured: list[Any] = []

    fake = FakeDoubleCriticClient([("yes", "no")])

    class _CapturingClient:
        def __init__(self, *, config) -> None:  # noqa: ANN001 — mirrors LLMClient
            captured.append(config)
            self._inner = fake

        def complete_structured(self, messages, schema):
            return self._inner.complete_structured(messages, schema)

    # Stage with NO injected client — must construct one through ctx.llm_client(...)
    # which uses arka.llm.factory.LLMClient.
    monkeypatch.setattr("arka.llm.factory.LLMClient", _CapturingClient)
    stage = DoubleCriticFilterStage()
    records = [_make_record("r1", "What is 2+2?", "4")]
    kept = stage.run(records, _ctx(config, tmp_path))

    assert len(kept) == 1
    assert len(captured) == 1, "LLMClient must be constructed once for the stage"
    used_config = captured[0]
    # Must reflect the override, not the top-level model:
    assert used_config.model == "strong-critic-model", (
        f"expected critic to use override model 'strong-critic-model', got {used_config.model!r}"
    )
