"""Slice 5 \u2014 Simula \u00a72.3 calibrated batch-Elo complexity scoring.

For each record we want a `complexity_elo` score that is comparable across
datasets. The procedure (Davidson et al. 2026, \u00a72.3):

1. Sample records into batches of size B such that each record appears
   K times across batches (K=4 in our default).
2. For each batch, ask an M3 to RANK its members by complexity.
3. Decompose the rank into pairwise win/loss outcomes
   (B*(B-1)/2 pairs per batch).
4. Update a per-record Elo rating from those pairwise outcomes
   (default K-factor = 32, starting Elo = 400).
5. Attach the final Elo to `record.scores.quality_per_dim['complexity_elo']`.

Tests cover the deterministic Elo math (no LLM), the stage end-to-end with
a mock ranker, and the no-conversation-record passthrough.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


def _make(rid: str, instr: str, resp: str) -> ConversationRecord:
    return ConversationRecord(
        id=rid,
        content_hash=f"h-{rid}",
        source=RecordSource(type="generated"),
        lineage=RecordLineage(root_id=f"r-{rid}", parent_ids=[]),
        payload=ConversationPayload(instruction=instr, response=resp),
        scores=RecordScores(),
        config_hash="cfg-1",
        created_at="2026-04-25T00:00:00Z",
    )


def _config(tmp_path: Path) -> Any:
    from arka.config.loader import ConfigLoader

    return ConfigLoader().load_dict(
        {
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
                "type": "prompt_based",
                "target_count": 1,
                "generation_multiplier": 1,
            },
            "filters": {
                "target_count": 1,
                "stages": [
                    {
                        "type": "complexity_elo",
                        "batch_size": 3,
                        "samples_per_record": 2,
                        "k_factor": 32,
                    }
                ],
            },
            "output": {"format": "jsonl", "path": "./out.jsonl"},
        }
    )


def _ctx(config, tmp_path: Path):
    from arka.pipeline.models import StageContext

    work = tmp_path / "stage"
    work.mkdir(parents=True, exist_ok=True)
    return StageContext(
        run_id="run-1",
        stage_name="02s_complexity_elo",
        work_dir=work,
        config=config,
        executor_mode=config.executor.mode,
        max_workers=config.executor.max_workers,
    )


# --- Test 1: pairwise Elo math is correct -----------------------------------


def test_pairwise_elo_update_when_higher_wins() -> None:
    """Standard chess Elo: if ratings are equal (1500 vs 1500), the higher-
    expected score is 0.5; the actual outcome of 1.0 means winner gains
    K*(1-0.5)=16, loser loses 16."""
    from arka.pipeline.complexity_elo_stage import elo_update_pair

    new_a, new_b = elo_update_pair(rating_a=1500.0, rating_b=1500.0, a_wins=True, k=32)
    assert abs(new_a - 1516.0) < 1e-6
    assert abs(new_b - 1484.0) < 1e-6


def test_pairwise_elo_update_when_lower_wins_gains_more() -> None:
    """Upset: low rating beats high rating \u2014 low gains MORE than 16."""
    from arka.pipeline.complexity_elo_stage import elo_update_pair

    new_a, new_b = elo_update_pair(rating_a=1300.0, rating_b=1700.0, a_wins=True, k=32)
    delta_a = new_a - 1300.0
    delta_b = 1700.0 - new_b
    assert delta_a > 16.0
    assert delta_b > 16.0
    assert abs(delta_a - delta_b) < 1e-6  # zero-sum


# --- Test 2: end-to-end with deterministic mock ranker ----------------------


class _FakeRanker:
    """Returns a deterministic ranking for any batch.

    Ranks by inferring the record id's trailing integer: higher number == more
    complex. This lets us verify the Elo settles in the expected order.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.received_messages: list[Any] = []

    def complete_structured(self, messages, schema, **kwargs):
        from arka.llm.models import LLMOutput, TokenUsage

        self.calls += 1
        self.received_messages.append(list(messages))
        # Parse the batch ids from the prompt. Format: 'ITEM <id>:' lines.
        text = "\n".join(m["content"] for m in messages)
        ids: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("ITEM "):
                ids.append(line[len("ITEM ") :].split(":", 1)[0])
        # Rank by trailing integer in id, descending (higher = more complex)
        ranked = sorted(ids, key=lambda i: int(i.split("-")[-1]), reverse=True)
        parsed = schema(ranked_ids=ranked)
        return LLMOutput(
            text=json.dumps({"ranked_ids": ranked}),
            parsed=parsed,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="stop",
            model="gpt-4o-mini",
            provider="openai",
            request_id=f"r-{self.calls}",
            latency_ms=5,
            error=None,
        )


def test_complexity_elo_stage_orders_records_by_mock_ranker(tmp_path: Path) -> None:
    """If the ranker says 'rec-2 > rec-1 > rec-0' consistently across batches,
    the final Elo ratings must reflect that order."""
    from arka.pipeline.complexity_elo_stage import ComplexityEloScoringStage

    config = _config(tmp_path)
    fake = _FakeRanker()
    stage = ComplexityEloScoringStage(llm_client=fake, seed=0)

    records = [
        _make("rec-0", "Trivial Q", "A1"),
        _make("rec-1", "Medium Q", "A2"),
        _make("rec-2", "Hard Q", "A3"),
    ]
    out = stage.run(records, _ctx(config, tmp_path))

    elos = {r.id: r.scores.quality_per_dim["complexity_elo"] for r in out}
    assert elos["rec-2"] > elos["rec-1"] > elos["rec-0"], elos
    # Stage made at least one ranker call
    assert fake.calls >= 1


def test_complexity_elo_stage_passes_through_non_conversation(tmp_path: Path) -> None:
    """Non-conversation records flow through untouched; ranker is not called."""
    from arka.pipeline.complexity_elo_stage import ComplexityEloScoringStage

    config = _config(tmp_path)
    fake = _FakeRanker()
    stage = ComplexityEloScoringStage(llm_client=fake, seed=0)

    out = stage.run([], _ctx(config, tmp_path))
    assert out == []
    assert fake.calls == 0


def test_complexity_elo_stage_writes_distribution_in_stats(tmp_path: Path) -> None:
    """stats.json must include a complexity_elo distribution summary."""
    from arka.pipeline.complexity_elo_stage import ComplexityEloScoringStage

    config = _config(tmp_path)
    fake = _FakeRanker()
    stage = ComplexityEloScoringStage(llm_client=fake, seed=0)

    records = [_make(f"rec-{i}", f"Q{i}", f"A{i}") for i in range(4)]
    ctx = _ctx(config, tmp_path)
    stage.run(records, ctx)

    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["count_in"] == 4
    assert stats["count_out"] == 4
    dist = stats["complexity_elo_distribution"]
    assert dist["min"] is not None
    assert dist["max"] is not None
    assert dist["median"] is not None
