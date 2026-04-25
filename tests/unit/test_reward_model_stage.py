"""Tests for the RewardModelScoringStage."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from arka.config.loader import ConfigLoader
from arka.pipeline.models import StageContext
from arka.pipeline.scoring_stages import RewardModelScoringStage
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


class FakeRewardClient:
    """Fake LLM client that returns a scalar score via complete()."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls = 0

    def complete(self, messages, **kwargs):
        score = self.scores[self.calls]
        self.calls += 1

        class FakeOutput:
            text = str(score)
            usage = type(
                "U",
                (),
                {
                    "cost_usd": None,
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "total_tokens": 6,
                },
            )()
            model = "nvidia/reward-model"
            provider = "openai"
            latency_ms = 10
            finish_reason = "stop"
            request_id = f"req_{self.calls}"
            error = None
            parsed = None

        return FakeOutput()


def _record(record_id: str, instruction: str, response: str) -> ConversationRecord:
    return ConversationRecord(
        id=record_id,
        content_hash=f"hash-{record_id}",
        source=RecordSource(type="generated"),
        lineage=RecordLineage(root_id=f"root-{record_id}", parent_ids=[]),
        payload=ConversationPayload(instruction=instruction, response=response),
        scores=RecordScores(),
        config_hash="cfg-1",
        created_at="2026-04-14T00:00:00Z",
    )


def _ctx(tmp_path: Path, **reward_overrides) -> StageContext:
    config = ConfigLoader().load_dict(
        {
            "version": "1",
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
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
                "stages": [{"type": "reward_model", **reward_overrides}],
            },
            "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
        }
    )
    work_dir = tmp_path / "runs" / "run-1" / "stages" / "02r_reward_score"
    work_dir.mkdir(parents=True, exist_ok=True)
    return StageContext(
        run_id="run-1",
        stage_name="02r_reward_score",
        work_dir=work_dir,
        config=config,
        executor_mode=config.executor.mode,
        max_workers=config.executor.max_workers,
    )


def test_reward_model_stage_scores_all_records_without_filtering(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    stage = RewardModelScoringStage(llm_client=FakeRewardClient([0.85, 0.30]))

    records = stage.run(
        [
            _record("1", "Explain gravity", "Good answer about gravity."),
            _record("2", "Tell me stuff", "Stuff."),
        ],
        ctx,
    )

    assert len(records) == 2
    assert records[0].scores.reward_model == 0.85
    assert records[1].scores.reward_model == 0.30

    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["count_in"] == 2
    assert stats["count_out"] == 2


def test_reward_model_stage_filters_by_min_score_when_configured(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path, min_score=0.50)
    stage = RewardModelScoringStage(llm_client=FakeRewardClient([0.85, 0.30]))

    records = stage.run(
        [
            _record("1", "Explain gravity", "Good answer about gravity."),
            _record("2", "Tell me stuff", "Stuff."),
        ],
        ctx,
    )

    assert len(records) == 1
    assert records[0].id == "1"
    assert records[0].scores.reward_model == 0.85

    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["count_in"] == 2
    assert stats["count_out"] == 1
    assert stats["dropped_count"] == 1
    assert stats["drop_reasons"] == {"low_reward_score": 1}

    dropped = pl.read_parquet(ctx.work_dir / "dropped.parquet")
    assert dropped.height == 1
