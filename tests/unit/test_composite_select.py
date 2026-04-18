"""Tests for the CompositeSelectStage."""

from __future__ import annotations

import json
from pathlib import Path

from arka.config.loader import ConfigLoader
from arka.pipeline.models import StageContext
from arka.pipeline.scoring_stages import CompositeSelectStage
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


def _record(
    record_id: str,
    *,
    quality: float | None = None,
    reward_model: float | None = None,
    ifd: float | None = None,
) -> ConversationRecord:
    return ConversationRecord(
        id=record_id,
        content_hash=f"hash-{record_id}",
        source=RecordSource(type="generated"),
        lineage=RecordLineage(root_id=f"root-{record_id}", parent_ids=[]),
        payload=ConversationPayload(instruction=f"Q-{record_id}", response=f"A-{record_id}"),
        scores=RecordScores(quality=quality, reward_model=reward_model, ifd=ifd),
        config_hash="cfg-1",
        created_at="2026-04-14T00:00:00Z",
    )


def _ctx(tmp_path: Path, **select_overrides) -> StageContext:
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
            "dedup": {"exact": {"enabled": False}},
            "filters": {
                "target_count": 10,
                "select": {"enabled": True, **select_overrides},
            },
            "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
        }
    )
    work_dir = tmp_path / "runs" / "run-1" / "stages" / "05_composite_select"
    work_dir.mkdir(parents=True, exist_ok=True)
    return StageContext(
        run_id="run-1",
        stage_name="05_composite_select",
        work_dir=work_dir,
        config=config,
        executor_mode=config.executor.mode,
        max_workers=config.executor.max_workers,
    )


def test_composite_select_keeps_top_n_by_weighted_score(tmp_path: Path) -> None:
    ctx = _ctx(
        tmp_path,
        target_count=2,
        weights={"quality": 0.5, "reward_model": 0.5},
        strategy="top_n",
    )
    stage = CompositeSelectStage()

    records = stage.run(
        [
            _record("1", quality=0.9, reward_model=0.8),   # composite = 0.85
            _record("2", quality=0.3, reward_model=0.2),   # composite = 0.25
            _record("3", quality=0.7, reward_model=0.6),   # composite = 0.65
        ],
        ctx,
    )

    assert len(records) == 2
    assert [r.id for r in records] == ["1", "3"]

    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["count_in"] == 3
    assert stats["count_out"] == 2
    assert stats["dropped_count"] == 1


def test_composite_select_handles_missing_scores_as_zero(tmp_path: Path) -> None:
    ctx = _ctx(
        tmp_path,
        target_count=1,
        weights={"quality": 0.5, "ifd": 0.5},
    )
    stage = CompositeSelectStage()

    records = stage.run(
        [
            _record("1", quality=0.8, ifd=None),      # composite = 0.4
            _record("2", quality=0.3, ifd=0.9),        # composite = 0.6
        ],
        ctx,
    )

    assert len(records) == 1
    assert records[0].id == "2"


def test_composite_select_passes_all_when_disabled(tmp_path: Path) -> None:
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
            "dedup": {"exact": {"enabled": False}},
            "filters": {"target_count": 10},
            "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
        }
    )
    work_dir = tmp_path / "runs" / "run-1" / "stages" / "05_composite_select"
    work_dir.mkdir(parents=True, exist_ok=True)
    ctx = StageContext(
        run_id="run-1",
        stage_name="05_composite_select",
        work_dir=work_dir,
        config=config,
        executor_mode=config.executor.mode,
        max_workers=config.executor.max_workers,
    )
    stage = CompositeSelectStage()

    all_records = [_record("1", quality=0.5), _record("2", quality=0.3)]
    result = stage.run(all_records, ctx)

    assert len(result) == 2
