"""Tests for the PairDeltaFilterStage."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from arka.config.loader import ConfigLoader
from arka.pipeline.models import StageContext
from arka.pipeline.scoring_stages import PairDeltaFilterStage
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


def _record(
    record_id: str,
    instruction: str,
    response: str,
    *,
    quality: float | None = None,
    parent_ids: list[str] | None = None,
) -> ConversationRecord:
    return ConversationRecord(
        id=record_id,
        content_hash=f"hash-{record_id}",
        source=RecordSource(type="generated"),
        lineage=RecordLineage(
            root_id=f"root-{record_id}",
            parent_ids=parent_ids or [],
        ),
        payload=ConversationPayload(instruction=instruction, response=response),
        scores=RecordScores(quality=quality),
        config_hash="cfg-1",
        created_at="2026-04-14T00:00:00Z",
    )


def _ctx(tmp_path: Path, **pair_delta_overrides) -> StageContext:
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
                "stages": [{"type": "pair_delta", **pair_delta_overrides}],
            },
            "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
        }
    )
    work_dir = tmp_path / "runs" / "run-1" / "stages" / "04_pair_delta"
    work_dir.mkdir(parents=True, exist_ok=True)
    return StageContext(
        run_id="run-1",
        stage_name="04_pair_delta",
        work_dir=work_dir,
        config=config,
        executor_mode=config.executor.mode,
        max_workers=config.executor.max_workers,
    )


def test_pair_delta_keeps_records_with_sufficient_improvement(
    tmp_path: Path,
) -> None:
    parent = _record("parent-1", "Hello", "Old text", quality=0.3)
    child = _record(
        "child-1", "Hello", "Better text", quality=0.8, parent_ids=["parent-1"]
    )

    ctx = _ctx(tmp_path, score_field="quality", min_delta=0.30)
    stage = PairDeltaFilterStage()

    records = stage.run(
        records=[child],
        ctx=ctx,
        parent_records=[parent],
    )

    assert len(records) == 1
    assert records[0].id == "child-1"


def test_pair_delta_drops_records_with_insufficient_improvement(
    tmp_path: Path,
) -> None:
    parent = _record("parent-1", "Hello", "Old text", quality=0.5)
    child = _record(
        "child-1", "Hello", "Slightly better", quality=0.6, parent_ids=["parent-1"]
    )

    ctx = _ctx(tmp_path, score_field="quality", min_delta=0.30)
    stage = PairDeltaFilterStage()

    records = stage.run(
        records=[child],
        ctx=ctx,
        parent_records=[parent],
    )

    assert len(records) == 0

    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["dropped_count"] == 1
    assert stats["drop_reasons"] == {"insufficient_delta": 1}

    dropped = pl.read_parquet(ctx.work_dir / "dropped.parquet")
    assert dropped.height == 1


def test_pair_delta_keeps_records_without_parent(tmp_path: Path) -> None:
    """Records without a matching parent pass through unchanged."""
    orphan = _record("orphan-1", "Hello", "Orphan text", quality=0.5)

    ctx = _ctx(tmp_path, score_field="quality", min_delta=0.30)
    stage = PairDeltaFilterStage()

    records = stage.run(records=[orphan], ctx=ctx, parent_records=[])

    assert len(records) == 1
    assert records[0].id == "orphan-1"


def test_pair_delta_filters_by_length_ratio(tmp_path: Path) -> None:
    parent = _record("parent-1", "Hello", "Short.", quality=0.3)
    child = _record(
        "child-1",
        "Hello",
        "This is a much much much much much longer response that exceeds the ratio.",
        quality=0.9,
        parent_ids=["parent-1"],
    )

    ctx = _ctx(
        tmp_path,
        score_field="quality",
        min_delta=0.10,
        length_ratio_max=1.30,
    )
    stage = PairDeltaFilterStage()

    records = stage.run(
        records=[child],
        ctx=ctx,
        parent_records=[parent],
    )

    assert len(records) == 0

    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["drop_reasons"] == {"length_ratio_exceeded": 1}
