"""Tests for the SentenceVarianceFilterStage."""

from __future__ import annotations

import json
from pathlib import Path

from arka.config.loader import ConfigLoader
from arka.pipeline.cheap_filters import SentenceVarianceFilterStage
from arka.pipeline.models import StageContext
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


def _record(record_id: str, response: str) -> ConversationRecord:
    return ConversationRecord(
        id=record_id,
        content_hash=f"hash-{record_id}",
        source=RecordSource(type="generated"),
        lineage=RecordLineage(root_id=f"root-{record_id}", parent_ids=[]),
        payload=ConversationPayload(
            instruction=f"Question {record_id}",
            response=response,
        ),
        scores=RecordScores(),
        config_hash="cfg-1",
        created_at="2026-04-14T00:00:00Z",
    )


def _ctx(tmp_path: Path, min_cv: float = 0.15) -> StageContext:
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
                "target_count": 2,
                "sentence_variance": {"enabled": True, "min_cv": min_cv},
            },
            "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
        }
    )
    work_dir = tmp_path / "runs" / "run-1" / "stages" / "02f_sentence_variance"
    work_dir.mkdir(parents=True, exist_ok=True)
    return StageContext(
        run_id="run-1",
        stage_name="02f_sentence_variance",
        work_dir=work_dir,
        config=config,
        executor_mode=config.executor.mode,
        max_workers=config.executor.max_workers,
    )


def test_keeps_varied_sentences_drops_uniform(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, min_cv=0.15)
    stage = SentenceVarianceFilterStage()

    varied = _record(
        "1",
        "Short. A much longer sentence with several extra words. Tiny.",
    )
    uniform = _record(
        "2",
        "Same length here. Same length here. Same length here. Same length here.",
    )

    records = stage.run([varied, uniform], ctx)

    assert len(records) == 1
    assert records[0].id == "1"

    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["dropped_count"] == 1
    assert stats["drop_reasons"] == {"low_sentence_variance": 1}


def test_passes_all_when_disabled(tmp_path: Path) -> None:
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
            "filters": {"target_count": 2},
            "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
        }
    )
    work_dir = tmp_path / "runs" / "run-1" / "stages" / "02f_sentence_variance"
    work_dir.mkdir(parents=True, exist_ok=True)
    ctx = StageContext(
        run_id="run-1",
        stage_name="02f_sentence_variance",
        work_dir=work_dir,
        config=config,
        executor_mode=config.executor.mode,
        max_workers=config.executor.max_workers,
    )
    stage = SentenceVarianceFilterStage()

    uniform = _record("1", "Same length. Same length. Same length.")
    result = stage.run([uniform], ctx)

    assert len(result) == 1


def test_keeps_records_with_only_one_sentence(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, min_cv=0.15)
    stage = SentenceVarianceFilterStage()

    single_sentence = _record(
        "1", "Just one sentence here without any period at the end"
    )
    result = stage.run([single_sentence], ctx)

    assert len(result) == 1
