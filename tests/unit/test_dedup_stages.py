from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from arka.config.models import ResolvedConfig
from arka.pipeline.dedup_stages import ExactDedupStage, NearDedupStage
from arka.pipeline.models import StageContext
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


def _base_config(**overrides) -> ResolvedConfig:
    return ResolvedConfig(
        **{
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
                "target_count": 2,
                "generation_multiplier": 1,
            },
            "dedup": {"exact": {"enabled": False}, "near": {"enabled": False}},
            "filters": {"target_count": 2},
            "embeddings": {"provider": "huggingface", "model": "all-MiniLM-L6-v2"},
            "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
            **overrides,
        }
    )


def _record(
    record_id: str,
    instruction: str,
    response: str,
    content_hash: str,
) -> ConversationRecord:
    return ConversationRecord(
        id=record_id,
        content_hash=content_hash,
        source=RecordSource(type="generated"),
        lineage=RecordLineage(root_id=record_id, parent_ids=[]),
        payload=ConversationPayload(instruction=instruction, response=response),
        scores=RecordScores(),
        config_hash="cfg",
        created_at="2026-01-01T00:00:00Z",
    )


def _ctx(config: ResolvedConfig, tmp_path: Path, stage_name: str) -> StageContext:
    work_dir = tmp_path / stage_name
    work_dir.mkdir(parents=True, exist_ok=True)
    return StageContext(
        run_id="test-run",
        stage_name=stage_name,
        work_dir=work_dir,
        config=config,
        executor_mode="threadpool",
        max_workers=1,
    )


def test_exact_dedup_disabled_passes_all(tmp_path: Path) -> None:
    config = _base_config(dedup={"exact": {"enabled": False}})
    stage = ExactDedupStage()
    records = [
        _record("r1", "Question", "Answer", "hash-1"),
        _record("r2", "Question", "Answer", "hash-1"),
    ]

    result = stage.run(records, _ctx(config, tmp_path, stage.name))

    assert len(result) == 2


def test_exact_dedup_drops_duplicate_content_hash_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    config = _base_config(dedup={"exact": {"enabled": True}})
    stage = ExactDedupStage()
    records = [
        _record("r1", "Question", "Answer", "hash-1"),
        _record("r2", "Question", "Answer", "hash-1"),
        _record("r3", "Different", "Response", "hash-2"),
    ]
    ctx = _ctx(config, tmp_path, stage.name)

    result = stage.run(records, ctx)

    assert [record.id for record in result] == ["r1", "r3"]

    dropped_frame = pl.read_parquet(ctx.work_dir / "dropped.parquet")
    assert dropped_frame.height == 1
    assert dropped_frame.select("id").to_series().to_list() == ["r2"]
    assert dropped_frame.select("drop_stage").to_series().to_list() == [
        "02c_exact_dedup"
    ]
    assert dropped_frame.select("drop_reason").to_series().to_list() == [
        "exact_duplicate"
    ]
    assert dropped_frame.select("drop_detail").to_series().to_list() == [
        "duplicate_of=r1"
    ]

    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats == {
        "stage": "02c_exact_dedup",
        "count_in": 3,
        "count_out": 2,
        "dropped_count": 1,
        "drop_reasons": {"exact_duplicate": 1},
        "cluster_count": 1,
    }

    clusters = pl.read_parquet(ctx.work_dir / "clusters.parquet")
    assert clusters.height == 1
    assert clusters.select("representative_id").to_series().to_list() == ["r1"]
    assert clusters.select("member_count").to_series().to_list() == [2]
    member_ids = json.loads(clusters.select("member_ids_json").to_series().item())
    assert member_ids == ["r1", "r2"]


def test_near_dedup_drops_lexically_similar_instructions_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    config = _base_config(
        dedup={"exact": {"enabled": False}, "near": {"enabled": True}}
    )
    stage = NearDedupStage()
    records = [
        _record(
            "r1",
            "Explain machine learning in simple terms for beginners with examples and practical applications in business healthcare education science finance manufacturing retail logistics agriculture government research and everyday software products used by students teachers doctors analysts engineers managers and support teams around the world today.",
            "Machine learning lets computers learn from data.",
            "hash-1",
        ),
        _record(
            "r2",
            "Explain machine learning in simple terms for beginner with examples and practical applications in business healthcare education science finance manufacturing retail logistics agriculture government research and everyday software products used by students teachers doctors analysts engineers managers and support teams around the world today.",
            "ML means computers learn from examples.",
            "hash-2",
        ),
        _record(
            "r3",
            "How do plants make food?",
            "Through photosynthesis.",
            "hash-3",
        ),
    ]
    ctx = _ctx(config, tmp_path, stage.name)

    result = stage.run(records, ctx)

    assert [record.id for record in result] == ["r1", "r3"]

    dropped_frame = pl.read_parquet(ctx.work_dir / "dropped.parquet")
    assert dropped_frame.height == 1
    assert dropped_frame.select("id").to_series().to_list() == ["r2"]
    assert dropped_frame.select("drop_reason").to_series().to_list() == [
        "near_duplicate_minhash"
    ]

    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["stage"] == "02d_near_dedup"
    assert stats["count_in"] == 3
    assert stats["count_out"] == 2
    assert stats["dropped_count"] == 1
    assert stats["drop_reasons"] == {"near_duplicate_minhash": 1}
    assert stats["cluster_count"] == 1

    clusters = pl.read_parquet(ctx.work_dir / "clusters.parquet")
    assert clusters.height == 1
    assert clusters.select("representative_id").to_series().to_list() == ["r1"]
    member_ids = json.loads(clusters.select("member_ids_json").to_series().item())
    assert member_ids == ["r1", "r2"]


def test_near_dedup_config_validation_raises_error_for_invalid_bands() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(
        ValidationError, match="num_bands \* rows_per_band must equal num_hashes"
    ):
        _base_config(
            dedup={
                "near": {
                    "enabled": True,
                    "num_hashes": 128,
                    "num_bands": 10,
                    "rows_per_band": 10,
                }
            }
        )
