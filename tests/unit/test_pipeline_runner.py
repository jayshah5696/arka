from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from arka.pipeline.output import OutputWriter
from arka.pipeline.runner import PipelineRunner
from arka.pipeline.stages import Stage
from arka.records.models import Record, RecordLineage, RecordScores, RecordSource


def build_record(record_id: str, value: str) -> Record:
    return Record(
        id=record_id,
        content_hash=f"hash-{record_id}",
        source=RecordSource(type="generated"),
        lineage=RecordLineage(root_id=f"root-{record_id}", parent_ids=[]),
        payload={"value": value},
        scores=RecordScores(),
        config_hash="cfg-1",
        created_at="2026-04-04T00:00:00Z",
    )


class SourceStage(Stage):
    name = "01_source"

    def run(self, records: list[Record], ctx) -> list[Record]:
        assert records == []
        return [build_record("1", "alpha")]


class TransformStage(Stage):
    name = "02_transform"

    def run(self, records: list[Record], ctx) -> list[Record]:
        return [
            record.model_copy(
                update={
                    "payload": {
                        **record.payload,
                        "value": str(record.payload["value"]).upper(),
                    }
                }
            )
            for record in records
        ]


@pytest.fixture
def config_dict() -> dict:
    return {
        "version": "1",
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
        },
        "executor": {"mode": "threadpool", "max_workers": 2},
        "data_source": {"type": "seeds", "path": "./seeds.jsonl"},
        "generator": {
            "type": "prompt_based",
            "target_count": 2,
            "generation_multiplier": 1,
        },
        "filters": {"target_count": 2},
        "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
    }


def test_pipeline_runner_writes_stage_parquet_and_jsonl(
    tmp_path: Path, config_dict: dict
) -> None:
    runner = PipelineRunner(project_root=tmp_path)

    result = runner.run(
        config=config_dict,
        stages=[SourceStage(), TransformStage()],
        run_id="run-1",
    )

    stage_one = tmp_path / "runs" / "run-1" / "stages" / "01_source" / "data.parquet"
    stage_two = tmp_path / "runs" / "run-1" / "stages" / "02_transform" / "data.parquet"
    dataset_path = tmp_path / "output" / "dataset.jsonl"

    assert result.final_count == 1
    assert stage_one.exists()
    assert stage_two.exists()
    assert dataset_path.exists()

    frame = pl.read_parquet(stage_two)
    assert frame.columns == [
        "record_type",
        "id",
        "content_hash",
        "source_json",
        "lineage_json",
        "payload_json",
        "scores_json",
        "stage_events_json",
        "config_hash",
        "created_at",
    ]

    restored_records = OutputWriter().read_parquet(stage_two)
    assert [record.payload for record in restored_records] == [{"value": "ALPHA"}]
    assert dataset_path.read_text().strip() == '{"value":"ALPHA"}'

    manifest = json.loads((tmp_path / "runs" / "run-1" / "manifest.json").read_text())
    assert manifest["stage_stats"] == [
        {
            "stage": "01_source",
            "count_in": 0,
            "count_out": 1,
            "status": "completed",
            "resumed": False,
            "dropped_count": 0,
        },
        {
            "stage": "02_transform",
            "count_in": 1,
            "count_out": 1,
            "status": "completed",
            "resumed": False,
            "dropped_count": 0,
        },
    ]


def test_pipeline_runner_resume_skips_completed_stages(
    tmp_path: Path, config_dict: dict
) -> None:
    runner = PipelineRunner(project_root=tmp_path)
    runner.run(
        config=config_dict, stages=[SourceStage(), TransformStage()], run_id="run-1"
    )

    class FailingSourceStage(SourceStage):
        def run(self, records: list[Record], ctx) -> list[Record]:
            raise AssertionError("source stage should have been skipped on resume")

    resumed = runner.run(
        config=config_dict,
        stages=[FailingSourceStage(), TransformStage()],
        run_id="run-1",
        resume=True,
    )

    assert resumed.final_count == 1

    manifest = json.loads((tmp_path / "runs" / "run-1" / "manifest.json").read_text())
    assert manifest["stage_stats"] == [
        {
            "stage": "01_source",
            "count_in": 1,
            "count_out": 1,
            "status": "resumed",
            "resumed": True,
            "dropped_count": 0,
        },
        {
            "stage": "02_transform",
            "count_in": 1,
            "count_out": 1,
            "status": "resumed",
            "resumed": True,
            "dropped_count": 0,
        },
    ]
