from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from arka.pipeline.checkpoint import CheckpointManager
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


class FailingStage(Stage):
    name = "02_transform"

    def run(self, records: list[Record], ctx) -> list[Record]:
        raise RuntimeError("boom")


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
        "dedup": {
            "exact": {"enabled": False},
            "near": {"enabled": False, "bands": 16, "rows": 8},
        },
        "filters": {"target_count": 2},
        "embeddings": {"provider": "huggingface", "model": "all-MiniLM-L6-v2"},
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


def test_pipeline_runner_resume_does_not_skip_failed_stage_checkpoint(
    tmp_path: Path, config_dict: dict
) -> None:
    runner = PipelineRunner(project_root=tmp_path)
    run_id = "run-1"
    run_dir = tmp_path / "runs" / run_id
    stage_path = run_dir / "stages" / "01_source" / "data.parquet"
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    OutputWriter().write_parquet([build_record("1", "alpha")], stage_path)

    checkpoint = CheckpointManager(tmp_path / "state.db")
    checkpoint.register_run(run_id=run_id, config_hash="abc123", status="failed")
    checkpoint.save_stage(
        run_id=run_id,
        stage_name="01_source",
        artifact_path=stage_path,
        count_in=0,
        count_out=1,
        status="failed",
    )

    result = runner.run(
        config=config_dict,
        stages=[SourceStage(), TransformStage()],
        run_id=run_id,
        resume=True,
    )

    assert result.final_count == 1
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["stage_stats"][0]["status"] == "completed"
    assert manifest["stage_stats"][0]["resumed"] is False


def test_pipeline_runner_writes_report_samples_and_canaries_artifacts(
    tmp_path: Path, config_dict: dict
) -> None:
    runner = PipelineRunner(project_root=tmp_path)

    runner.run(
        config=config_dict,
        stages=[SourceStage(), TransformStage()],
        run_id="run-artifacts",
    )

    samples_path = tmp_path / "runs" / "run-artifacts" / "report" / "samples.jsonl"
    canaries_path = tmp_path / "runs" / "run-artifacts" / "report" / "canaries.json"

    assert samples_path.exists()
    assert canaries_path.exists()

    samples = [
        json.loads(line)
        for line in samples_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(samples) == 1
    assert samples[0] == {"value": "ALPHA"}

    canaries = json.loads(canaries_path.read_text())
    assert canaries == {
        "known_good": [],
        "known_bad": [],
        "status": None,
    }


def test_pipeline_runner_marks_failed_run_and_persists_failure_report(
    tmp_path: Path, config_dict: dict
) -> None:
    runner = PipelineRunner(project_root=tmp_path)

    with pytest.raises(RuntimeError, match="boom"):
        runner.run(
            config=config_dict,
            stages=[SourceStage(), FailingStage()],
            run_id="run-fail",
        )

    run_dir = tmp_path / "runs" / "run-fail"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    report = json.loads((run_dir / "report" / "run_report.json").read_text())
    stage_one = run_dir / "stages" / "01_source" / "data.parquet"
    stage_two = run_dir / "stages" / "02_transform" / "data.parquet"

    assert stage_one.exists()
    assert not stage_two.exists()
    assert manifest["status"] == "failed"
    assert manifest["error"] == {
        "stage": "02_transform",
        "type": "RuntimeError",
        "message": "boom",
    }
    assert manifest["final_count"] == 1
    assert report["status"] == "failed"
    assert report["error"] == {
        "stage": "02_transform",
        "type": "RuntimeError",
        "message": "boom",
    }
    assert report["final_count"] == 1
    assert report["dataset_path"] is None
    assert report["stage_yields"] == [
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
            "status": "failed",
            "resumed": False,
            "dropped_count": 0,
            "error": {"type": "RuntimeError", "message": "boom"},
        },
    ]
    assert report["cost_usd"] is None

    checkpoint_runs = CheckpointManager(tmp_path / "state.db").list_runs()
    assert checkpoint_runs == [
        {
            "run_id": "run-fail",
            "config_hash": manifest["config_hash"],
            "status": "failed",
        }
    ]
