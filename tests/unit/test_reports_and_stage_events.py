from __future__ import annotations

import json
from pathlib import Path

from arka.pipeline.runner import PipelineRunner
from arka.pipeline.stages import Stage
from arka.records.models import Record, RecordLineage, RecordScores, RecordSource


class SourceStage(Stage):
    name = "01_source"

    def run(self, records: list[Record], ctx) -> list[Record]:
        return [
            Record(
                id="rec-1",
                content_hash="hash-1",
                source=RecordSource(type="generated"),
                lineage=RecordLineage(root_id="root-1", parent_ids=[]),
                payload={"value": "alpha"},
                scores=RecordScores(),
                config_hash="cfg-1",
                created_at="2026-04-04T00:00:00Z",
            )
        ]


class TransformStage(Stage):
    name = "02_transform"

    def run(self, records: list[Record], ctx) -> list[Record]:
        return [
            record.model_copy(
                update={"payload": {"value": str(record.payload["value"]).upper()}}
            )
            for record in records
        ]


CONFIG = {
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
        "target_count": 1,
        "generation_multiplier": 1,
    },
    "filters": {"target_count": 1},
    "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
}


def test_runner_appends_stage_events_and_writes_run_report(tmp_path: Path) -> None:
    runner = PipelineRunner(project_root=tmp_path)

    runner.run(
        config=CONFIG,
        stages=[SourceStage(), TransformStage()],
        run_id="run-1",
    )

    stage_path = (
        tmp_path / "runs" / "run-1" / "stages" / "02_transform" / "data.parquet"
    )
    report_path = tmp_path / "runs" / "run-1" / "report" / "run_report.json"

    restored = runner.output_writer.read_parquet(stage_path)
    assert restored[0].stage_events[0].stage == "01_source"
    assert restored[0].stage_events[0].seq == 1
    assert restored[0].stage_events[1].stage == "02_transform"
    assert restored[0].stage_events[1].seq == 2

    report = json.loads(report_path.read_text())
    assert report["run_id"] == "run-1"
    assert report["final_count"] == 1
    assert report["stage_yields"] == [
        {
            "stage": "01_source",
            "count_in": 0,
            "count_out": 1,
            "status": "completed",
            "resumed": False,
        },
        {
            "stage": "02_transform",
            "count_in": 1,
            "count_out": 1,
            "status": "completed",
            "resumed": False,
        },
    ]
    assert report["dataset_path"].endswith("output/dataset.jsonl")
