from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from arka.pipeline.runner import PipelineRunner
from arka.pipeline.stages import Stage
from arka.records.models import (
    Record,
    RecordLineage,
    RecordScores,
    RecordSource,
    StageEvent,
)


class SourceStage(Stage):
    name = "01_source"
    stage_action = "sourced"

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


class ConversationSourceStage(Stage):
    name = "01_source"
    stage_action = "sourced"

    def run(self, records: list[Record], ctx) -> list[Record]:
        from arka.records.models import ConversationPayload, ConversationRecord

        return [
            ConversationRecord(
                id="conv-1",
                content_hash="hash-conv-1",
                source=RecordSource(type="generated"),
                lineage=RecordLineage(root_id="root-conv-1", parent_ids=[]),
                payload=ConversationPayload(
                    instruction="Explain gravity simply.",
                    response="Gravity pulls objects toward each other.",
                ),
                scores=RecordScores(),
                config_hash="cfg-1",
                created_at="2026-04-04T00:00:00Z",
            ),
            ConversationRecord(
                id="conv-2",
                content_hash="hash-conv-2",
                source=RecordSource(type="generated"),
                lineage=RecordLineage(root_id="root-conv-2", parent_ids=[]),
                payload=ConversationPayload(
                    instruction="Explain photosynthesis simply.",
                    response="Plants use sunlight to make food.",
                ),
                scores=RecordScores(),
                config_hash="cfg-1",
                created_at="2026-04-04T00:00:00Z",
            ),
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


class FilteringStage(Stage):
    name = "03_filter"
    stage_action = "filtered"

    def __init__(self, *, cost_usd: float | None = None) -> None:
        self.cost_usd = cost_usd

    def run(self, records: list[Record], ctx) -> list[Record]:
        kept_records: list[Record] = []
        dropped_records = []
        for record in records:
            if record.id == "rec-1":
                kept_records.append(record)
                continue
            dropped_records.append(
                record.model_copy(
                    update={
                        "stage_events": [
                            *record.stage_events,
                            StageEvent(
                                stage=self.name,
                                action="dropped",
                                reason_code="too_short",
                                seq=len(record.stage_events) + 1,
                            ),
                        ]
                    }
                )
            )
        ctx.work_dir.joinpath("stats.json").write_text(
            json.dumps(
                {
                    "stage": self.name,
                    "count_in": len(records),
                    "count_out": len(kept_records),
                    "dropped_count": len(dropped_records),
                    "drop_reasons": {"too_short": len(dropped_records)},
                    "quality_distribution": {
                        "mean": 4.5,
                        "std": 0.5,
                        "min": 4.0,
                        "max": 5.0,
                    },
                    "cost_usd": self.cost_usd,
                }
            )
        )
        return kept_records


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
    "dedup": {
        "exact": {"enabled": False},
        "near": {"enabled": False, "bands": 16, "rows": 8},
    },
    "filters": {"target_count": 1},
    "embeddings": {"provider": "huggingface", "model": "all-MiniLM-L6-v2"},
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
    assert restored[0].stage_events[0].action == "sourced"
    assert restored[0].stage_events[0].seq == 1
    assert restored[0].stage_events[1].stage == "02_transform"
    assert restored[0].stage_events[1].action == "transformed"
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
    assert report["cost_usd"] is None
    assert report["dataset_path"].endswith("output/dataset.jsonl")


def test_runner_includes_stage_stats_and_drop_reasons_from_stage_stats_file(
    tmp_path: Path,
) -> None:
    runner = PipelineRunner(project_root=tmp_path)

    runner.run(
        config=CONFIG,
        stages=[
            SourceStage(),
            TransformStage(),
            FilteringStage(),
        ],
        run_id="run-with-filter",
    )

    report_path = tmp_path / "runs" / "run-with-filter" / "report" / "run_report.json"
    report = json.loads(report_path.read_text())

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
            "status": "completed",
            "resumed": False,
            "dropped_count": 0,
        },
        {
            "stage": "03_filter",
            "count_in": 1,
            "count_out": 1,
            "status": "completed",
            "resumed": False,
            "dropped_count": 0,
            "drop_reasons": {"too_short": 0},
            "quality_distribution": {
                "mean": 4.5,
                "std": 0.5,
                "min": 4.0,
                "max": 5.0,
            },
        },
    ]
    assert report["drop_reasons"] == {"too_short": 0}
    assert report["quality_distribution"] == {
        "mean": 4.5,
        "std": 0.5,
        "min": 4.0,
        "max": 5.0,
    }
    assert report["cost_usd"] is None


def test_run_report_includes_samples_diversity_and_canaries_when_available(
    tmp_path: Path, monkeypatch
) -> None:
    runner = PipelineRunner(project_root=tmp_path)
    monkeypatch.setattr(
        PipelineRunner,
        "_embed_texts_huggingface",
        lambda self, *, config, texts: np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    report_dir = tmp_path / "runs" / "run-rich-report" / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_dir.joinpath("canaries.json").write_text(
        json.dumps(
            {
                "known_good": [
                    {"id": "good-1", "expected": "high", "actual_score": 5.0}
                ],
                "known_bad": [{"id": "bad-1", "expected": "low", "actual_score": 1.0}],
                "status": "pass",
            }
        )
    )

    runner.run(
        config=CONFIG,
        stages=[ConversationSourceStage()],
        run_id="run-rich-report",
    )

    report_path = report_dir / "run_report.json"
    report = json.loads(report_path.read_text())
    assert report["samples_path"].endswith("samples.jsonl")
    assert report["canaries_path"].endswith("canaries.json")
    assert report["canaries"]["status"] == "pass"
    assert isinstance(report["diversity_score"], float)


def test_run_report_includes_cost_usd_when_stage_stats_provide_it(
    tmp_path: Path,
) -> None:
    runner = PipelineRunner(project_root=tmp_path)

    runner.run(
        config=CONFIG,
        stages=[SourceStage(), TransformStage(), FilteringStage(cost_usd=0.001)],
        run_id="run-with-cost",
    )

    report_path = tmp_path / "runs" / "run-with-cost" / "report" / "run_report.json"
    report = json.loads(report_path.read_text())

    assert report["cost_usd"] == 0.001
    assert report["stage_yields"][2]["cost_usd"] == 0.001
