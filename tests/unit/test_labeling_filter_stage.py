from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from pydantic import BaseModel

from arka.config.loader import ConfigLoader
from arka.labeling.judges import JudgeResponse
from arka.llm.client import LLMClientError
from arka.llm.models import LLMOutput, TokenUsage
from arka.pipeline.filter_stages import LabelingQualityFilterStage
from arka.pipeline.models import StageContext
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


class SequentialFakeLLMClient:
    def __init__(self, responses: list[JudgeResponse | Exception]) -> None:
        self.responses = responses
        self.calls = 0

    def complete_structured(self, messages, schema: type[BaseModel]) -> LLMOutput:
        response = self.responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return LLMOutput(
            text="{}",
            parsed=response,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="stop",
            model="gpt-4o-mini",
            provider="openai",
            request_id=f"req_{self.calls}",
            latency_ms=10,
            error=None,
        )


def build_record(record_id: str, instruction: str, response: str) -> ConversationRecord:
    return ConversationRecord(
        id=record_id,
        content_hash=f"hash-{record_id}",
        source=RecordSource(type="generated"),
        lineage=RecordLineage(root_id=f"root-{record_id}", parent_ids=[]),
        payload=ConversationPayload(instruction=instruction, response=response),
        scores=RecordScores(),
        config_hash="cfg-1",
        created_at="2026-04-04T00:00:00Z",
    )


BASE_CONFIG = {
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
        "labeling_engine": {
            "enabled": True,
            "rubric_path": str(Path("rubrics/sft_quality.yaml").resolve()),
            "min_overall_score": 3.5,
        },
    },
    "labeling_engine": {
        "rubric_path": str(Path("rubrics/sft_quality.yaml").resolve()),
        "mode": "single",
    },
    "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
}


def test_labeling_filter_stage_scores_records_and_filters_low_quality(
    tmp_path: Path,
) -> None:
    config = ConfigLoader().load_dict(BASE_CONFIG)
    ctx = StageContext(
        run_id="run-1",
        stage_name="03_label_quality",
        work_dir=tmp_path / "work",
        config=config,
        executor_mode=config.executor.mode,
        max_workers=config.executor.max_workers,
    )
    fake_client = SequentialFakeLLMClient(
        [
            JudgeResponse(
                scores={"instruction_clarity": 5, "response_quality": 5},
                reasoning="excellent",
            ),
            JudgeResponse(
                scores={"instruction_clarity": 1, "response_quality": 1},
                reasoning="poor",
            ),
            JudgeResponse(
                scores={"instruction_clarity": 5, "response_quality": 5},
                reasoning="good canary",
            ),
            JudgeResponse(
                scores={"instruction_clarity": 1, "response_quality": 1},
                reasoning="bad canary",
            ),
        ]
    )
    stage = LabelingQualityFilterStage(project_root=tmp_path, llm_client=fake_client)

    records = stage.run(
        [
            build_record("1", "Explain gravity", "Gravity attracts masses."),
            build_record("2", "Tell me stuff", "Stuff."),
        ],
        ctx,
    )

    assert len(records) == 1
    kept = records[0]
    assert kept.scores.quality == 5.0
    assert kept.scores.quality_per_dim == {
        "instruction_clarity": 5,
        "response_quality": 5,
    }
    assert kept.scores.rubric_version == "1.0"
    assert kept.scores.judge_model == "gpt-4o-mini"

    dropped_path = ctx.work_dir / "dropped.parquet"
    stats_path = ctx.work_dir / "stats.json"

    assert dropped_path.exists()
    assert stats_path.exists()

    dropped_frame = pl.read_parquet(dropped_path)
    assert dropped_frame.height == 1
    assert dropped_frame.select("id").to_series().to_list() == ["2"]
    assert dropped_frame.select("drop_stage").to_series().to_list() == [
        "03_label_quality"
    ]
    assert dropped_frame.select("drop_reason").to_series().to_list() == [
        "low_quality_score"
    ]

    stats = json.loads(stats_path.read_text())
    assert stats["stage"] == "03_label_quality"
    assert stats["scored_count"] == 2
    assert stats["dropped_count"] == 1
    assert stats["drop_reasons"] == {"low_quality_score": 1}
    assert stats["quality_distribution"] == {
        "mean": 3.0,
        "std": 2.0,
        "min": 1.0,
        "max": 5.0,
    }


def test_labeling_filter_stage_classifies_parse_failures_as_dropped_records(
    tmp_path: Path,
) -> None:
    config = ConfigLoader().load_dict(BASE_CONFIG)
    ctx = StageContext(
        run_id="run-1",
        stage_name="03_label_quality",
        work_dir=tmp_path / "work",
        config=config,
        executor_mode=config.executor.mode,
        max_workers=config.executor.max_workers,
    )
    fake_client = SequentialFakeLLMClient(
        [
            LLMClientError("parse_error", "bad json"),
        ]
    )
    stage = LabelingQualityFilterStage(project_root=tmp_path, llm_client=fake_client)

    records = stage.run(
        [
            build_record("1", "Explain gravity", "Gravity attracts masses."),
            build_record("2", "Tell me stuff", "Stuff."),
        ],
        ctx,
    )

    assert records == []

    dropped_frame = pl.read_parquet(ctx.work_dir / "dropped.parquet")
    assert dropped_frame.height == 2
    assert dropped_frame.select("drop_reason").to_series().to_list() == [
        "label_parse_failure",
        "label_parse_failure",
    ]

    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["scored_count"] == 0
    assert stats["dropped_count"] == 2
    assert stats["drop_reasons"] == {"label_parse_failure": 2}
    assert stats["quality_distribution"] is None


def test_labeling_filter_stage_wraps_missing_rubric_path_with_config_context(
    tmp_path: Path,
) -> None:
    config = ConfigLoader().load_dict(
        {
            **BASE_CONFIG,
            "filters": {
                "target_count": 2,
                "labeling_engine": {
                    "enabled": True,
                    "rubric_path": "rubrics/missing.yaml",
                    "min_overall_score": 3.5,
                },
            },
        }
    )
    ctx = StageContext(
        run_id="run-1",
        stage_name="03_label_quality",
        work_dir=tmp_path / "work",
        config=config,
        executor_mode=config.executor.mode,
        max_workers=config.executor.max_workers,
    )
    stage = LabelingQualityFilterStage(project_root=tmp_path)

    with pytest.raises(
        ValueError,
        match=r"filters\.labeling_engine\.rubric_path points to a missing file: ",
    ):
        stage.run(
            [build_record("1", "Explain gravity", "Gravity attracts masses.")], ctx
        )
