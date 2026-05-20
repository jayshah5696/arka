from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from arka.config.loader import ConfigLoader
from arka.labeling.judges import JudgeResponse
from arka.llm.models import LLMOutput, TokenUsage
from arka.pipeline.models import StageContext
from arka.pipeline.scoring_stages import LabelingScoreStage
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


class SequentialFakeLLMClient:
    def __init__(self, responses: list[JudgeResponse]) -> None:
        self.responses = responses
        self.calls = 0

    def complete_structured(self, messages, schema: type[BaseModel]) -> LLMOutput:
        response = self.responses[self.calls]
        self.calls += 1
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


def _ctx(tmp_path: Path) -> StageContext:
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
                "stages": [
                    {
                        "type": "labeling_engine",
                        "rubric_path": "rubrics/sft_quality.yaml",
                    },
                ],
            },
            "labeling_engine": {
                "rubric_path": "rubrics/sft_quality.yaml",
                "mode": "single",
            },
            "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
        }
    )
    work_dir = tmp_path / "runs" / "run-1" / "stages" / "02s_label_score"
    work_dir.mkdir(parents=True, exist_ok=True)
    return StageContext(
        run_id="run-1",
        stage_name="02s_label_score",
        work_dir=work_dir,
        config=config,
        executor_mode=config.executor.mode,
        max_workers=config.executor.max_workers,
    )


def _write_rubric(tmp_path: Path) -> None:
    rubric_dir = tmp_path / "rubrics"
    rubric_dir.mkdir(parents=True, exist_ok=True)
    (rubric_dir / "sft_quality.yaml").write_text(
        """
version: "1.0"
description: "SFT quality rubric"
dimensions:
  - name: instruction_clarity
    description: "How clear is the instruction?"
    scale_min: 1
    scale_max: 5
    criteria:
      1: "Incomprehensible"
      5: "Crystal clear"
  - name: response_quality
    description: "How good is the response?"
    scale_min: 1
    scale_max: 5
    criteria:
      1: "Completely wrong"
      5: "Excellent"
overall_weights:
  instruction_clarity: 0.5
  response_quality: 0.5
"""
    )


def test_labeling_score_stage_annotates_all_records_without_dropping(
    tmp_path: Path,
) -> None:
    _write_rubric(tmp_path)
    ctx = _ctx(tmp_path)
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
        ]
    )
    stage = LabelingScoreStage(project_root=tmp_path, llm_client=fake_client)

    records = stage.run(
        [
            _record("1", "Explain gravity", "Gravity attracts masses."),
            _record("2", "Tell me stuff", "Stuff."),
        ],
        ctx,
    )

    # Both records kept — score stage does NOT filter
    assert len(records) == 2
    assert records[0].scores.quality == 5.0
    assert records[0].scores.quality_per_dim == {
        "instruction_clarity": 5,
        "response_quality": 5,
    }
    assert records[0].scores.rubric_version == "1.0"
    assert records[0].scores.judge_model == "gpt-4o-mini"

    assert records[1].scores.quality == 1.0
    assert records[1].scores.quality_per_dim == {
        "instruction_clarity": 1,
        "response_quality": 1,
    }

    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["stage"] == "02s_label_score"
    assert stats["count_in"] == 2
    assert stats["count_out"] == 2
    assert stats["scored_count"] == 2
    assert stats["quality_distribution"]["mean"] == 3.0


def test_labeling_score_stage_passes_non_conversation_records_unchanged(
    tmp_path: Path,
) -> None:
    from arka.records.models import Record

    _write_rubric(tmp_path)
    ctx = _ctx(tmp_path)
    stage = LabelingScoreStage(
        project_root=tmp_path, llm_client=SequentialFakeLLMClient([])
    )

    plain_record = Record(
        id="plain-1",
        content_hash="hash-plain-1",
        source=RecordSource(type="generated"),
        lineage=RecordLineage(root_id="root-plain-1", parent_ids=[]),
        payload={"value": "alpha"},
        scores=RecordScores(),
        config_hash="cfg-1",
        created_at="2026-04-14T00:00:00Z",
    )

    records = stage.run([plain_record], ctx)

    assert len(records) == 1
    assert records[0].id == "plain-1"
    assert records[0].scores.quality is None
