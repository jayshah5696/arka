from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from arka.config.loader import ConfigLoader
from arka.llm.models import SequenceScore
from arka.pipeline.ifd_stage import IFDFilterStage, compute_ifd, ifd_distribution
from arka.pipeline.models import StageContext
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


class FakeScoringLLMClient:
    def __init__(self, conditioned: list[float], unconditioned: list[float]) -> None:
        self.conditioned = conditioned
        self.unconditioned = unconditioned
        self.calls = 0

    def supports_sequence_scoring(self) -> bool:
        return True

    def score_response(self, *, messages, target_text: str) -> SequenceScore:
        score_list = self.conditioned if messages[0]["content"] else self.unconditioned
        mean_logprob = score_list[self.calls // 2]
        self.calls += 1
        return SequenceScore(
            token_count=max(len(target_text.split()), 1),
            mean_logprob=mean_logprob,
            total_logprob=mean_logprob * max(len(target_text.split()), 1),
            provider="openai",
            model="gpt-4o-mini",
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
        created_at="2026-04-05T00:00:00Z",
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
            "dedup": {
                "exact": {"enabled": False},
                "near": {"enabled": False, "bands": 16, "rows": 8},
            },
            "filters": {"target_count": 2, "ifd": {"enabled": True, "min_score": 0.2}},
            "embeddings": {"provider": "huggingface", "model": "all-MiniLM-L6-v2"},
            "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
        }
    )
    work_dir = tmp_path / "runs" / "run-1" / "stages" / "02e_ifd_filter"
    work_dir.mkdir(parents=True, exist_ok=True)
    return StageContext(
        run_id="run-1",
        stage_name="02e_ifd_filter",
        work_dir=work_dir,
        config=config,
        executor_mode=config.executor.mode,
        max_workers=config.executor.max_workers,
    )


def test_compute_ifd_prefers_conditioned_score() -> None:
    conditioned = SequenceScore(
        token_count=3,
        mean_logprob=-0.5,
        total_logprob=-1.5,
        provider="openai",
        model="gpt-4o-mini",
    )
    unconditioned = SequenceScore(
        token_count=3,
        mean_logprob=-1.0,
        total_logprob=-3.0,
        provider="openai",
        model="gpt-4o-mini",
    )

    assert compute_ifd(conditioned, unconditioned) == 0.5


def test_ifd_distribution_summarizes_scores() -> None:
    assert ifd_distribution([0.2, 0.4, 0.6]) == {
        "mean": 0.4,
        "std": 0.1633,
        "min": 0.2,
        "max": 0.6,
    }


def test_ifd_filter_stage_scores_and_drops_low_ifd(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    stage = IFDFilterStage(
        project_root=tmp_path,
        llm_client=FakeScoringLLMClient(
            conditioned=[-0.4, -1.0],
            unconditioned=[-0.8, -1.1],
        ),
    )

    result = stage.run(
        [
            _record("r1", "Explain gravity", "Gravity attracts masses."),
            _record("r2", "Tell me stuff", "Stuff."),
        ],
        ctx,
    )

    assert len(result) == 1
    assert result[0].id == "r1"
    assert result[0].scores.ifd == 0.4

    dropped = pl.read_parquet(ctx.work_dir / "dropped.parquet")
    assert dropped.height == 1
    assert dropped.select("drop_reason").to_series().to_list() == ["low_ifd"]

    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["scored_count"] == 2
    assert stats["dropped_count"] == 1
    assert stats["drop_reasons"] == {"low_ifd": 1}
    assert stats["ifd_distribution"] == {
        "mean": 0.25,
        "std": 0.15,
        "min": 0.1,
        "max": 0.4,
    }


def test_ifd_filter_stage_requires_supported_backend(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)

    class UnsupportedClient:
        def supports_sequence_scoring(self) -> bool:
            return False

    with pytest.raises(
        ValueError, match="IFD requires provider/model response-scoring capability"
    ):
        IFDFilterStage(project_root=tmp_path, llm_client=UnsupportedClient()).run(
            [_record("r1", "Explain gravity", "Gravity attracts masses.")],
            ctx,
        )
