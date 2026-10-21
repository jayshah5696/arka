"""Audit failures must remain distinct from quality failures.

Failure cases: order-sensitive sampling, lost completed judgments on interruption,
changed evidence/configuration on resume, missing evidence, privacy rejection,
uncertain decisions, small populations, and duplicate record identities.
"""

import json
from dataclasses import replace

import pytest
import yaml

from arka.config.loader import ConfigLoader
from arka.llm.models import LLMOutput, TokenUsage
from arka.pipeline.models import StageContext
from arka.pipeline.stage_builder import StageBuilder
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


def record(name, source="a", response="answer"):
    return ConversationRecord(
        id=name,
        content_hash=name,
        source=RecordSource(type="seed", seed_file_hash=source),
        lineage=RecordLineage(root_id=name, parent_ids=[]),
        payload=ConversationPayload(instruction=f"Task {name}", response=response),
        scores=RecordScores(quality=99),
        config_hash="fixture",
        created_at="2026-10-19",
    )


class Judge:
    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def complete_structured(self, messages, schema, **kwargs):
        self.calls.append(messages)
        if len(self.calls) == self.fail_at:
            raise RuntimeError("interrupted")
        return LLMOutput(
            parsed=schema.model_validate(
                {"scores": {"correctness": 1}, "reasoning": "weak"}
            ),
            usage=TokenUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.01
            ),
            model="fixture",
            provider="openai",
            latency_ms=1,
        )


def context(tmp_path, fake, **overrides):
    rubric = tmp_path / "rubric.yaml"
    if not rubric.exists():
        rubric.write_text(
            yaml.safe_dump(
                {
                    "version": "1",
                    "description": "Quality",
                    "dimensions": [
                        {
                            "name": "correctness",
                            "description": "Correct",
                            "scale_min": 0,
                            "scale_max": 4,
                            "criteria": {0: "Bad", 4: "Good"},
                        }
                    ],
                    "overall_weights": {"correctness": 1},
                }
            )
        )
    cfg = ConfigLoader().load_dict(
        {
            "version": "1",
            "llm": {
                "provider": "openai",
                "model": "fixture",
                "api_key": "dummy",
                "base_url": "http://localhost:9000/v1",
            },
            "executor": {"mode": "threadpool", "max_workers": 1},
            "pipeline": [
                {
                    "type": "source_audit",
                    "rubric_path": str(rubric),
                    "min_scores": {"correctness": 3},
                    "sample_size": 2,
                    **overrides,
                }
            ],
            "output": {"format": "jsonl", "path": "output.jsonl"},
        }
    )
    stage = StageBuilder(cfg, tmp_path).build()[0]
    return stage, StageContext(
        run_id="audit",
        stage_name=stage.name,
        work_dir=tmp_path / "audit",
        config=cfg,
        executor_mode="threadpool",
        max_workers=1,
        llm_client_factory=lambda cfg: fake,
    )


def test_sampling_is_per_source_order_independent_and_preserves_records(tmp_path):
    fake = Judge()
    stage, ctx = context(tmp_path, fake)
    records = [record(str(i), "a" if i < 6 else "b") for i in range(9)]
    result = stage.run(records, ctx)
    assert result.kept == records and not result.dropped
    stats = result.report.model_dump()
    assert stats["by_source"]["seeds:a"]["population"] == 6
    assert stats["by_source"]["seeds:b"]["population"] == 3
    assert all(s["sampled"] == s["failed"] == 2 for s in stats["by_source"].values())
    assert len(fake.calls) == 4
    sample = json.loads((ctx.work_dir / "audit.json").read_text())["sample_ids"]
    second = tmp_path / "second"
    second.mkdir()
    stage2, ctx2 = context(second, Judge())
    stage2.run(list(reversed(records)), ctx2)
    assert (
        json.loads((ctx2.work_dir / "audit.json").read_text())["sample_ids"] == sample
    )


def test_interrupted_audit_reuses_completed_judgments(tmp_path):
    fake = Judge(fail_at=2)
    stage, ctx = context(tmp_path, fake)
    records = [record(str(i)) for i in range(4)]
    with pytest.raises(RuntimeError, match="interrupted"):
        stage.run(records, ctx)
    resumed = Judge()
    result = stage.run(
        records, replace(ctx, resume=True, llm_client_factory=lambda cfg: resumed)
    )
    assert len(resumed.calls) == 1
    assert result.report.model_dump()["cost_usd"] == pytest.approx(0.02)
    stage.run(
        records, replace(ctx, resume=True, llm_client_factory=lambda cfg: resumed)
    )
    assert len(resumed.calls) == 1


@pytest.mark.parametrize("change", ["input", "rubric", "config", "fresh", "order"])
def test_audit_rejects_incompatible_resume_before_calls(tmp_path, change):
    fake = Judge()
    stage, ctx = context(tmp_path, fake)
    records = [record("a"), record("b")] if change == "order" else [record("a")]
    stage.run(records, ctx)
    initial_calls = len(fake.calls)
    if change == "input":
        records = [record("b")]
    elif change == "rubric":
        path = tmp_path / "rubric.yaml"
        path.write_text(path.read_text().replace("Quality", "Changed"))
    elif change == "config":
        stage.config = stage.config.model_copy(update={"sample_size": 3})
    elif change == "order":
        records.reverse()
    with pytest.raises(ValueError, match="resume|new run ID"):
        stage.run(records, replace(ctx, resume=change != "fresh"))
    assert len(fake.calls) == initial_calls


def test_missing_grounding_is_reported_without_a_judgment(tmp_path):
    fake = Judge()
    stage, ctx = context(tmp_path, fake)
    grounded = record("a").model_copy(
        update={
            "source": RecordSource(type="generated", doc_id="doc", chunk_id="missing")
        }
    )
    result = stage.run([grounded], ctx)
    assert result.kept == [grounded]
    group = result.report.model_dump()["by_source"]["document:doc"]
    assert group["missing_evidence"] == 1 and group["judged"] == group["failed"] == 0
    assert group["failure_rate"] is None
    assert not fake.calls


def test_small_and_ungrouped_populations_are_explicit(tmp_path):
    stage, ctx = context(tmp_path, Judge(), sample_size=10)
    result = stage.run([record("a", source=None)], ctx)
    group = result.report.model_dump()["by_source"]["ungrouped"]
    assert group["population"] == group["sampled"] == 1
    assert group["census"] is True
    assert "recommendation" not in group


def test_duplicate_ids_fail_before_provider_calls(tmp_path):
    fake = Judge()
    stage, ctx = context(tmp_path, fake)
    with pytest.raises(ValueError, match="duplicate"):
        stage.run([record("a"), record("a")], ctx)
    assert not fake.calls
