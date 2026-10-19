from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from arka.config.loader import ConfigLoader
from arka.config.models import QualityRepairConfig
from arka.decisions.models import DecisionResult, ScoreAnswer
from arka.llm.models import LLMOutput, TokenUsage
from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.pipeline.quality_repair import QualityRepairStage
from arka.pipeline.stage_builder import StageBuilder
from arka.privacy.config import PIIConfig
from arka.privacy.runtime import build_policy
from arka.records.identity import content_hash
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    GroundedChunkPayload,
    GroundedChunkRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def complete_structured(self, messages, schema, **kwargs):
        self.calls.append(messages)
        result = next(self.outputs)
        if isinstance(result, Exception):
            raise result
        return LLMOutput(
            parsed=schema.model_validate(result),
            usage=TokenUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.01
            ),
            model="fixture",
            provider="openai",
            latency_ms=1,
        )


def judgment(score=4, **scores):
    return {
        "scores": {"correctness": score, "clarity": 4, **scores},
        "reasoning": "Judge feedback",
    }


def record(name="a", **payload):
    p = ConversationPayload(
        instruction=f"Explain {name}",
        response="Old answer",
        reasoning="Stale trace",
        system="Be precise",
        **payload,
    )
    return ConversationRecord(
        id=name,
        content_hash=content_hash(p),
        payload=p,
        source=RecordSource(type="seed"),
        lineage=RecordLineage(root_id=name, parent_ids=[]),
        scores=RecordScores(
            quality=99,
            reward_model=99,
            annotations={
                "taxonomy_nodes": {"topic": ["math"]},
                "double_critic": {"stale": True},
            },
        ),
        config_hash="old",
        created_at="2026-10-18T00:00:00Z",
    )


def context(tmp_path, fake, **options):
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text(
        yaml.safe_dump(
            {
                "version": "1",
                "description": "Answer quality",
                "dimensions": [
                    {
                        "name": name,
                        "description": name,
                        "scale_min": 0,
                        "scale_max": 4,
                        "criteria": {0: "Wrong", 4: "Good"},
                    }
                    for name in ("correctness", "clarity")
                ],
                "overall_weights": {"correctness": 0.5, "clarity": 0.5},
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
                "base_url": "http://localhost/v1",
            },
            "executor": {"mode": "threadpool", "max_workers": 1},
            "pipeline": [
                {
                    "type": "quality_repair",
                    "rubric_path": str(rubric),
                    "min_scores": {"correctness": 3, "clarity": 3},
                    "repair_dimensions": ["correctness"],
                    "max_repairs": 2,
                    **options,
                }
            ],
            "output": {"format": "jsonl", "path": "dataset.jsonl"},
            **(
                {
                    "decisions": {
                        "provider": "openai",
                        "model": "fixture",
                        "api_key": "dummy",
                        "base_url": "http://localhost/v1",
                    }
                }
                if options.get("backend") == "decision"
                else {}
            ),
        }
    )
    work = tmp_path / "runs/r/stages/03_quality_repair"
    work.mkdir(parents=True, exist_ok=True)
    ctx = StageContext(
        run_id="r",
        stage_name="03_quality_repair",
        work_dir=work,
        config=cfg,
        executor_mode="threadpool",
        max_workers=1,
        llm_client_factory=lambda config: fake,
    )
    return StageBuilder(cfg, tmp_path).build()[0], ctx


def test_pass_repair_lineage_metadata_and_order(tmp_path):
    fake = FakeLLM([judgment(), judgment(1), {"response": "Fixed answer"}, judgment()])
    stage, ctx = context(tmp_path, fake)
    passing, failed = record("pass"), record("fail")
    result = stage.run([passing, failed], ctx)
    assert [r.payload.instruction for r in result.kept] == [
        passing.payload.instruction,
        failed.payload.instruction,
    ]
    repaired = result.kept[1]
    assert result.kept[0].id == passing.id
    assert repaired.payload.response == "Fixed answer"
    assert repaired.payload.system == failed.payload.system
    assert repaired.payload.reasoning is None
    assert repaired.id != failed.id and repaired.content_hash == content_hash(
        repaired.payload
    )
    assert (
        repaired.lineage.parent_ids == [failed.id]
        and repaired.lineage.root_id == failed.lineage.root_id
    )
    assert repaired.scores.reward_model is None
    assert repaired.annotations["taxonomy_nodes"] == {"topic": ["math"]}
    assert "double_critic" not in repaired.annotations
    assert len(fake.calls) == 4
    assert "Judge feedback" in json.dumps(fake.calls[2])
    assert result.report.model_dump()["accepted_repairs"] == 1


def test_failures_limit_and_excluded_dimensions(tmp_path):
    fake = FakeLLM(
        [
            judgment(1),
            {"response": "Still wrong"},
            judgment(1),
            judgment(1),
            judgment(4, clarity=1),
        ]
    )
    stage, ctx = context(tmp_path, fake, max_repairs=1)
    result = stage.run([record("a"), record("b"), record("c")], ctx)
    assert result.kept == []
    assert [r.stage_events[-1].reason_code for r in result.dropped] == [
        "repair_exhausted",
        "repair_limit",
        "repair_ineligible_dimensions",
    ]
    assert len(fake.calls) == 5


def chunk():
    return GroundedChunkRecord(
        id="chunk-root",
        content_hash="chunk-content",
        source=RecordSource(type="pdf_chunk", doc_id="doc", chunk_id="doc:0"),
        lineage=RecordLineage(root_id="chunk-root", parent_ids=[]),
        payload=GroundedChunkPayload(
            text="Exact permitted evidence",
            doc_id="doc",
            chunk_idx=0,
            page_start=1,
            page_end=1,
            char_start=0,
            char_end=24,
            word_count=3,
            chunk_strategy="fixed",
        ),
        scores=RecordScores(),
        config_hash="old",
        created_at="2026-10-18T00:00:00Z",
    )


@pytest.mark.parametrize("evidence", ["missing", "wrong_root", "correct"])
def test_grounding_requires_matching_chunk_in_both_judgments(tmp_path, evidence):
    fake = FakeLLM([judgment(1), {"response": "Grounded fix"}, judgment()])
    stage, ctx = context(tmp_path, fake)
    grounded = record().model_copy(
        update={
            "source": RecordSource(type="generated", doc_id="doc", chunk_id="doc:0"),
            "lineage": RecordLineage(root_id="chunk-root", parent_ids=["chunk-root"]),
        }
    )
    if evidence != "missing":
        source = ctx.work_dir.parent / "01_source/data.parquet"
        source.parent.mkdir()
        c = chunk()
        if evidence == "wrong_root":
            c = c.model_copy(update={"id": "unrelated"})
        OutputWriter().write_parquet([c], source)
    result = stage.run([grounded], ctx)
    if evidence == "correct":
        assert len(result.kept) == 1
        assert all(
            "Exact permitted evidence" in json.dumps(call) for call in fake.calls
        )
    else:
        assert not result.kept and not fake.calls
        assert (
            result.dropped[0].stage_events[-1].reason_code == "repair_missing_evidence"
        )


def test_multiturn_is_explicitly_ineligible(tmp_path):
    fake = FakeLLM([])
    stage, ctx = context(tmp_path, fake)
    result = stage.run([record(turns=[{"role": "user", "content": "Follow up"}])], ctx)
    assert result.dropped[0].stage_events[-1].reason_code == "repair_unsupported_turns"
    assert not fake.calls


def test_resume_after_generation_reuses_initial_judgment_and_candidate(tmp_path):
    fake = FakeLLM(
        [
            judgment(1),
            {"response": "Fixed answer"},
            RuntimeError("provider unavailable"),
        ]
    )
    stage, ctx = context(tmp_path, fake)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        stage.run([record()], ctx)
    resumed = FakeLLM([judgment()])
    ctx = replace(ctx, resume=True, llm_client_factory=lambda config: resumed)
    result = QualityRepairStage(config=stage.config, project_root=tmp_path).run(
        [record()], ctx
    )
    assert len(resumed.calls) == 1 and result.kept[0].payload.response == "Fixed answer"
    assert result.report.model_dump()["llm_usage"]["requests"] == 3
    assert result.report.model_dump()["cost_usd"] == pytest.approx(0.03)


@pytest.mark.parametrize("change", ["config", "rubric", "input", "evidence"])
def test_resume_rejects_changed_identity(tmp_path, change):
    fake = FakeLLM([judgment()])
    stage, ctx = context(tmp_path, fake)
    stage.run([record()], ctx)
    ctx = replace(ctx, resume=True)
    records = [record()]
    if change == "config":
        stage.config = stage.config.model_copy(update={"max_repairs": 99})
    elif change == "rubric":
        path = Path(stage.config.rubric_path)
        path.write_text(path.read_text().replace("Answer quality", "Changed quality"))
    elif change == "input":
        records = [record("different")]
    else:
        source = ctx.work_dir.parent / "01_source/data.parquet"
        source.parent.mkdir()
        OutputWriter().write_parquet([chunk()], source)
    with pytest.raises(ValueError, match="new run ID"):
        if change != "input":
            stage.validate_resume(ctx)
        else:
            stage.run(records, ctx)
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "output",
    [
        judgment(99),
        {"scores": {"correctness": 4}, "reasoning": "incomplete"},
        {
            "scores": {"correctness": 4, "clarity": 4, "unexpected": 4},
            "reasoning": "extra",
        },
    ],
)
def test_invalid_judge_scores_fail_before_repair(tmp_path, output):
    fake = FakeLLM([output])
    stage, ctx = context(tmp_path, fake)
    with pytest.raises(ValueError, match="scores"):
        stage.run([record()], ctx)
    assert len(fake.calls) == 1


@pytest.mark.parametrize("with_pii", [False, True])
def test_malformed_repair_fails_without_a_second_attempt(tmp_path, with_pii):
    class RawMalformedLLM:
        def __init__(self):
            self.calls = []

        def complete_structured(self, messages, schema, **kwargs):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return LLMOutput(
                    parsed=schema.model_validate(judgment(1)),
                    usage=TokenUsage(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                        cost_usd=0.01,
                    ),
                    model="fixture",
                    provider="openai",
                    latency_ms=1,
                )
            return LLMOutput(
                text='{"instruction": "Changed task", "response": "alice@example.com"}',
                parsed=None,
                usage=TokenUsage(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    cost_usd=0.01,
                ),
                model="fixture",
                provider="openai",
                latency_ms=1,
            )

    fake = RawMalformedLLM()
    stage, ctx = context(tmp_path, fake)
    if with_pii:
        policy = build_policy(
            PIIConfig.model_validate(
                {
                    "backend": "regex",
                    "languages": ["en"],
                    "entities": ["email"],
                }
            ),
            tmp_path,
        )
        ctx = replace(ctx, privacy=policy)
    with pytest.raises(Exception) as exc_info:
        stage.run([record()], ctx)
    assert "alice@example.com" not in str(exc_info.value)
    assert len(fake.calls) == 2


def test_decision_scores_without_fabricated_criticism_and_uncertainty(tmp_path):
    fake = FakeLLM([{"response": "Fixed answer"}])
    stage, ctx = context(tmp_path, fake, backend="decision")

    class Decisions:
        def __init__(self):
            self.calls = 0

        def decide(self, state, questions):
            self.calls += 1
            scores = [1, 4] if self.calls == 1 else [4, 4]
            answers = {
                q.name: ScoreAnswer(
                    score=score,
                    confidence=1 if self.calls < 3 else 0.1,
                    probabilities={i: float(i == score) for i in range(5)},
                )
                for q, score in zip(questions, scores, strict=True)
            }
            return DecisionResult(
                answers=answers,
                model="fixture",
                provider="openai",
                requests_sent=1,
                usage=TokenUsage(prompt_tokens=7),
            )

        def close(self):
            pass

    decisions = Decisions()
    ctx = replace(ctx, decision_client_factory=lambda config, **kwargs: decisions)
    result = stage.run([record("a"), record("b")], ctx)
    assert len(result.kept) == 1
    assert result.dropped[0].stage_events[-1].reason_code == "repair_uncertain_judgment"
    assert "Judge feedback" not in json.dumps(fake.calls)
    assert decisions.calls == 3 and len(fake.calls) == 1
    assert result.report.model_dump()["decision_usage"]["requests"] == 3


def test_pii_rejection_preserves_usage_without_judging_candidate(tmp_path):
    fake = FakeLLM([judgment(1), {"response": "Contact alice@example.com"}])
    stage, ctx = context(tmp_path, fake)
    policy = build_policy(
        PIIConfig.model_validate(
            {
                "backend": "regex",
                "languages": ["en"],
                "entities": ["email"],
                "blocked_entities": ["email"],
            }
        ),
        tmp_path,
    )
    ctx = replace(ctx, privacy=policy)
    result = stage.run([record()], ctx)
    assert not result.kept
    assert result.dropped[0].stage_events[-1].reason_code.startswith("pii_")
    assert len(fake.calls) == 2
    assert result.report.model_dump()["llm_usage"]["requests"] == 2
    assert all(
        "alice@example.com" not in p.read_text() for p in ctx.work_dir.rglob("*.json")
    )


@pytest.mark.parametrize(
    "rejected_judgment", ["passing_initial", "failing_initial", "final"]
)
def test_pii_rejected_judgment_stops_record_and_survives_resume(
    tmp_path, rejected_judgment
):
    rejected = {
        **judgment(1 if rejected_judgment == "failing_initial" else 4),
        "reasoning": "Contact alice@example.com",
    }
    outputs = (
        [judgment(1), {"response": "Fixed answer"}, rejected]
        if rejected_judgment == "final"
        else [rejected]
    )
    fake = FakeLLM(outputs)
    stage, ctx = context(tmp_path, fake)
    policy = build_policy(
        PIIConfig.model_validate(
            {
                "backend": "regex",
                "languages": ["en"],
                "entities": ["email"],
                "blocked_entities": ["email"],
            }
        ),
        tmp_path,
    )
    ctx = replace(ctx, privacy=policy)
    result = stage.run([record()], ctx)
    assert not result.kept
    assert result.dropped[0].stage_events[-1].reason_code == "pii_blocked_entity"
    assert result.dropped[0].scores.judge_model != "fixture"
    stats = result.report.model_dump()
    assert stats["initially_passing"] == 0
    assert stats["accepted_repairs"] == 0
    assert stats["by_source"]["ungrouped"]["failed_records"] == int(
        rejected_judgment == "final"
    )
    assert stats["llm_usage"]["requests"] == len(outputs)
    assert stats["cost_usd"] == pytest.approx(0.01 * len(outputs))
    assert len(fake.calls) == len(outputs)
    assert all(
        "alice@example.com" not in p.read_text() for p in ctx.work_dir.rglob("*.json")
    )
    saved = json.loads(next(ctx.work_dir.glob("attempt-*.json")).read_text())
    judgment_key = "final" if rejected_judgment == "final" else "initial"
    assert saved[judgment_key]["drop_reason"] == "pii_blocked_entity"
    assert saved[judgment_key]["reasoning"] == "Contact [email]"
    resumed = stage.run([record()], replace(ctx, resume=True))
    assert not resumed.kept
    assert resumed.report.model_dump() == stats
    assert len(fake.calls) == len(outputs)


def test_resume_rejects_attempts_without_judgment_privacy_outcomes(tmp_path):
    fake = FakeLLM([judgment()])
    stage, ctx = context(tmp_path, fake)
    stage.run([record()], ctx)
    path = ctx.work_dir / "repair_identity.json"
    identity = json.loads(path.read_text())
    identity["version"] = 1
    path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="new run ID"):
        stage.validate_resume(replace(ctx, resume=True), records=[record()])
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "options",
    [
        {"min_scores": {"missing": 3}},
        {"min_scores": {"correctness": 9}},
        {"repair_dimensions": ["missing"]},
    ],
)
def test_invalid_rubric_contract_fails_without_calls(tmp_path, options):
    fake = FakeLLM([])
    stage, ctx = context(tmp_path, fake, **options)
    with pytest.raises(ValueError):
        stage.run([record()], ctx)
    assert not fake.calls


def test_required_bounds_and_registry(tmp_path):
    with pytest.raises(ValueError):
        QualityRepairConfig(
            rubric_path="r",
            min_scores={"correctness": 3},
            repair_dimensions=["correctness"],
            max_repairs=-1,
        )
    stage, _ = context(tmp_path, FakeLLM([]))
    assert isinstance(stage, QualityRepairStage)


def test_rerun_without_resume_preserves_upstream_artifacts(tmp_path):
    from arka.pipeline.runner import PipelineRunner

    (tmp_path / "seeds.jsonl").write_text(
        '{"instruction": "Explain gravity", "response": "Old answer"}\n'
    )
    fake = FakeLLM([judgment()])
    _, ctx = context(tmp_path, fake)
    from arka.config.models import SeedSourceConfig

    cfg = ctx.config.model_copy(
        update={
            "pipeline": [
                SeedSourceConfig(path=str(tmp_path / "seeds.jsonl")),
                *ctx.config.pipeline,
            ]
        }
    )
    stages = StageBuilder(cfg, tmp_path).build()
    runner = PipelineRunner(tmp_path)
    original_build = StageContext.llm_client
    StageContext.llm_client = lambda self, override=None: fake
    try:
        runner.run(cfg, stages, "r", resume=False)
        source_parquet = tmp_path / "runs/r/stages/01_source/data.parquet"
        source_bytes = source_parquet.read_bytes()
        with pytest.raises(
            ValueError,
            match="Quality repair artifacts exist; resume or use a new run ID",
        ):
            runner.run(cfg, stages, "r", resume=False)
        assert source_parquet.read_bytes() == source_bytes
        runner.run(cfg, stages, "r", resume=True)
        assert len(fake.calls) == 1
    finally:
        StageContext.llm_client = original_build


def test_sqlite_llm_cache_disabled_across_records_on_resume(tmp_path):
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    from arka.pipeline.checkpoint import CheckpointManager

    responses = iter(
        [
            judgment(4),
            judgment(1),
            {"response": "Fixed answer"},
            RuntimeError("interrupted before final judgment"),
            judgment(1),
        ]
    )
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs["messages"])
            item = next(responses)
            if isinstance(item, Exception):
                raise item
            return ChatCompletion(
                id="chatcmpl-1",
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=0,
                        message=ChatCompletionMessage(
                            role="assistant", content=json.dumps(item)
                        ),
                    )
                ],
                created=1,
                model="fixture",
                object="chat.completion",
                usage=CompletionUsage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
            )

    class FakeOpenAI:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    stage, ctx = context(tmp_path, FakeLLM([]))
    checkpoint = CheckpointManager(tmp_path / "state.sqlite")
    try:
        from arka.llm.client import LLMClient

        ctx = replace(
            ctx,
            checkpoint_manager=checkpoint,
            llm_client_factory=lambda cfg: LLMClient(
                cfg, client_factory=lambda _c: FakeOpenAI()
            ),
        )
        records = [
            record("shared").model_copy(
                update={
                    "payload": record("shared").payload.model_copy(
                        update={"response": "Fixed answer"}
                    )
                }
            ),
            record("shared"),
        ]
        with pytest.raises(RuntimeError, match="interrupted"):
            stage.run(records, ctx)
        assert len(calls) == 4
        resumed_ctx = replace(ctx, resume=True)
        result = QualityRepairStage(config=stage.config, project_root=tmp_path).run(
            records, resumed_ctx
        )
        assert len(calls) == 5
        assert len(result.kept) == 1
        assert result.dropped[0].stage_events[-1].reason_code == "repair_exhausted"
    finally:
        checkpoint.close()


def test_validate_resume_matches_protected_inputs_and_cached_decision_costs(tmp_path):
    fake = FakeLLM([judgment()])
    stage, ctx = context(tmp_path, fake)
    policy = build_policy(
        PIIConfig.model_validate(
            {
                "backend": "regex",
                "languages": ["en"],
                "entities": ["email"],
            }
        ),
        tmp_path,
    )
    ctx = replace(ctx, privacy=policy)
    raw_records = [
        record().model_copy(
            update={
                "payload": record().payload.model_copy(
                    update={"instruction": "Email alice@example.com"}
                )
            }
        )
    ]
    stage.run(raw_records, ctx)
    pii_report_path = ctx.work_dir / "pii_report.json"
    pii_report_bytes = pii_report_path.read_bytes()
    stage.validate_resume(replace(ctx, resume=True), records=raw_records)
    assert pii_report_path.read_bytes() == pii_report_bytes

    # Cached decision judgments with requests_sent=0 must not mark cost_complete False.
    dec_fake = FakeLLM([{"response": "Fixed answer"}])
    (tmp_path / "dec").mkdir()
    dec_stage, dec_ctx = context(tmp_path / "dec", dec_fake, backend="decision")

    class CachedDecisions:
        def __init__(self):
            self.calls = 0

        def decide(self, state, questions):
            self.calls += 1
            score = 1 if self.calls == 1 else 4
            cached = self.calls == 2
            answers = {
                q.name: ScoreAnswer(
                    score=score,
                    confidence=1,
                    probabilities={i: float(i == score) for i in range(5)},
                )
                for q in questions
            }
            return DecisionResult(
                answers=answers,
                model="fixture",
                provider="openai",
                requests_sent=0 if cached else 1,
                cache_hits=len(questions) if cached else 0,
                usage=TokenUsage(
                    prompt_tokens=0 if cached else 7,
                    cost_usd=None if cached else 0.01,
                ),
            )

        def close(self):
            pass

    dec_ctx = replace(
        dec_ctx, decision_client_factory=lambda config, **kwargs: CachedDecisions()
    )
    dec_result = dec_stage.run([record("a")], dec_ctx)
    dump = dec_result.report.model_dump()
    assert dump["cost_complete"] is True
    assert dump["repair_cost_complete"] is True
