from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError

from arka.config.loader import ConfigLoader
from arka.config.models import (
    ComplexityEloFilterConfig,
    DoubleCriticFilterConfig,
    LabelingFilterConfig,
    TaxonomyGeneratorConfig,
)
from arka.decisions.models import DecisionResult
from arka.llm.client import LLMClient, LLMClientError
from arka.llm.models import LLMOutput, TokenUsage
from arka.pipeline.complexity_elo_stage import ComplexityEloScoringStage
from arka.pipeline.double_critic_stage import DoubleCriticFilterStage
from arka.pipeline.filter_stages import LabelingQualityFilterStage
from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.pipeline.runner import PipelineRunner
from arka.pipeline.source_stages import SeedSourceStage
from arka.pipeline.stages import Stage
from arka.pipeline.taxonomy_generator import TaxonomyGeneratorStage
from arka.privacy.runtime import build_policy


def config(**pii):
    return ConfigLoader().load_dict(
        {
            "version": "1",
            "llm": {
                "provider": "openai",
                "model": "fixture",
                "api_key": "fixture",
                "base_url": "http://127.0.0.1:1/v1",
                "max_retries": 1,
                "supports_json_schema": True,
            },
            "decisions": {
                "provider": "openai",
                "model": "fixture",
                "api_key": "fixture",
                "base_url": "http://127.0.0.1:1/v1",
            },
            "executor": {"mode": "threadpool", "max_workers": 1},
            "pii": {
                "backend": "regex",
                "entities": ["email"],
                "languages": ["en"],
                **pii,
            },
            "pipeline": [{"type": "seed_source", "path": "seeds.jsonl"}],
            "output": {"format": "jsonl", "path": "dataset.jsonl"},
        }
    )


def context(tmp_path, cfg, name="stage"):
    return StageContext(
        run_id="recovery",
        stage_name=name,
        work_dir=tmp_path / name,
        config=cfg,
        executor_mode="threadpool",
        max_workers=1,
        privacy=build_policy(cfg.pii, tmp_path),
    )


def source(tmp_path, cfg):
    (tmp_path / "seeds.jsonl").write_text(
        '{"instruction":"Explain gravity", "response":"Mass attracts mass"}\n{"instruction":"Explain light", "response":"Light carries energy"}\n'
    )
    return SeedSourceStage(tmp_path).run([], context(tmp_path, cfg, "source"))


def broken_native_client(cfg, failure):
    def create(**kwargs):
        if failure == "provider":
            raise APIConnectionError(
                message="echo private@example.com",
                request=httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions"),
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="not json private@example.com"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=20, total_tokens=30
            ),
            model="fixture",
            id="fixture",
        )

    return LLMClient(
        cfg.llm,
        client_factory=lambda config: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )


@pytest.mark.parametrize("kind", ["label", "critic", "elo"])
@pytest.mark.parametrize("failure", ["provider", "malformed"])
def test_judges_keep_nonfatal_error_handling_with_pii(tmp_path, kind, failure):
    cfg = config()
    records = source(tmp_path, cfg)
    client = broken_native_client(cfg, failure)
    ctx = context(tmp_path, cfg, kind)
    rubric = tmp_path / "rubric.json"
    rubric.write_text(
        json.dumps(
            {
                "version": "1",
                "description": "Quality",
                "dimensions": [
                    {
                        "name": "clarity",
                        "description": "Clarity",
                        "scale_min": 1,
                        "scale_max": 5,
                        "criteria": {"1": "Poor", "5": "Great"},
                    }
                ],
                "overall_weights": {"clarity": 1},
            }
        )
    )
    if kind == "label":
        stage = LabelingQualityFilterStage(
            project_root=tmp_path,
            config=LabelingFilterConfig(rubric_path=str(rubric), min_overall_score=3),
            llm_client=client,
        )
    elif kind == "critic":
        stage = DoubleCriticFilterStage(
            config=DoubleCriticFilterConfig(), llm_client=client
        )
    else:
        stage = ComplexityEloScoringStage(
            config=ComplexityEloFilterConfig(), llm_client=client
        )

    result = stage.run(records, ctx)

    if kind == "elo":
        assert [r.id for r in result] == [r.id for r in records]
        assert {r.scores.quality_per_dim["complexity_elo"] for r in result} == {
            stage.DEFAULT_STARTING_ELO
        }
    else:
        assert result == []
        dropped = OutputWriter().read_parquet(ctx.work_dir / "dropped.parquet")
        expected = (
            "double_critic_llm_error"
            if kind == "critic"
            else (
                "label_retryable_api_failure"
                if failure == "provider"
                else "label_parse_failure"
            )
        )
        assert [r.stage_events[-1].reason_code for r in dropped] == [expected, expected]
        assert "private@example.com" not in json.dumps(
            [r.model_dump(mode="json") for r in dropped]
        )


@pytest.mark.parametrize(
    "method", ["complete", "complete_structured", "score_response"]
)
def test_guard_preserves_error_codes_without_provider_payload(tmp_path, method):
    class Client:
        def fail(self, **kwargs):
            raise LLMClientError("auth_error", "private@example.com")

        complete = complete_structured = score_response = fail

    ctx = context(tmp_path, config())
    kwargs = {"messages": []}
    if method == "complete_structured":
        from arka.pipeline.generator_stages import GeneratedConversation

        kwargs["schema"] = GeneratedConversation
    elif method == "score_response":
        kwargs["target_text"] = "Answer"
    with pytest.raises(LLMClientError) as caught:
        getattr(ctx.guard_llm(Client()), method)(**kwargs)
    assert caught.value.code == "auth_error"
    assert "private@example.com" not in str(caught.value)
    assert caught.value.__suppress_context__


def test_generator_rejection_mode_keeps_usage_without_changing_judge_client(tmp_path):
    from arka.pipeline.generator_stages import GeneratedConversation

    cfg = config()
    ctx = context(tmp_path, cfg)
    judge = ctx.guard_llm(broken_native_client(cfg, "malformed"))
    generator = ctx.guard_llm(judge, retain_rejected_output=True)
    output = generator.complete_structured(messages=[], schema=GeneratedConversation)
    assert output.usage.total_tokens == 30
    assert output.privacy.drop_reason == "pii_unparseable_response"
    assert "example.com" not in output.text
    with pytest.raises(LLMClientError) as caught:
        judge.complete_structured(messages=[], schema=GeneratedConversation)
    assert caught.value.code == "parse_error"


@pytest.mark.parametrize("structured", [False, True])
def test_pii_context_does_not_persist_unprotected_provider_output(tmp_path, structured):
    import sqlite3
    from dataclasses import replace

    from arka.pipeline.checkpoint import CheckpointManager
    from arka.pipeline.generator_stages import GeneratedConversation

    cfg = config()
    checkpoint = CheckpointManager(tmp_path / "state.sqlite")
    native = broken_native_client(cfg, "malformed")

    def factory(config, cache_db_path=None, read_cache=True):
        native.cache_db_path = cache_db_path
        native.read_cache = read_cache
        return native

    ctx = replace(
        context(tmp_path, cfg),
        checkpoint_manager=checkpoint,
        llm_client_factory=factory,
    )
    client = ctx.guard_llm(ctx.llm_client(), retain_rejected_output=True)
    if structured:
        output = client.complete_structured(messages=[], schema=GeneratedConversation)
        assert output.privacy.drop_reason == "pii_unparseable_response"
    else:
        output = client.complete(messages=[])
        assert output.text == "not json [email]"
    with sqlite3.connect(checkpoint.sqlite_path) as conn:
        rows = conn.execute("SELECT response_json FROM llm_cache").fetchall()
    assert "private@example.com" not in json.dumps(rows)
    checkpoint.close()


@pytest.mark.parametrize("resume", [True, False])
def test_retry_resets_pii_artifacts_but_completed_resume_preserves_them(
    tmp_path, resume
):
    cfg = config(blocked_entities=["email"])
    (tmp_path / "seeds.jsonl").write_text(
        '{"instruction":"private@example.com", "response":"Drop"}\n{"instruction":"Explain gravity", "response":"Keep"}\n'
    )

    class FlakySource(Stage):
        name = "01_source"
        calls = 0

        def run(self, records, ctx):
            self.calls += 1
            result = SeedSourceStage(tmp_path).run(records, ctx)
            if self.calls == 1:
                raise RuntimeError("failed after source protection")
            return result

    stage = FlakySource()
    runner = PipelineRunner(tmp_path)
    with pytest.raises(RuntimeError):
        runner.run(cfg, [stage], "retry")
    result = runner.run(cfg, [stage], "retry", resume=resume)
    stats = json.loads(result.manifest_path.read_text())["stage_stats"][0]
    path = tmp_path / "runs/retry/stages/01_source"
    dropped = OutputWriter().read_parquet(path / "pii_dropped.parquet")
    report = json.loads((path / "pii_report.json").read_text())
    assert stats["dropped_count"] == len(dropped) == 1
    assert stats["drop_reasons"] == {"pii_blocked_entity": 1}
    assert report["categories"] == {"email": 1}
    assert report["checked_units"] == 3  # Source records, then the one survivor.
    before = {
        name: (path / name).read_bytes()
        for name in ["pii_report.json", "pii_dropped.parquet"]
    }
    runner.run(cfg, [stage], "retry", resume=True)
    assert stage.calls == 2
    assert before == {name: (path / name).read_bytes() for name in before}


@pytest.mark.parametrize("rejection", ["blocked", "ratio", "malformed"])
@pytest.mark.parametrize("decision", [False, True])
def test_taxonomy_privacy_drops_match_artifacts_with_or_without_decisions(
    tmp_path, monkeypatch, rejection, decision
):
    pii = (
        {"blocked_entities": ["email"]}
        if rejection == "blocked"
        else {"max_redacted_ratio": 0.1}
        if rejection == "ratio"
        else {}
    )
    cfg = config(**pii)
    taxonomy = tmp_path / "taxonomy.json"
    taxonomy.write_text(
        json.dumps(
            {
                "factors": [
                    {
                        "factor": "topic",
                        "root": {"name": "topic", "children": [{"name": "science"}]},
                    }
                ]
            }
        )
    )
    raw = (
        "broken private\\u0040example.com"
        if rejection == "malformed"
        else '{"instruction":"Ask private@example.com", "response":"Answer"}'
    )

    class Client:
        def __init__(self):
            self.rows = iter(
                [
                    raw,
                    '{"instruction":"Explain gravity", "response":"Mass attracts mass"}',
                ]
            )

        def complete_structured(self, messages, schema, **kwargs):
            text = next(self.rows)
            try:
                parsed = schema.model_validate_json(text)
            except ValueError:
                parsed = None
            return LLMOutput(
                text=text,
                parsed=parsed,
                usage=TokenUsage(total_tokens=30, cost_usd=0.001),
                model="fixture",
                provider="openai",
                latency_ms=1,
            )

    def choose(records, *args, **kwargs):
        assert len(records) == 1
        return (
            [],
            [records[0].dropped_by("02_generate", "best_of_n_not_selected")],
            [DecisionResult(answers={}, model="fixture", provider="openai")],
        )

    monkeypatch.setattr("arka.pipeline.decision_checks.best_of_n", choose)
    ctx = context(tmp_path, cfg)
    stage = TaxonomyGeneratorStage(
        config=TaxonomyGeneratorConfig(
            taxonomy_path=str(taxonomy), target_count=2, best_of_n={"enabled": decision}
        ),
        llm_client=Client(),
    )

    kept = stage.run([], ctx)
    dropped = OutputWriter().read_parquet(ctx.work_dir / "dropped.parquet")
    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    expected = {
        "blocked": "pii_blocked_entity",
        "ratio": "pii_ratio",
        "malformed": "pii_unparseable_response",
    }[rejection]
    assert len(kept) == (0 if decision else 1)
    assert len(dropped) == stats["dropped_count"] == (2 if decision else 1)
    assert stats["drop_reasons"] == dict(
        Counter(r.stage_events[-1].reason_code for r in dropped)
    )
    privacy_drop = next(
        r for r in dropped if r.stage_events[-1].reason_code == expected
    )
    assert privacy_drop.payload.instruction == (
        "[PII_UNPARSEABLE_RESPONSE]" if rejection == "malformed" else "Ask [email]"
    )
    assert json.loads(privacy_drop.stage_events[-1].details)["drop_reason"] == expected
    assert stats["cost_usd"] == 0.002
    assert "example.com" not in json.dumps([r.model_dump(mode="json") for r in dropped])
