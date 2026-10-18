from __future__ import annotations

import json

import pytest

from arka.config.loader import ConfigLoader
from arka.config.models import EvolInstructGeneratorConfig
from arka.llm.models import LLMOutput, TokenUsage
from arka.pipeline.evol_generator_stage import EvolInstructRoundStage
from arka.pipeline.generator_stages import PromptBasedGeneratorStage
from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.pipeline.runner import PipelineRunner
from arka.pipeline.source_stages import SeedSourceStage
from arka.privacy.boundaries import ProtectedLLMClient
from arka.privacy.policy import PIIPolicy, Span
from arka.privacy.runtime import build_policy
from arka.records.identity import content_hash


def config(**pii):
    return ConfigLoader().load_dict(
        {
            "version": "1",
            "llm": {
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
            "pipeline": [
                {"type": "seed_source", "path": "seeds.jsonl"},
                {
                    "type": "prompt_based_generator",
                    "target_count": 1,
                    "generation_multiplier": 1,
                },
            ],
            "output": {"format": "jsonl", "path": "dataset.jsonl"},
        }
    )


def context(tmp_path, cfg, name="02_generate"):
    return StageContext(
        run_id="pii-test",
        stage_name=name,
        work_dir=tmp_path / "runs/pii-test/stages" / name,
        config=cfg,
        executor_mode="threadpool",
        max_workers=1,
        privacy=build_policy(cfg.pii, tmp_path),
    )


def seeds(tmp_path, ctx):
    (tmp_path / "seeds.jsonl").write_text(
        json.dumps(
            {
                "instruction": "Contact source@example.com",
                "response": "Explain gravity",
                "reasoning": "Ask reason@example.com",
            }
        )
        + "\n"
    )
    return SeedSourceStage(tmp_path).run([], ctx)


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def complete_structured(self, messages, schema, **kwargs):
        self.messages.append(messages)
        text = next(self.responses)
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


def test_source_and_generated_text_protected_before_raw_and_dataset(tmp_path):
    cfg = config()
    ctx = context(tmp_path, cfg)
    source = seeds(tmp_path, ctx)
    assert source[0].payload.instruction == "Contact [email]"
    assert source[0].payload.reasoning == "Ask [email]"
    assert source[0].content_hash == content_hash(source[0].payload)
    llm = FakeLLM(
        [
            '{"instruction":"Email generated@example.com", "response":"Done", "reasoning":"why@example.com"}'
        ]
    )
    generated = PromptBasedGeneratorStage(llm_client=llm).run(source, ctx)
    assert generated[0].payload.instruction == "Email [email]"
    assert generated[0].payload.reasoning == "[email]"
    assert "source@example.com" not in json.dumps(llm.messages)
    raw = (ctx.work_dir / "raw_responses.jsonl").read_text()
    assert "generated@example.com" not in raw and "why@example.com" not in raw
    assert json.loads(raw)["usage"]["total_tokens"] == 30


def test_malformed_and_unicode_escaped_json_do_not_leak_to_drop_artifacts(tmp_path):
    cfg = config()
    ctx = context(tmp_path, cfg)
    source = seeds(tmp_path, ctx)
    llm = FakeLLM(['{"instruction":"broken\\u0040example.com"'])
    assert PromptBasedGeneratorStage(llm_client=llm).run(source, ctx) == []
    assert "example.com" not in (ctx.work_dir / "raw_responses.jsonl").read_text()
    dropped = OutputWriter().read_parquet(ctx.work_dir / "dropped.parquet")
    assert "example.com" not in json.dumps([r.model_dump(mode="json") for r in dropped])


def test_generated_ratio_drop_keeps_usage_and_protected_diagnostics(tmp_path):
    cfg = config(max_redacted_ratio=0.5)
    ctx = context(tmp_path, cfg)
    # Keep the source sample so the rejection is measured on generated text.
    (tmp_path / "seeds.jsonl").write_text(
        '{"instruction":"Explain gravity", "response":"Attraction between masses"}\n'
    )
    source = SeedSourceStage(tmp_path).run([], ctx)
    llm = FakeLLM(['{"instruction":"a@example.com", "response":"b@example.com"}'])
    assert PromptBasedGeneratorStage(llm_client=llm).run(source, ctx) == []
    raw = json.loads((ctx.work_dir / "raw_responses.jsonl").read_text())
    assert raw["usage"]["cost_usd"] == 0.001
    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["drop_reasons"]["pii_ratio"] == 1


def test_evol_instruction_protected_before_response_call(tmp_path):
    cfg = config()
    ctx = context(tmp_path, cfg, "03_evol_round_01")
    source = seeds(tmp_path, ctx)
    llm = FakeLLM(
        [
            '{"instruction":"Explain gravity to evolved@example.com with detailed formulas."}',
            '{"response":"Contact answer@example.com"}',
        ]
    )
    result = EvolInstructRoundStage(
        round_number=1,
        llm_client=llm,
        config=EvolInstructGeneratorConfig(
            rounds=1, branching_factor=1, operators=["deepen"]
        ),
    ).run(source, ctx)
    assert len(result) == 2
    assert "evolved@example.com" not in json.dumps(llm.messages[1])
    assert "example.com" not in (ctx.work_dir / "raw_responses.jsonl").read_text()


def test_changed_privacy_policy_cannot_reuse_completed_source(tmp_path):
    cfg = config()
    (tmp_path / "seeds.jsonl").write_text(
        '{"instruction":"a@example.com", "response":"A"}\n'
    )
    runner = PipelineRunner(tmp_path)
    first = runner.run(cfg, [SeedSourceStage(tmp_path)], "resume-pii")
    saved = first.dataset_path.read_text()
    with pytest.raises(ValueError, match="PII policy.*new run"):
        runner.run(
            config(placeholder_template="<{entity}>"),
            [SeedSourceStage(tmp_path)],
            "resume-pii",
            resume=True,
        )
    assert first.dataset_path.read_text() == saved


def test_provider_exception_is_payload_free_in_manifest_and_cli_error(tmp_path):
    cfg = config()
    ctx = context(tmp_path, cfg)

    class Broken:
        def complete_structured(self, **kwargs):
            raise ValueError("provider echoed leak@example.com")

    with pytest.raises(RuntimeError) as caught:
        ctx.guard_llm(Broken()).complete_structured(messages=[], schema=object)
    assert "leak@example.com" not in str(caught.value)


def test_blocked_source_is_reported_and_only_protected_copy_is_saved(tmp_path):
    cfg = config(blocked_entities=["email"])
    ctx = context(tmp_path, cfg, "01_source")

    assert seeds(tmp_path, ctx) == []
    dropped = OutputWriter().read_parquet(ctx.work_dir / "pii_dropped.parquet")
    assert len(dropped) == 1
    assert dropped[0].payload.instruction == "Contact [email]"
    assert dropped[0].stage_events[-1].reason_code == "pii_blocked_entity"
    report = json.loads((ctx.work_dir / "pii_report.json").read_text())
    assert report["dropped_count"] == 1
    assert report["categories"] == {"email": 2}


def test_enabling_pii_rejects_old_unprotected_run_before_artifact_read(tmp_path):
    cfg = config()
    legacy = cfg.model_copy(update={"pii": None})
    (tmp_path / "seeds.jsonl").write_text(
        '{"instruction":"a@example.com", "response":"A"}\n'
    )
    runner = PipelineRunner(tmp_path)
    first = runner.run(legacy, [SeedSourceStage(tmp_path)], "old-run")
    with pytest.raises(ValueError, match="PII policy.*new run"):
        runner.run(cfg, [SeedSourceStage(tmp_path)], "old-run", resume=True)
    assert "a@example.com" in first.dataset_path.read_text()


def test_evol_blocked_instruction_stops_before_next_provider_call_and_keeps_usage(
    tmp_path,
):
    cfg = config(blocked_entities=["email"])
    ctx = context(tmp_path, cfg, "03_evol_round_01")
    (tmp_path / "seeds.jsonl").write_text(
        '{"instruction":"Explain gravity", "response":"Attraction between masses"}\n'
    )
    source = SeedSourceStage(tmp_path).run([], ctx)
    llm = FakeLLM(
        [
            '{"instruction":"Explain gravity to evolved@example.com with detailed formulas."}'
        ]
    )
    result = EvolInstructRoundStage(
        round_number=1,
        llm_client=llm,
        config=EvolInstructGeneratorConfig(
            rounds=1, branching_factor=1, operators=["deepen"]
        ),
    ).run(source, ctx)
    assert len(result) == 1 and len(llm.messages) == 1
    raw = json.loads((ctx.work_dir / "raw_responses.jsonl").read_text())
    assert raw["usage_instruction"]["total_tokens"] == 30


def test_model_protocol_roles_are_not_scanned_as_user_text():
    class Detector:
        def detect(self, text):
            return [Span(0, len(text), "person")] if text == "user" else []

    client = FakeLLM(['{"instruction":"Question", "response":"Answer"}'])
    from arka.pipeline.generator_stages import GeneratedConversation

    ProtectedLLMClient(client, PIIPolicy(Detector())).complete_structured(
        messages=[{"role": "user", "content": "Question"}], schema=GeneratedConversation
    )
    assert client.messages[0][0]["role"] == "user"


def test_native_provider_malformed_response_retains_usage_when_rejected(tmp_path):
    from types import SimpleNamespace

    from arka.llm.client import LLMClient
    from arka.pipeline.generator_stages import GeneratedConversation

    cfg = config()
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"instruction":"broken@example.com"'),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        model="fixture",
        id="request",
    )
    sdk = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response)
        )
    )
    client = LLMClient(cfg.llm, client_factory=lambda config: sdk)
    ctx = context(tmp_path, cfg)
    output = ctx.guard_llm(client).complete_structured(
        messages=[], schema=GeneratedConversation
    )
    assert output.usage.total_tokens == 30
    assert output.privacy.drop_reason == "pii_unparseable_response"
    assert "example.com" not in output.text


def test_turn_roles_preserved_while_system_reasoning_and_text_are_protected(tmp_path):
    from arka.privacy.records import protect_record

    cfg = config()
    ctx = context(tmp_path, cfg)
    record = seeds(tmp_path, ctx)[0]
    payload = record.payload.model_copy(
        update={
            "system": "system@example.com",
            "turns": [{"role": "user", "content": "turn@example.com"}],
        }
    )

    class Detector:
        def detect(self, text):
            return (
                [Span(0, len(text), "person")] if text == "user" or "@" in text else []
            )

    protected, _ = protect_record(
        record.model_copy(update={"payload": payload}), PIIPolicy(Detector())
    )
    assert protected.payload.system == "[person]"
    assert protected.payload.turns == [{"role": "user", "content": "[person]"}]


def test_source_pii_rejection_is_counted_in_manifest(tmp_path):
    cfg = config(blocked_entities=["email"])
    (tmp_path / "seeds.jsonl").write_text(
        '{"instruction":"a@example.com", "response":"A"}\n'
    )
    result = PipelineRunner(tmp_path).run(cfg, [SeedSourceStage(tmp_path)], "blocked")
    stage = json.loads(result.manifest_path.read_text())["stage_stats"][0]
    assert stage["dropped_count"] == 1
    assert stage["drop_reasons"] == {"pii_blocked_entity": 1}


def test_transform_privacy_drop_has_correct_counts_reason_and_cost(tmp_path):
    from arka.config.models import TransformGeneratorConfig
    from arka.pipeline.generator_stages import TransformGeneratorStage

    cfg = config(blocked_entities=["email"])
    ctx = context(tmp_path, cfg)
    (tmp_path / "seeds.jsonl").write_text(
        '{"instruction":"Explain gravity", "response":"Gravity attracts masses"}\n'
    )
    source = SeedSourceStage(tmp_path).run([], ctx)
    llm = FakeLLM(['{"text":"Contact generated@example.com"}'])
    stage = TransformGeneratorStage(
        llm_client=llm,
        config=TransformGeneratorConfig(
            input_field="payload.instruction",
            output_field="payload.response",
            prompt_template="{input_text}",
        ),
    )
    assert stage.run(source, ctx) == []
    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["count_in"] == 1 and stats["count_out"] == 0
    assert stats["drop_reasons"] == {"pii_blocked_entity": 1}
    assert stats["cost_usd"] == 0.001


def test_invalid_schema_cannot_persist_sensitive_extra_field_names(tmp_path):
    cfg = config()
    ctx = context(tmp_path, cfg)
    source = seeds(tmp_path, ctx)
    llm = FakeLLM(
        ['{"instruction":"Question", "response":"Answer", "key@example.com":"extra"}']
    )
    assert PromptBasedGeneratorStage(llm_client=llm).run(source, ctx) == []
    raw = (ctx.work_dir / "raw_responses.jsonl").read_text()
    assert "key@example.com" not in raw
    assert json.loads(raw)["privacy"]["drop_reason"] == "pii_unparseable_response"
