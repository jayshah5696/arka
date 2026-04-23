from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from pydantic import BaseModel

from arka.config.loader import ConfigLoader
from arka.llm.models import LLMOutput, TokenUsage
from arka.pipeline.generator_stages import TransformGeneratorStage
from arka.pipeline.models import StageContext
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


class RewriteResponse(BaseModel):
    text: str


class SequentialFakeLLMClient:
    def __init__(self, responses: list[str], *, cost_usd: float | None = None) -> None:
        self.responses = responses
        self.cost_usd = cost_usd
        self.calls = 0
        self.call_args: list[dict[str, object | None]] = []

    def complete(self, *args, **kwargs) -> LLMOutput:
        raise AssertionError("Transform stage should use complete_structured")

    def complete_structured(
        self,
        messages,
        schema: type[BaseModel],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMOutput:
        text = self.responses[self.calls]
        parsed = schema.model_validate(json.loads(text))
        self.calls += 1
        self.call_args.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return LLMOutput(
            text=text,
            parsed=parsed,
            usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=12,
                total_tokens=22,
                cost_usd=self.cost_usd,
            ),
            finish_reason="stop",
            model="gpt-4o-mini",
            provider="openai",
            request_id=f"req_{self.calls}",
            latency_ms=15,
            error=None,
        )


def _ctx(tmp_path: Path, **generator_overrides) -> StageContext:
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
                "type": "transform",
                "input_field": "payload.instruction",
                "output_field": "payload.response",
                "prompt_template": "Rewrite this text:\n{input_text}",
                "temperature": 0.3,
                "max_tokens": 256,
                "preserve_original": True,
                **generator_overrides,
            },
            
            "filters": {"target_count": 1},
            "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
        }
    )
    work_dir = tmp_path / "runs" / "run-1" / "stages" / "02_transform_generate"
    work_dir.mkdir(parents=True, exist_ok=True)
    return StageContext(
        run_id="run-1",
        stage_name="02_transform_generate",
        work_dir=work_dir,
        config=config,
        executor_mode=config.executor.mode,
        max_workers=config.executor.max_workers,
    )


def _record(record_id: str, instruction: str, response: str) -> ConversationRecord:
    return ConversationRecord(
        id=record_id,
        content_hash=f"hash-{record_id}",
        source=RecordSource(type="seed", seed_file_hash="seed-file-hash"),
        lineage=RecordLineage(root_id=f"root-{record_id}", parent_ids=[]),
        payload=ConversationPayload(instruction=instruction, response=response),
        scores=RecordScores(),
        config_hash="cfg-1",
        created_at="2026-04-14T00:00:00Z",
    )


def test_transform_generator_rewrites_configured_field_and_preserves_original(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    stage = TransformGeneratorStage(
        llm_client=SequentialFakeLLMClient([json.dumps({"text": "Humanized hello"})])
    )

    records = stage.run([_record("seed-1", "Original hello", "Old response")], ctx)

    assert len(records) == 1
    transformed = records[0]
    assert transformed.payload.instruction == "Original hello"
    assert transformed.payload.response == "Humanized hello"
    assert transformed.payload.system == json.dumps(
        {
            "transform_original": {
                "field": "payload.response",
                "text": "Old response",
            }
        },
        separators=(",", ":"),
    )
    assert transformed.lineage.parent_ids == ["seed-1"]
    assert transformed.lineage.operator == "transform"
    assert transformed.lineage.depth == 1


def test_transform_generator_uses_input_field_text_in_prompt(tmp_path: Path) -> None:
    ctx = _ctx(
        tmp_path,
        input_field="payload.instruction",
        output_field="payload.response",
        prompt_template="Rewrite carefully:\n{input_text}",
    )
    fake_client = SequentialFakeLLMClient([json.dumps({"text": "Rewritten"})])
    stage = TransformGeneratorStage(llm_client=fake_client)

    stage.run([_record("seed-1", "Prompt me", "Old response")], ctx)

    assert fake_client.call_args[0]["messages"] == [
        {"role": "user", "content": "Rewrite carefully:\nPrompt me"}
    ]
    assert fake_client.call_args[0]["temperature"] == 0.3
    assert fake_client.call_args[0]["max_tokens"] == 256


def test_transform_generator_writes_stats_with_cost(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    stage = TransformGeneratorStage(
        llm_client=SequentialFakeLLMClient(
            [json.dumps({"text": "Humanized hello"})],
            cost_usd=0.0042,
        )
    )

    stage.run([_record("seed-1", "Original hello", "Old response")], ctx)

    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats == {
        "stage": "02_transform_generate",
        "count_in": 1,
        "count_out": 1,
        "dropped_count": 0,
        "drop_reasons": {},
        "cost_usd": 0.0042,
    }

    dropped = pl.read_parquet(ctx.work_dir / "dropped.parquet")
    assert dropped.height == 0


def test_transform_generator_uses_llm_override_model(tmp_path: Path) -> None:
    """When llm_override.model is set, the stage should build a client with that model."""
    ctx = _ctx(
        tmp_path,
        llm_override={"model": "qwen/qwen3.5-9b"},
    )
    # The fake client is injected, so we just verify config was accepted
    # and the stage runs. Real integration would check the built client model.
    stage = TransformGeneratorStage(
        llm_client=SequentialFakeLLMClient([json.dumps({"text": "Overridden"})])
    )

    records = stage.run([_record("seed-1", "Hello", "World")], ctx)

    assert len(records) == 1
    assert records[0].payload.response == "Overridden"
    assert ctx.config.generator.llm_override is not None
    assert ctx.config.generator.llm_override.model == "qwen/qwen3.5-9b"


def test_transform_generator_rejects_invalid_field_path(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, input_field="payload.missing")
    stage = TransformGeneratorStage(
        llm_client=SequentialFakeLLMClient([json.dumps({"text": "noop"})])
    )

    try:
        stage.run([_record("seed-1", "Original hello", "Old response")], ctx)
    except ValueError as exc:
        assert str(exc) == "Unsupported transform field path: payload.missing"
    else:
        raise AssertionError("Expected ValueError for unsupported field path")
