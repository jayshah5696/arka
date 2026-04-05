from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from arka.config.loader import ConfigLoader
from arka.llm.models import LLMOutput, TokenUsage
from arka.pipeline.generation_stages import PromptBasedGeneratorStage
from arka.pipeline.models import StageContext
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


class SequentialFakeLLMClient:
    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        self.pairs = pairs
        self.calls = 0

    def complete_structured(self, messages, schema: type[BaseModel]) -> LLMOutput:
        instruction, response = self.pairs[self.calls]
        self.calls += 1
        parsed = schema.model_validate(
            {"instruction": instruction, "response": response}
        )
        return LLMOutput(
            text=parsed.model_dump_json(),
            parsed=parsed,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            finish_reason="stop",
            model="gpt-4o-mini",
            provider="openai",
            request_id=f"req_{self.calls}",
            latency_ms=25,
            error=None,
        )


def _config(tmp_path: Path, **generator_overrides) -> StageContext:
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
                **generator_overrides,
            },
            "dedup": {"exact": {"enabled": False}},
            "filters": {"target_count": 2},
            "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
        }
    )
    work_dir = tmp_path / "02_generate"
    work_dir.mkdir(parents=True, exist_ok=True)
    return StageContext(
        run_id="run-1",
        stage_name="02_generate",
        work_dir=work_dir,
        config=config,
        executor_mode=config.executor.mode,
        max_workers=config.executor.max_workers,
    )


def _seed_record(record_id: str, instruction: str, response: str) -> ConversationRecord:
    return ConversationRecord(
        id=record_id,
        content_hash=f"hash-{record_id}",
        source=RecordSource(type="seed", seed_file_hash="seed-file-hash"),
        lineage=RecordLineage(root_id=f"root-{record_id}", parent_ids=[]),
        payload=ConversationPayload(instruction=instruction, response=response),
        scores=RecordScores(),
        config_hash="cfg-1",
        created_at="2026-04-04T00:00:00Z",
    )


def test_prompt_based_generator_emits_target_count_times_multiplier(
    tmp_path: Path,
) -> None:
    ctx = _config(tmp_path, target_count=3, generation_multiplier=2)
    stage = PromptBasedGeneratorStage(
        llm_client=SequentialFakeLLMClient(
            [
                ("gen-1", "resp-1"),
                ("gen-2", "resp-2"),
                ("gen-3", "resp-3"),
                ("gen-4", "resp-4"),
                ("gen-5", "resp-5"),
                ("gen-6", "resp-6"),
            ]
        )
    )

    records = stage.run(
        [
            _seed_record("seed-1", "Seed instruction 1", "Seed response 1"),
            _seed_record("seed-2", "Seed instruction 2", "Seed response 2"),
        ],
        ctx,
    )

    assert len(records) == 6
    assert [record.lineage.parent_ids for record in records] == [
        ["seed-1"],
        ["seed-2"],
        ["seed-1"],
        ["seed-2"],
        ["seed-1"],
        ["seed-2"],
    ]


def test_prompt_based_generator_sets_generated_source_and_lineage(
    tmp_path: Path,
) -> None:
    ctx = _config(tmp_path, target_count=1, generation_multiplier=1)
    stage = PromptBasedGeneratorStage(
        llm_client=SequentialFakeLLMClient([("New instruction", "New response")])
    )
    seed = _seed_record("seed-1", "Seed instruction", "Seed response")

    records = stage.run([seed], ctx)

    assert len(records) == 1
    generated = records[0]
    assert generated.source.type == "generated"
    assert generated.source.seed_file_hash == "seed-file-hash"
    assert generated.lineage.root_id == "root-seed-1"
    assert generated.lineage.parent_ids == ["seed-1"]
    assert generated.lineage.operator == "prompt_based"
    assert generated.lineage.round == 1
    assert generated.lineage.depth == 1
    assert generated.payload.instruction == "New instruction"
    assert generated.payload.response == "New response"


def test_prompt_based_generator_ids_are_content_stable_per_parent(
    tmp_path: Path,
) -> None:
    ctx = _config(tmp_path, target_count=1, generation_multiplier=2)
    stage = PromptBasedGeneratorStage(
        llm_client=SequentialFakeLLMClient(
            [
                ("Same instruction", "Same response"),
                ("Same instruction", "Same response"),
            ]
        )
    )
    seed = _seed_record("seed-1", "Seed instruction", "Seed response")

    records = stage.run([seed], ctx)

    assert len(records) == 2
    assert records[0].id == records[1].id
    assert records[0].content_hash == records[1].content_hash


def test_prompt_based_generator_returns_empty_for_no_seed_records(
    tmp_path: Path,
) -> None:
    ctx = _config(tmp_path)
    stage = PromptBasedGeneratorStage(llm_client=SequentialFakeLLMClient([]))

    records = stage.run([], ctx)

    assert records == []
