from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel

import arka.pipeline.generation_stages as generation_stages
from arka.config.loader import ConfigLoader
from arka.llm.models import LLMOutput, TokenUsage
from arka.pipeline.checkpoint import CheckpointManager
from arka.pipeline.evol_generator_stage import EvolInstructRoundStage
from arka.pipeline.generator_stages import (
    PromptBasedGeneratorStage,
    compute_prompt_hash,
)
from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    GroundedChunkPayload,
    GroundedChunkRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


class SequentialFakeLLMClient:
    def __init__(
        self,
        responses: list[str],
        *,
        return_parsed_only: bool = False,
        cost_usd: float | None = None,
    ) -> None:
        self.responses = responses
        self.return_parsed_only = return_parsed_only
        self.cost_usd = cost_usd
        self.calls = 0
        self.call_args: list[dict[str, object | None]] = []

    def complete(self, *args, **kwargs) -> LLMOutput:
        raise AssertionError("Generator should use complete_structured, not complete")

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
            text=None if self.return_parsed_only else text,
            parsed=parsed,
            usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                cost_usd=self.cost_usd,
            ),
            finish_reason="stop",
            model="gpt-4o-mini",
            provider="openai",
            request_id=f"req_{self.calls}",
            latency_ms=25,
            error=None,
        )


class FailingFakeLLMClient:
    def complete(self, *args, **kwargs) -> LLMOutput:
        raise AssertionError("LLM should not have been called")

    def complete_structured(self, *args, **kwargs) -> LLMOutput:
        raise AssertionError("LLM should not have been called")


def _generated_json(instruction: str, response: str) -> str:
    return json.dumps({"instruction": instruction, "response": response})


def _instruction_json(instruction: str) -> str:
    return json.dumps({"instruction": instruction})


def _response_json(response: str) -> str:
    return json.dumps({"response": response})


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
    work_dir = tmp_path / "runs" / "run-1" / "stages" / "02_generate"
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


def _chunk_record(record_id: str, text: str) -> GroundedChunkRecord:
    return GroundedChunkRecord(
        id=record_id,
        content_hash=f"hash-{record_id}",
        source=RecordSource(
            type="pdf_chunk",
            doc_id="doc-1",
            chunk_id="doc-1:0",
            page_start=1,
            page_end=1,
            char_start=0,
            char_end=len(text),
            source_hash="source-hash",
        ),
        lineage=RecordLineage(root_id=record_id, parent_ids=[]),
        payload=GroundedChunkPayload(
            text=text,
            doc_id="doc-1",
            chunk_idx=0,
            page_start=1,
            page_end=1,
            char_start=0,
            char_end=len(text),
            word_count=len(text.split()),
            chunk_strategy="fixed",
        ),
        scores=RecordScores(),
        config_hash="cfg-1",
        created_at="2026-04-05T00:00:00Z",
    )


def _raw_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_prompt_based_generator_emits_target_count_times_multiplier_and_writes_raw_responses(
    tmp_path: Path,
) -> None:
    ctx = _config(tmp_path, target_count=3, generation_multiplier=2)
    llm_client = SequentialFakeLLMClient(
        [
            _generated_json("gen-1", "resp-1"),
            _generated_json("gen-2", "resp-2"),
            _generated_json("gen-3", "resp-3"),
            _generated_json("gen-4", "resp-4"),
            _generated_json("gen-5", "resp-5"),
            _generated_json("gen-6", "resp-6"),
        ]
    )
    stage = PromptBasedGeneratorStage(llm_client=llm_client)

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
    assert llm_client.calls == 6
    assert llm_client.call_args[0]["temperature"] == 0.7
    assert llm_client.call_args[0]["max_tokens"] == 512

    raw_rows = _raw_rows(ctx.work_dir / "raw_responses.jsonl")
    assert [row["plan_index"] for row in raw_rows] == [0, 1, 2, 3, 4, 5]
    assert [row["seed_id"] for row in raw_rows] == [
        "seed-1",
        "seed-2",
        "seed-1",
        "seed-2",
        "seed-1",
        "seed-2",
    ]

    cached = CheckpointManager(tmp_path / "state.db").load_generator(
        ctx.run_id, stage.name
    )
    assert cached is not None
    assert cached["response_count"] == 6
    assert cached["status"] == "completed"


def test_prompt_based_generator_accepts_grounded_chunk_records(tmp_path: Path) -> None:
    ctx = _config(tmp_path, target_count=1, generation_multiplier=1)
    stage = PromptBasedGeneratorStage(
        llm_client=SequentialFakeLLMClient(
            [_generated_json("Grounded question", "Grounded answer")]
        )
    )

    records = stage.run([_chunk_record("chunk-1", "Facts about gravity.")], ctx)

    assert len(records) == 1
    assert records[0].source.doc_id == "doc-1"
    assert records[0].source.chunk_id == "doc-1:0"
    assert records[0].lineage.parent_ids == ["chunk-1"]


def test_prompt_based_generator_writes_cost_usd_to_stats_when_available(
    tmp_path: Path,
) -> None:
    ctx = _config(tmp_path)
    stage = PromptBasedGeneratorStage(
        llm_client=SequentialFakeLLMClient(
            [
                _generated_json("gen-1", "resp-1"),
                _generated_json("gen-2", "resp-2"),
            ],
            cost_usd=0.001,
        )
    )

    records = stage.run(
        [_seed_record("seed-1", "Seed instruction", "Seed response")],
        ctx,
    )

    assert len(records) == 2
    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["cost_usd"] == 0.002


def test_prompt_based_generator_resumes_from_parquet_when_prompt_is_unchanged(
    tmp_path: Path,
) -> None:
    ctx = _config(tmp_path)
    seeds = [_seed_record("seed-1", "Seed instruction", "Seed response")]
    first_stage = PromptBasedGeneratorStage(
        llm_client=SequentialFakeLLMClient(
            [
                _generated_json("Generated instruction 1", "Generated response 1"),
                _generated_json("Generated instruction 2", "Generated response 2"),
            ]
        )
    )

    first_records = first_stage.run(seeds, ctx)
    OutputWriter().write_parquet(first_records, ctx.work_dir / "data.parquet")

    resumed_records = PromptBasedGeneratorStage(llm_client=FailingFakeLLMClient()).run(
        seeds,
        ctx,
    )

    assert [record.id for record in resumed_records] == [
        record.id for record in first_records
    ]
    assert [record.payload.instruction for record in resumed_records] == [
        "Generated instruction 1",
        "Generated instruction 2",
    ]


def test_prompt_based_generator_regenerates_when_prompt_changes(
    tmp_path: Path,
) -> None:
    seeds = [_seed_record("seed-1", "Seed instruction", "Seed response")]
    ctx_v1 = _config(
        tmp_path, prompt_template="Version A: {seed_instruction}\n{seed_response}"
    )
    stage_v1 = PromptBasedGeneratorStage(
        llm_client=SequentialFakeLLMClient(
            [
                _generated_json("A1", "A1 response"),
                _generated_json("A2", "A2 response"),
            ]
        )
    )
    first_records = stage_v1.run(seeds, ctx_v1)
    OutputWriter().write_parquet(first_records, ctx_v1.work_dir / "data.parquet")

    ctx_v2 = _config(
        tmp_path, prompt_template="Version B: {seed_instruction}\n{seed_response}"
    )
    llm_v2 = SequentialFakeLLMClient(
        [
            _generated_json("B1", "B1 response"),
            _generated_json("B2", "B2 response"),
        ]
    )
    regenerated_records = PromptBasedGeneratorStage(llm_client=llm_v2).run(
        seeds, ctx_v2
    )

    assert llm_v2.calls == 2
    assert [record.payload.instruction for record in regenerated_records] == [
        "B1",
        "B2",
    ]

    raw_rows = _raw_rows(ctx_v2.work_dir / "raw_responses.jsonl")
    expected_hash = compute_prompt_hash(ctx_v2.config.generator, ctx_v2.config.llm)
    assert len(raw_rows) == 2
    assert {row["prompt_hash"] for row in raw_rows} == {expected_hash}
    assert [row["generated_text"] for row in raw_rows] == [
        _generated_json("B1", "B1 response"),
        _generated_json("B2", "B2 response"),
    ]


def test_prompt_based_generator_resumes_partial_raw_response_file(
    tmp_path: Path,
) -> None:
    ctx = _config(tmp_path, target_count=3, generation_multiplier=2)
    stage = PromptBasedGeneratorStage(
        llm_client=SequentialFakeLLMClient(
            [
                _generated_json("gen-4", "resp-4"),
                _generated_json("gen-5", "resp-5"),
                _generated_json("gen-6", "resp-6"),
            ]
        )
    )
    seeds = [
        _seed_record("seed-1", "Seed instruction 1", "Seed response 1"),
        _seed_record("seed-2", "Seed instruction 2", "Seed response 2"),
    ]
    prompt_hash = compute_prompt_hash(ctx.config.generator, ctx.config.llm)
    responses_path = ctx.work_dir / "raw_responses.jsonl"
    partial_rows = [
        {
            "plan_index": 0,
            "seed_id": "seed-1",
            "generated_text": _generated_json("gen-1", "resp-1"),
            "prompt_hash": prompt_hash,
            "model": "gpt-4o-mini",
            "latency_ms": 25,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        },
        {
            "plan_index": 1,
            "seed_id": "seed-2",
            "generated_text": _generated_json("gen-2", "resp-2"),
            "prompt_hash": prompt_hash,
            "model": "gpt-4o-mini",
            "latency_ms": 25,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        },
        {
            "plan_index": 2,
            "seed_id": "seed-1",
            "generated_text": _generated_json("gen-3", "resp-3"),
            "prompt_hash": prompt_hash,
            "model": "gpt-4o-mini",
            "latency_ms": 25,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        },
    ]
    responses_path.write_text("\n".join(json.dumps(row) for row in partial_rows) + "\n")
    CheckpointManager(tmp_path / "state.db").save_generator(
        run_id=ctx.run_id,
        stage_name=stage.name,
        prompt_hash=prompt_hash,
        responses_path=responses_path,
        response_count=3,
        status="running",
    )

    records = stage.run(seeds, ctx)

    assert len(records) == 6
    assert stage._llm_client.calls == 3
    assert [record.lineage.parent_ids for record in records] == [
        ["seed-1"],
        ["seed-2"],
        ["seed-1"],
        ["seed-2"],
        ["seed-1"],
        ["seed-2"],
    ]

    raw_rows = _raw_rows(responses_path)
    assert [row["plan_index"] for row in raw_rows] == [0, 1, 2, 3, 4, 5]
    assert [row["seed_id"] for row in raw_rows] == [
        "seed-1",
        "seed-2",
        "seed-1",
        "seed-2",
        "seed-1",
        "seed-2",
    ]


def test_prompt_based_generator_skips_malformed_rows_and_records_drop_stats(
    tmp_path: Path,
    caplog,
) -> None:
    ctx = _config(tmp_path)
    prompt_hash = compute_prompt_hash(ctx.config.generator, ctx.config.llm)
    stage = PromptBasedGeneratorStage(
        checkpoint_manager=CheckpointManager(tmp_path / "state.db")
    )
    seeds = [_seed_record("seed-1", "Seed instruction", "Seed response")]
    raw_rows = [
        {
            "plan_index": 0,
            "seed_id": "seed-1",
            "generated_text": _generated_json("good-1", "resp-1"),
            "prompt_hash": prompt_hash,
            "model": "gpt-4o-mini",
            "latency_ms": 25,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        },
        {
            "plan_index": 1,
            "seed_id": "seed-1",
            "generated_text": '{"instruction":"broken"}',
            "prompt_hash": prompt_hash,
            "model": "gpt-4o-mini",
            "latency_ms": 25,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        },
    ]

    responses_path = ctx.work_dir / "raw_responses.jsonl"
    responses_path.write_text("\n".join(json.dumps(row) for row in raw_rows) + "\n")

    with caplog.at_level(logging.WARNING, logger="arka.pipeline.generator_stages"):
        records = stage._parse_responses(
            stage._load_raw_responses(responses_path),
            stage._generation_plan(seeds, ctx.config.generator),
            ctx,
        )

    assert len(records) == 1
    assert records[0].payload.instruction == "good-1"
    assert "Skipping malformed generator response for plan_index=1" in caplog.text

    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["count_in"] == 2
    assert stats["count_out"] == 1
    assert stats["dropped_count"] == 1
    assert stats["drop_reasons"] == {"generator_parse_failure": 1}

    dropped = OutputWriter().read_parquet(ctx.work_dir / "dropped.parquet")
    assert len(dropped) == 1
    assert dropped[0].stage_events[-1].reason_code == "generator_parse_failure"


def test_prompt_based_generator_uses_injected_checkpoint_manager_without_ctx_path_inference(
    tmp_path: Path,
) -> None:
    checkpoint_manager = CheckpointManager(tmp_path / "state.db")
    stage = PromptBasedGeneratorStage(checkpoint_manager=checkpoint_manager)
    ctx = _config(tmp_path)

    assert stage._checkpoint_manager(ctx) is checkpoint_manager


def test_generation_stages_shim_exports_only_public_symbols() -> None:
    assert hasattr(generation_stages, "PromptBasedGeneratorStage")
    assert hasattr(generation_stages, "compute_prompt_hash")
    assert not hasattr(generation_stages, "GenerationPlanItem")
    assert not hasattr(generation_stages, "RawGeneratorResponse")
    assert not hasattr(generation_stages, "_DEFAULT_PROMPT_TEMPLATE")


def test_prompt_based_generator_returns_empty_for_no_seed_records(
    tmp_path: Path,
) -> None:
    ctx = _config(tmp_path)
    stage = PromptBasedGeneratorStage(llm_client=SequentialFakeLLMClient([]))

    records = stage.run([], ctx)

    assert records == []


def test_evol_round_stage_generates_evolved_records_with_lineage(
    tmp_path: Path,
) -> None:
    ctx = _config(
        tmp_path,
        type="evol_instruct",
        rounds=1,
        branching_factor=1,
        operators=["deepen"],
    )
    ctx = StageContext(
        run_id=ctx.run_id,
        stage_name="03_evol_round_01",
        work_dir=tmp_path / "runs" / "run-1" / "stages" / "03_evol_round_01",
        config=ctx.config,
        executor_mode=ctx.executor_mode,
        max_workers=ctx.max_workers,
    )
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    stage = EvolInstructRoundStage(
        round_number=1,
        llm_client=SequentialFakeLLMClient(
            [
                _instruction_json(
                    "Explain gravity with Newtonian intuition and caveats."
                ),
                _response_json("Gravity describes attraction between masses."),
            ]
        ),
    )

    result = stage.run(
        [_seed_record("seed-1", "Explain gravity", "Gravity attracts masses.")],
        ctx,
    )

    assert len(result) == 2
    evolved = result[-1]
    assert evolved.source.type == "evolved"
    assert evolved.lineage.parent_ids == ["seed-1"]
    assert evolved.lineage.root_id == "root-seed-1"
    assert evolved.lineage.operator == "deepen"
    assert evolved.lineage.round == 1
    assert evolved.lineage.depth == 1


def test_evol_round_stage_drops_identical_or_refusal_candidates(tmp_path: Path) -> None:
    ctx = _config(
        tmp_path,
        type="evol_instruct",
        rounds=1,
        branching_factor=2,
        operators=["deepen", "add_constraints"],
    )
    ctx = StageContext(
        run_id=ctx.run_id,
        stage_name="03_evol_round_01",
        work_dir=tmp_path / "runs" / "run-1" / "stages" / "03_evol_round_01",
        config=ctx.config,
        executor_mode=ctx.executor_mode,
        max_workers=ctx.max_workers,
    )
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    stage = EvolInstructRoundStage(
        round_number=1,
        llm_client=SequentialFakeLLMClient(
            [
                _instruction_json("Explain gravity"),
                _instruction_json("As an AI, I cannot help with that."),
            ]
        ),
    )

    result = stage.run(
        [_seed_record("seed-1", "Explain gravity", "Gravity attracts masses.")],
        ctx,
    )

    assert len(result) == 1
    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["generated_count"] == 0
    assert stats["dropped_count"] == 2
    assert stats["drop_reasons"] == {
        "evol_identical_to_parent": 1,
        "evol_refusal": 1,
    }


def test_evol_round_stage_round_two_uses_frontier_only(tmp_path: Path) -> None:
    ctx = _config(
        tmp_path,
        type="evol_instruct",
        rounds=2,
        branching_factor=1,
        operators=["deepen"],
    )
    stage_one_ctx = StageContext(
        run_id=ctx.run_id,
        stage_name="03_evol_round_01",
        work_dir=tmp_path / "runs" / "run-1" / "stages" / "03_evol_round_01",
        config=ctx.config,
        executor_mode=ctx.executor_mode,
        max_workers=ctx.max_workers,
    )
    stage_two_ctx = StageContext(
        run_id=ctx.run_id,
        stage_name="04_evol_round_02",
        work_dir=tmp_path / "runs" / "run-1" / "stages" / "04_evol_round_02",
        config=ctx.config,
        executor_mode=ctx.executor_mode,
        max_workers=ctx.max_workers,
    )
    stage_one_ctx.work_dir.mkdir(parents=True, exist_ok=True)
    stage_two_ctx.work_dir.mkdir(parents=True, exist_ok=True)

    round_one = EvolInstructRoundStage(
        round_number=1,
        llm_client=SequentialFakeLLMClient(
            [
                _instruction_json("Explain gravity with formulas and intuition."),
                _response_json("Gravity can be modeled with Newton's law."),
            ]
        ),
    )
    after_round_one = round_one.run(
        [_seed_record("seed-1", "Explain gravity", "Gravity attracts masses.")],
        stage_one_ctx,
    )

    round_two_client = SequentialFakeLLMClient(
        [
            _instruction_json("Compare Newtonian and relativistic gravity."),
            _response_json(
                "Newtonian gravity works well classically; relativity handles spacetime."
            ),
        ]
    )
    after_round_two = EvolInstructRoundStage(
        round_number=2,
        llm_client=round_two_client,
    ).run(after_round_one, stage_two_ctx)

    assert len(after_round_two) == 3
    assert round_two_client.calls == 2
    assert after_round_two[-1].lineage.parent_ids == [after_round_one[-1].id]
    assert after_round_two[-1].lineage.round == 2
