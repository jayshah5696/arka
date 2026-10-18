"""Unit tests for the 7 architecture deepening candidates."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from arka.config.loader import ConfigLoader
from arka.config.models import (
    DecisionStageOptions,
    LabelingFilterConfig,
    LabelingScoreConfig,
    LengthFilterConfig,
    ResolvedConfig,
    StageLLMOverride,
    TaxonomyGeneratorConfig,
)
from arka.labeling.judges import JudgeResponse
from arka.llm.models import LLMOutput, TokenUsage
from arka.pipeline.checkpoint import CheckpointManager
from arka.pipeline.decision_stages import (
    apply_uncertain,
    get_decision_spec,
)
from arka.pipeline.filter_stages import LabelingQualityFilterStage
from arka.pipeline.generator_engine import (
    RawGeneratorResponse,
    apply_post_generation_decisions,
    execute_generator_slots,
    existing_rows_for_prompt,
    extract_json_text,
    load_raw_responses,
    resolve_generator_llm_client,
)
from arka.pipeline.labeling_evaluator import evaluate_labeling
from arka.pipeline.models import (
    StageContext,
    StageResult,
)
from arka.pipeline.registry import get_stage_spec
from arka.pipeline.runner import PipelineRunner
from arka.pipeline.scoring_stages import LabelingScoreStage
from arka.pipeline.stage_builder import StageBuilder
from arka.pipeline.stages import Stage
from arka.pipeline.taxonomy_generator import TaxonomyGeneratorStage
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    Record,
    RecordLineage,
    RecordScores,
    RecordSource,
)
from arka.taxonomy.coverage import extract_sampled_from_record


def _create_record(
    record_id: str = "rec-1",
    instruction: str = "Hello",
    response: str = "World",
) -> ConversationRecord:
    payload = ConversationPayload(instruction=instruction, response=response)
    lineage = RecordLineage(root_id=record_id, parent_ids=[])
    return ConversationRecord(
        id=record_id,
        content_hash=f"hash-{record_id}",
        source=RecordSource(type="generated"),
        lineage=lineage,
        payload=payload,
        scores=RecordScores(),
        config_hash="cfg-1",
        created_at="2026-04-14T00:00:00Z",
    )


def _create_plain_record(record_id: str = "plain-1") -> Record:
    return Record(
        id=record_id,
        content_hash=f"hash-{record_id}",
        source=RecordSource(type="seed"),
        lineage=RecordLineage(root_id=record_id, parent_ids=[]),
        payload={"data": "raw"},
        scores=RecordScores(),
        config_hash="cfg-1",
        created_at="2026-04-14T00:00:00Z",
    )


def _minimal_config(tmp_path: Path) -> ResolvedConfig:
    return ConfigLoader().load_dict(
        {
            "version": "1",
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "test-key",
                "base_url": "https://api.openai.com/v1",
            },
            "executor": {"mode": "threadpool", "max_workers": 2},
            "data_source": {"type": "seeds", "path": "./seeds.jsonl"},
            "generator": {
                "type": "prompt_based",
                "target_count": 2,
                "generation_multiplier": 1,
            },
            "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
        }
    )


# ---------------------------------------------------------------------------
# Candidate 1: StageResult
# ---------------------------------------------------------------------------


def test_stage_result_protocol_and_partitioning() -> None:
    conv = _create_record("conv-1")
    plain = _create_plain_record("plain-1")

    # Partitioning
    convs, others = StageResult.partition_conversations([conv, plain])
    assert convs == [conv]
    assert others == [plain]

    # Construction and interface
    result = StageResult(
        kept=[conv],
        dropped=[plain],
        stats={"count_in": 2, "count_out": 1},
    )
    assert len(result) == 1
    assert result[0] == conv
    assert list(result) == [conv]
    assert result == [conv]  # list comparison
    assert result == StageResult(
        kept=[conv],
        dropped=[plain],
        stats={"count_in": 2, "count_out": 1},
    )


def test_stage_result_in_pipeline_runner(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    runner = PipelineRunner(project_root=tmp_path)

    class ResultProducingStage(Stage):
        name = "01_result_stage"
        stage_action = "transformed"

        def run(self, records: list[Record], ctx: StageContext) -> StageResult:
            rec = _create_record("r-1")
            return StageResult(
                kept=[rec],
                dropped=[],
                stats={
                    "stage": self.name,
                    "count_in": len(records),
                    "count_out": 1,
                    "dropped_count": 0,
                    "drop_reasons": {},
                },
            )

    run_paths = runner.run(
        run_id="run-1", config=config, stages=[ResultProducingStage()]
    )
    assert run_paths.final_count == 1
    assert run_paths.manifest_path.exists()
    manifest = json.loads(run_paths.manifest_path.read_text())
    assert manifest["stage_stats"][0]["stage"] == "01_result_stage"
    assert manifest["stage_stats"][0]["count_out"] == 1


# ---------------------------------------------------------------------------
# Candidate 2: DecisionStageSpec & Unified evaluate_labeling
# ---------------------------------------------------------------------------


def test_decision_stage_spec_registry() -> None:
    seed_gate = get_decision_spec("seed_gate")
    assert seed_gate is not None
    assert seed_gate.kind == "seed_gate"
    assert seed_gate.requires_conversation is False

    reward = get_decision_spec("reward_model")
    assert reward is not None
    assert reward.kind == "reward_model"


def test_unified_evaluate_labeling(tmp_path: Path) -> None:
    rubric_path = tmp_path / "rubric.yaml"
    rubric_path.write_text(
        """
version: "1.0"
description: "Test Rubric"
dimensions:
  - name: clarity
    description: "Clarity"
    scale_min: 1
    scale_max: 5
    criteria: {1: "Poor", 5: "Great"}
overall_weights:
  clarity: 1.0
"""
    )

    class FakeClient:
        def complete_structured(self, messages, schema):
            return LLMOutput(
                text="{}",
                parsed=JudgeResponse(scores={"clarity": 2}, reasoning="low"),
                usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
                model="gpt-4o-mini",
                provider="openai",
                latency_ms=10,
            )

    config = LabelingFilterConfig(
        rubric_path="rubric.yaml",
        min_overall_score=3.0,
    )
    work_dir = tmp_path / "runs" / "r1" / "stages" / "label_test"
    work_dir.mkdir(parents=True, exist_ok=True)
    ctx = StageContext(
        run_id="r1",
        stage_name="label_test",
        work_dir=work_dir,
        config=_minimal_config(tmp_path),
        executor_mode="threadpool",
        max_workers=1,
    )

    stage_filter = LabelingQualityFilterStage(project_root=tmp_path)
    stage_score = LabelingScoreStage(project_root=tmp_path)

    records = [_create_record("rec-1")]

    # When drop_below_threshold=True, score 2 < min 3 -> dropped
    filtered = evaluate_labeling(
        stage=stage_filter,
        records=records,
        ctx=ctx,
        config=config,
        project_root=tmp_path,
        llm_client=FakeClient(),
        drop_below_threshold=True,
    )
    assert len(filtered) == 0

    # When drop_below_threshold=False, score 2 is kept and annotated
    scored = evaluate_labeling(
        stage=stage_score,
        records=records,
        ctx=ctx,
        config=config,
        project_root=tmp_path,
        llm_client=FakeClient(),
        drop_below_threshold=False,
    )
    assert len(scored) == 1
    assert scored[0].scores.quality == 2.0


# ---------------------------------------------------------------------------
# Candidate 3: generator_engine helpers & TaxonomyGeneratorStage
# ---------------------------------------------------------------------------


def test_generator_engine_helpers(tmp_path: Path) -> None:
    # 1. extract_json_text
    assert extract_json_text('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json_text('Some text {"b": 2} more text') == '{"b": 2}'
    assert extract_json_text('{"c": 3}') == '{"c": 3}'

    # 2. load_raw_responses & existing_rows_for_prompt
    missing = tmp_path / "nonexistent.jsonl"
    assert load_raw_responses(missing) == []
    assert existing_rows_for_prompt(missing, "hash123") == []

    p = tmp_path / "raw.jsonl"
    p.write_text(
        json.dumps(
            {
                "plan_index": 0,
                "seed_id": "s1",
                "generated_text": "resp",
                "prompt_hash": "hash123",
                "model": "m1",
                "latency_ms": 10,
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )
        + "\n"
    )
    rows = load_raw_responses(p)
    assert len(rows) == 1
    assert rows[0].seed_id == "s1"
    assert len(existing_rows_for_prompt(p, "hash123")) == 1
    assert len(existing_rows_for_prompt(p, "other_hash")) == 0

    # 3. resolve_generator_llm_client with override
    ctx = StageContext(
        run_id="r1",
        stage_name="gen",
        work_dir=tmp_path,
        config=_minimal_config(tmp_path),
        executor_mode="threadpool",
        max_workers=1,
    )
    cfg = TaxonomyGeneratorConfig(
        target_count=1,
        taxonomy_path="./tax.yaml",
        llm_override=StageLLMOverride(model="custom-model"),
    )
    client = resolve_generator_llm_client(ctx, cfg)
    assert client.config.model == "custom-model"

    # 4. apply_post_generation_decisions pass-through when disabled
    rec = _create_record("r1")
    kept, dropped, results = apply_post_generation_decisions(
        records=[rec],
        ctx=ctx,
        stage_config=cfg,
        stage_name="gen",
    )
    assert kept == [rec]
    assert dropped == []
    assert results == []


def test_taxonomy_generator_artifacts_and_override(tmp_path: Path) -> None:
    tax_path = tmp_path / "tax.yaml"
    tax_path.write_text(
        """
version: "1"
factors:
  - name: domain
    root:
      name: Domain
      children:
        - name: science
strategies:
  - name: all_science
    include_factors: [domain]
"""
    )

    class FakeTaxClient:
        def __init__(self) -> None:
            self.config = type("C", (), {"model": "fake-tax-model"})()

        def complete_structured(self, **kwargs):
            from arka.pipeline.taxonomy_generator import _GeneratedConversation

            return LLMOutput(
                text="{}",
                parsed=_GeneratedConversation(
                    instruction="Explain chemistry",
                    response="Chemistry is science",
                ),
                usage=TokenUsage(
                    prompt_tokens=10, completion_tokens=10, total_tokens=20
                ),
                model="fake-tax-model",
                provider="openai",
                latency_ms=10,
            )

    gen_cfg = TaxonomyGeneratorConfig(
        target_count=1,
        taxonomy_path=str(tax_path),
    )
    work_dir = tmp_path / "runs" / "r1" / "stages" / "01a_taxonomy_prompt"
    work_dir.mkdir(parents=True, exist_ok=True)
    ctx = StageContext(
        run_id="r1",
        stage_name="01a_taxonomy_prompt",
        work_dir=work_dir,
        config=_minimal_config(tmp_path),
        executor_mode="threadpool",
        max_workers=1,
    )

    stage = TaxonomyGeneratorStage(
        config=gen_cfg,
        project_root=tmp_path,
        llm_client=FakeTaxClient(),
    )
    out = stage.run([], ctx)
    assert len(out) == 1
    assert out[0].scores.get_annotation("taxonomy_nodes") == {"domain": ["science"]}

    # Verify single StageArtifacts write produced stats.json
    stats_file = work_dir / "stats.json"
    assert stats_file.exists()
    stats = json.loads(stats_file.read_text())
    assert stats["stage"] == "02_generate"
    assert stats["count_out"] == 1


# ---------------------------------------------------------------------------
# Candidate 4: CheckpointManager LLM call cache + map_parallel
# ---------------------------------------------------------------------------


def test_checkpoint_llm_cache_and_map_parallel(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    mgr = CheckpointManager(db_path)

    # LLM Cache
    mgr.save_llm_cache(
        cache_key="key-abc",
        response_json='{"content": "hello"}',
    )
    cached = mgr.load_llm_cache("key-abc")
    assert cached == '{"content": "hello"}'
    assert mgr.load_llm_cache("nonexistent") is None

    # StageContext.map_parallel
    ctx = StageContext(
        run_id="r1",
        stage_name="parallel_test",
        work_dir=tmp_path,
        config=_minimal_config(tmp_path),
        executor_mode="threadpool",
        max_workers=4,
    )
    items = [1, 2, 3, 4, 5]
    results = ctx.map_parallel(items, lambda x: x * 2)
    assert results == [2, 4, 6, 8, 10]


# ---------------------------------------------------------------------------
# Candidate 5: Multiple stages of same class preserve bound config
# ---------------------------------------------------------------------------


def test_multiple_stages_same_class_keep_own_bound_config(
    tmp_path: Path,
) -> None:
    from arka.pipeline.cheap_filters import LengthFilterStage

    ctx = StageContext(
        run_id="r1",
        stage_name="pipeline",
        work_dir=tmp_path,
        config=_minimal_config(tmp_path),
        executor_mode="threadpool",
        max_workers=1,
    )

    stage_min = LengthFilterStage(
        config=LengthFilterConfig(
            min_instruction_chars=10, max_instruction_chars=100, min_response_chars=0
        )
    )
    stage_max = LengthFilterStage(
        config=LengthFilterConfig(
            min_instruction_chars=0, max_instruction_chars=20, min_response_chars=0
        )
    )

    rec_short = _create_record("short", instruction="a", response="b")
    rec_med = _create_record("med", instruction="1234567890", response="b")
    rec_long = _create_record("long", instruction="x" * 30, response="b")

    # stage_min should keep its own min_instruction_chars=10
    kept_min = stage_min.run([rec_short, rec_med, rec_long], ctx)
    assert len(kept_min) == 2
    assert [r.id for r in kept_min] == ["med", "long"]

    # stage_max should keep its own max_instruction_chars=20
    kept_max = stage_max.run([rec_short, rec_med, rec_long], ctx)
    assert len(kept_max) == 2
    assert [r.id for r in kept_max] == ["short", "med"]


# ---------------------------------------------------------------------------
# Candidate 6: StageSpec registry & Declarative Capability Checks
# ---------------------------------------------------------------------------


def test_stage_spec_registry_and_capabilities() -> None:
    spec_score = get_stage_spec("labeling_score")
    assert spec_score is not None
    assert spec_score.get_stage_class() == LabelingScoreStage

    spec_ifd = get_stage_spec("ifd")
    assert spec_ifd is not None
    assert spec_ifd.requires_sequence_scoring is True

    spec_sem = get_stage_spec("semantic_similarity")
    assert spec_sem is not None
    assert spec_sem.requires_embeddings is True

    # Check declarative decision checks
    spec_panel = get_stage_spec("decision_panel")
    assert spec_panel is not None
    assert spec_panel.check_requires_decisions(None) is True

    spec_gen = get_stage_spec("prompt_based_generator")
    assert spec_gen is not None
    assert spec_gen.check_requires_decisions(None) is False
    # With best_of_n enabled
    cfg_with_bon = type(
        "C",
        (),
        {
            "best_of_n": type("B", (), {"enabled": True})(),
            "fidelity_check": None,
        },
    )()
    assert spec_gen.check_requires_decisions(cfg_with_bon) is True


# ---------------------------------------------------------------------------
# Candidate 7: Record.with_annotation & single-event escalation
# ---------------------------------------------------------------------------


def test_record_annotations_and_escalations() -> None:
    rec = _create_record("r1")

    # with_annotation keeps annotations and quality_per_dim in sync
    annotated = rec.with_annotation("seed_domain", "physics")
    assert annotated.scores.get_annotation("seed_domain") == "physics"
    assert annotated.scores.quality_per_dim["seed_domain"] == "physics"
    assert annotated.annotations["seed_domain"] == "physics"

    # with_escalation creates exactly 1 event and structured escalation entry
    escalated = annotated.with_escalation(
        stage="test_stage",
        reason_code="uncertain_answer",
        details="prob=0.5",
        question="q1",
        probability=0.5,
        confidence=0.0,
    )
    events = [e for e in escalated.stage_events if e.action == "escalated"]
    assert len(events) == 1
    assert events[0].reason_code == "uncertain_answer"

    escalations = escalated.scores.annotations.get("escalations", [])
    assert len(escalations) == 1
    assert escalations[0]["question"] == "q1"
    assert escalations[0]["probability"] == 0.5

    # apply_uncertain with on_uncertain="escalate" creates exactly ONE event
    opts = DecisionStageOptions(on_uncertain="escalate")
    res_rec, keep = apply_uncertain(
        rec,
        stage="gate",
        config=opts,
        reason="gate_uncertain",
        details=json.dumps({"q1": {"type": "yes_no", "probability": 0.5}}),
    )
    assert keep is True
    gate_events = [e for e in res_rec.stage_events if e.action == "escalated"]
    assert len(gate_events) == 1
    assert gate_events[0].stage == "gate"
    assert gate_events[0].reason_code == "gate_uncertain"

    # extract_sampled_from_record reads via get_annotation
    tax_rec = rec.with_annotation("taxonomy_nodes", {"domain": ["math"]})
    assert extract_sampled_from_record(tax_rec) == {"domain": ["math"]}


def test_execute_generator_slots_unlinks_stale_and_supports_direct_llm_output(
    tmp_path: Path,
) -> None:
    ctx = StageContext(
        run_id="r1",
        stage_name="gen",
        work_dir=tmp_path,
        config=_minimal_config(tmp_path),
        executor_mode="threadpool",
        max_workers=2,
    )
    p = tmp_path / "raw.jsonl"
    stale_row = RawGeneratorResponse(
        plan_index=0,
        seed_id="old-item",
        generated_text="old",
        prompt_hash="old_hash",
        model="test-model",
    )
    p.write_text(stale_row.model_dump_json() + "\n")

    out = execute_generator_slots(
        items=["item-1"],
        execute_slot_fn=lambda x: LLMOutput(
            text='{"instruction": "hi", "response": "hello"}',
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model="test-model",
            provider="openai",
            latency_ms=5,
        ),
        responses_path=p,
        ctx=ctx,
        stage_name="gen",
        prompt_hash="new_hash",
        get_item_id=lambda x: x,
    )
    assert len(out) == 1
    assert out[0].generated_text == '{"instruction": "hi", "response": "hello"}'
    assert out[0].prompt_hash == "new_hash"
    assert out[0].seed_id == "item-1"

    from pydantic import BaseModel

    class SimpleParsed(BaseModel):
        msg: str

    p2 = tmp_path / "raw2.jsonl"
    out2 = execute_generator_slots(
        items=["item-2"],
        execute_slot_fn=lambda x: LLMOutput(
            parsed=SimpleParsed(msg="world"),
            usage=TokenUsage(),
            model="test-model",
            provider="openai",
            latency_ms=5,
        ),
        responses_path=p2,
        ctx=ctx,
        stage_name="gen",
        prompt_hash="hash2",
        get_item_id=lambda x: x,
    )
    assert len(out2) == 1
    assert out2[0].generated_text == '{"msg":"world"}'


def test_apply_post_generation_decisions_hooks(monkeypatch, tmp_path: Path) -> None:
    ctx = StageContext(
        run_id="r1",
        stage_name="gen",
        work_dir=tmp_path,
        config=_minimal_config(tmp_path),
        executor_mode="threadpool",
        max_workers=1,
    )
    rec = _create_record("r1")
    stage_cfg = SimpleNamespace(
        fidelity_check=SimpleNamespace(enabled=True),
        best_of_n=SimpleNamespace(enabled=True),
    )
    import arka.pipeline.decision_checks as dc

    monkeypatch.setattr(
        dc, "taxonomy_fidelity", lambda records, ctx, cfg, name: (records, [], [])
    )
    monkeypatch.setattr(
        dc, "best_of_n", lambda records, ctx, cfg, name, kind: (records, [], [])
    )

    kept, dropped, results = apply_post_generation_decisions(
        records=[rec],
        ctx=ctx,
        stage_config=stage_cfg,
        stage_name="gen",
    )
    assert kept == [rec]


# ---------------------------------------------------------------------------
# PR #217 Review Regressions (Issues 1-4)
# ---------------------------------------------------------------------------


def test_review_issue_1_failed_generation_batch_saves_completed_responses(
    tmp_path: Path,
) -> None:
    cfg = _minimal_config(tmp_path)
    checkpoint = CheckpointManager(tmp_path / "state.db")
    ctx = StageContext(
        run_id="proof",
        stage_name="gen",
        work_dir=tmp_path,
        config=cfg,
        executor_mode="threadpool",
        max_workers=1,
        checkpoint_manager=checkpoint,
    )
    calls: list[str] = []

    def slot(item: str) -> Any:
        calls.append(item)
        if item == "bad":
            raise RuntimeError("simulated provider failure")
        return SimpleNamespace(text=f"valid response for {item}", model="fake")

    raw = tmp_path / "raw.jsonl"
    with pytest.raises(RuntimeError, match="simulated provider failure"):
        execute_generator_slots(
            items=["good", "bad"],
            execute_slot_fn=slot,
            responses_path=raw,
            ctx=ctx,
            stage_name="gen",
            prompt_hash="h",
            get_item_id=str,
        )

    assert calls == ["good", "bad"]
    saved = load_raw_responses(raw)
    assert len(saved) == 1
    assert saved[0].seed_id == "good"
    assert saved[0].generated_text == "valid response for good"

    # Resuming after fixing the provider failure should only execute "bad"
    calls.clear()
    resumed = execute_generator_slots(
        items=["good", "bad"],
        execute_slot_fn=lambda item: (
            calls.append(item)
            or SimpleNamespace(text=f"recovered response for {item}", model="fake")
        ),
        responses_path=raw,
        ctx=ctx,
        stage_name="gen",
        prompt_hash="h",
        get_item_id=str,
    )
    assert calls == ["bad"]
    assert len(resumed) == 2
    assert resumed[0].generated_text == "valid response for good"
    assert resumed[1].generated_text == "recovered response for bad"


def test_review_issue_2_truncated_checkpoint_preserves_valid_prefix_and_reports_corruption(
    tmp_path: Path,
) -> None:
    cfg = _minimal_config(tmp_path)
    ctx = StageContext(
        run_id="proof",
        stage_name="gen",
        work_dir=tmp_path,
        config=cfg,
        executor_mode="threadpool",
        max_workers=1,
    )
    raw = tmp_path / "raw.jsonl"
    valid = {
        "plan_index": 0,
        "seed_id": "good",
        "generated_text": "paid response",
        "prompt_hash": "h",
        "model": "fake",
        "latency_ms": 1,
        "usage": {},
    }
    raw.write_text(json.dumps(valid) + "\n" + '{"plan_index":1')

    loaded = load_raw_responses(raw)
    assert len(loaded) == 1
    assert loaded[0].generated_text == "paid response"

    calls: list[str] = []
    out = execute_generator_slots(
        items=["good"],
        execute_slot_fn=lambda item: (
            calls.append(item) or SimpleNamespace(text="new", model="fake")
        ),
        responses_path=raw,
        ctx=ctx,
        stage_name="gen",
        prompt_hash="h",
        get_item_id=str,
    )
    assert calls == []
    assert len(out) == 1
    assert "paid response" in raw.read_text()

    # Mid-file corruption (non-final line invalid) must raise ValueError instead of deleting
    corrupt_mid = tmp_path / "corrupt_mid.jsonl"
    corrupt_mid.write_text('{"corrupt": true}\n' + json.dumps(valid) + "\n")
    with pytest.raises(ValueError, match="Corrupted generator checkpoint"):
        load_raw_responses(corrupt_mid)
    assert corrupt_mid.exists()


def test_review_issue_3_llm_cache_enabled_in_pipeline_runner_and_populated_on_initial_run(
    tmp_path: Path,
) -> None:
    cfg = _minimal_config(tmp_path)
    seen: list[dict[str, Any]] = []

    class ProbeStage(Stage):
        name = "probe"

        def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
            client = ctx.llm_client()
            seen.append(
                {
                    "resume": ctx.resume,
                    "cache_db_path": client.cache_db_path,
                    "read_cache": getattr(client, "read_cache", None),
                }
            )
            if not ctx.resume:
                # Populate cache on initial run before stage fails
                key = client._compute_cache_key(
                    [{"role": "user", "content": "hello"}],
                )
                client._save_cached_llm(
                    key,
                    LLMOutput(
                        text="cached-answer",
                        usage=TokenUsage(
                            prompt_tokens=1, completion_tokens=1, total_tokens=2
                        ),
                        model=client.config.model,
                        provider=client.config.provider,
                        latency_ms=10,
                    ).model_dump_json(exclude={"parsed"}),
                )
                raise RuntimeError("leave stage incomplete")
            # On resume, reading the same prompt from client should hit the SQLite cache
            out = client.complete([{"role": "user", "content": "hello"}])
            assert out.text == "cached-answer"
            assert client.cached_calls == 1
            return [_create_record("r1", response=out.text)]

    with pytest.raises(RuntimeError, match="leave stage incomplete"):
        PipelineRunner(project_root=tmp_path).run(
            config=cfg,
            stages=[ProbeStage()],
            run_id="cache-proof",
            resume=False,
        )

    resumed_paths = PipelineRunner(project_root=tmp_path).run(
        config=cfg,
        stages=[ProbeStage()],
        run_id="cache-proof",
        resume=True,
    )
    assert resumed_paths.final_count == 1
    assert len(seen) == 2
    assert seen[0]["resume"] is False
    assert seen[0]["cache_db_path"] is not None
    assert seen[0]["read_cache"] is False
    assert seen[1]["resume"] is True
    assert seen[1]["cache_db_path"] == seen[0]["cache_db_path"]
    assert seen[1]["read_cache"] is True

    # Generator client disables prompt-addressed cache reads so repeated seed slots stay independent
    gen_ctx = StageContext(
        run_id="cache-proof",
        stage_name="02_generate",
        work_dir=tmp_path,
        config=cfg,
        executor_mode="threadpool",
        max_workers=1,
        checkpoint_manager=CheckpointManager(seen[0]["cache_db_path"]),
        resume=True,
    )
    gen_client = resolve_generator_llm_client(gen_ctx, cfg.generator)
    assert gen_client.cache_db_path == seen[0]["cache_db_path"]
    assert gen_client.read_cache is False


def test_review_issue_4_repeated_labeling_score_stages_retain_bound_rubrics(
    tmp_path: Path,
) -> None:
    rubric_1 = tmp_path / "first-rubric.yaml"
    rubric_2 = tmp_path / "second-rubric.yaml"
    for r_path, desc in ((rubric_1, "First"), (rubric_2, "Second")):
        r_path.write_text(
            f"""
version: "1.0"
description: "{desc}"
dimensions:
  - name: clarity
    description: "Clarity"
    scale_min: 1
    scale_max: 5
    criteria: {{1: "Poor", 5: "Great"}}
overall_weights:
  clarity: 1.0
"""
        )

    cfg = _minimal_config(tmp_path).model_copy(
        update={
            "pipeline": [
                LabelingScoreConfig(rubric_path="first-rubric.yaml"),
                LabelingScoreConfig(rubric_path="second-rubric.yaml"),
            ]
        }
    )
    stages = StageBuilder(cfg, tmp_path).build()
    assert len(stages) == 2
    stage_0, stage_1 = stages[0], stages[1]

    score_ctx_1 = StageContext(
        run_id="score",
        stage_name=stage_1.name,
        work_dir=tmp_path / "score_1",
        config=cfg,
        executor_mode="threadpool",
        max_workers=1,
    )
    stage_1.run([], score_ctx_1)
    assert isinstance(stage_1.config, LabelingScoreConfig)
    assert stage_1.config.rubric_path == "second-rubric.yaml"
    assert stage_0.config.rubric_path == "first-rubric.yaml"


def test_review_recheck_1_interrupted_checkpoint_rewrite_and_truncated_repair_are_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _minimal_config(tmp_path)
    raw = tmp_path / "raw.jsonl"
    ctx = StageContext(
        run_id="proof",
        stage_name="gen",
        work_dir=tmp_path,
        config=cfg,
        executor_mode="threadpool",
        max_workers=1,
    )
    original_write_text = Path.write_text

    def fail_rewrite(path: Path, text: str, *args: Any, **kwargs: Any) -> int:
        if path == raw or path.name.endswith(".tmp"):
            with path.open("w", encoding="utf-8") as handle:
                handle.write(text[:10])
            raise OSError("simulated interrupted rewrite")
        return original_write_text(path, text, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(Path, "write_text", fail_rewrite)
        try:
            execute_generator_slots(
                items=["good"],
                execute_slot_fn=lambda _: SimpleNamespace(
                    text="paid response", model="fake"
                ),
                responses_path=raw,
                ctx=ctx,
                stage_name="gen",
                prompt_hash="h",
                get_item_id=str,
            )
        except OSError:
            pass

    assert len(load_raw_responses(raw)) == 1
    assert load_raw_responses(raw)[0].generated_text == "paid response"

    # Truncated-tail repair must also replace atomically via a temporary file
    valid = {
        "plan_index": 0,
        "seed_id": "good",
        "generated_text": "saved before crash",
        "prompt_hash": "h",
        "model": "fake",
        "latency_ms": 1,
        "usage": {},
    }
    raw_tail = tmp_path / "raw_tail.jsonl"
    original_write_text(raw_tail, json.dumps(valid) + "\n" + '{"plan_index":1')
    with monkeypatch.context() as m:
        m.setattr(Path, "write_text", fail_rewrite)
        with pytest.raises(OSError, match="simulated interrupted rewrite"):
            load_raw_responses(raw_tail)
    # Even after interrupted repair write, the original valid prefix is still recoverable
    recovered = load_raw_responses(raw_tail)
    assert len(recovered) == 1
    assert recovered[0].generated_text == "saved before crash"


def test_review_recheck_2_evol_instruct_resume_preserves_distinct_branch_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from arka.config.models import EvolInstructGeneratorConfig
    from arka.llm.client import LLMClient
    from arka.pipeline.evol_generator_stage import EvolInstructRoundStage

    cfg = _minimal_config(tmp_path)
    evol_cfg = EvolInstructGeneratorConfig(
        rounds=1, branching_factor=2, operators=["deepen"]
    )
    cfg = cfg.model_copy(
        update={
            "pipeline": [evol_cfg],
            "llm": cfg.llm.model_copy(update={"supports_json_schema": True}),
        }
    )

    responses = [
        json.dumps({"instruction": "Explain gravity with Newtonian formulas."}),
        json.dumps({"response": "Newtonian gravity is proportional to mass."}),
        json.dumps(
            {"instruction": "Compare gravity in Newtonian mechanics and relativity."}
        ),
        json.dumps({"response": "Relativity describes spacetime curvature."}),
    ]
    call_count = 0

    class FakeCompletions:
        def create(self, **kwargs: Any) -> Any:
            nonlocal call_count
            assert call_count < len(responses), "Unexpected extra provider call"
            text = responses[call_count]
            call_count += 1
            return SimpleNamespace(
                id=f"req-{call_count}",
                model="gpt-4o-mini",
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content=text),
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=5, completion_tokens=5, total_tokens=10
                ),
            )

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    def factory(
        config: Any, cache_db_path: Path | None = None, read_cache: bool = True
    ) -> LLMClient:
        return LLMClient(
            config,
            client_factory=lambda _: sdk,
            cache_db_path=cache_db_path,
            read_cache=read_cache,
        )

    checkpoint = CheckpointManager(tmp_path / "state.db")
    stage = EvolInstructRoundStage(round_number=1, config=evol_cfg)
    evol_ctx = StageContext(
        run_id="proof",
        stage_name=stage.name,
        work_dir=tmp_path,
        config=cfg,
        executor_mode="threadpool",
        max_workers=1,
        checkpoint_manager=checkpoint,
        llm_client_factory=factory,
    )
    parent = _create_record(
        "seed", instruction="Explain gravity", response="Gravity attracts masses."
    )

    with monkeypatch.context() as m:
        m.setattr(
            stage,
            "_write_artifacts",
            lambda **kwargs: (_ for _ in ()).throw(
                OSError("simulated artifact write failure")
            ),
        )
        with pytest.raises(OSError, match="simulated artifact write failure"):
            stage.run([parent], evol_ctx)

    assert call_count == 4
    out = stage.run([parent], replace(evol_ctx, resume=True))
    assert call_count == 4
    assert [r.payload.instruction for r in out[1:]] == [
        "Explain gravity with Newtonian formulas.",
        "Compare gravity in Newtonian mechanics and relativity.",
    ]
