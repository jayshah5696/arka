"""Exercise decision stages with real transport, cache, artifacts, and no LLM.

Failure cases include erased scores, coupled inverse critics, wrong rubric
scales, uncertain records accepted silently, and panel answers reused after
changing provider settings or input.
"""

import json
from dataclasses import replace
from pathlib import Path

import httpx
import polars as pl
import pytest

from arka.config.loader import ConfigLoader, ConfigValidationError
from arka.decisions.client import DecisionClient
from arka.pipeline.checkpoint import CheckpointManager
from arka.pipeline.models import StageContext
from arka.pipeline.stage_builder import StageBuilder
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)

ROOT = Path(__file__).resolve().parents[2]


def record(rid="1", **scores):
    return ConversationRecord(
        id=rid,
        content_hash=rid,
        source=RecordSource(type="seed"),
        lineage=RecordLineage(root_id=rid, parent_ids=[]),
        payload=ConversationPayload(
            instruction="Explain gravity.", response="Mass attracts mass."
        ),
        scores=RecordScores(**scores),
        config_hash="cfg",
        created_at="2026-10-12T00:00:00Z",
    )


def context(tmp_path, pipeline, values=None):
    config = ConfigLoader().load_dict(
        {
            "version": "1",
            "llm": {
                "provider": "openai",
                "model": "generator",
                "api_key": "secret",
                "base_url": "https://example.test/v1",
            },
            "decisions": {
                "provider": "openai",
                "model": "decision",
                "api_key": "secret",
                "base_url": "https://example.test/v1",
            },
            "executor": {"max_workers": 2},
            "pipeline": pipeline,
            "output": {"format": "jsonl", "path": "output.jsonl"},
        }
    )
    calls = []

    def handle(request):
        payload = json.loads(request.content)
        calls.append(payload)
        answers = []
        for q in payload["questions"]:
            name = q["name"]
            value = (values or {}).get(name, 0.9)
            if callable(value):
                value = value(payload)
            if q["type"] == "predicate":
                answers.append(
                    {"name": name, "type": "predicate", "probability": value}
                )
            else:
                options = (
                    [c["value"] for c in q["choices"]]
                    if q["type"] == "choice"
                    else list(range(len(q["levels"])))
                )
                choice = value if q["type"] == "choice" else len(options) - 1
                confidence = (values or {}).get(name + ".confidence", 0.95)
                answers.append(
                    {
                        "name": name,
                        "type": q["type"],
                        q["type"]: choice,
                        "probabilities": [
                            {"value": x, "probability": float(x == choice)}
                            for x in options
                        ],
                        "confidence": confidence,
                    }
                )
        return httpx.Response(
            200,
            json={
                "answers": answers,
                "usage": {"input_tokens": 10},
                "model": "decision",
            },
        )

    checkpoint = CheckpointManager(tmp_path / "checkpoint.sqlite")

    def factory(cfg):
        return DecisionClient(
            cfg, checkpoint=checkpoint, transport=httpx.MockTransport(handle)
        )

    def no_llm(cfg):
        raise AssertionError("Decision stages must never call an LLM fallback")

    ctx = StageContext(
        run_id="test",
        stage_name="test",
        work_dir=tmp_path,
        config=config,
        executor_mode="threadpool",
        max_workers=2,
        checkpoint_manager=checkpoint,
        decision_client_factory=factory,
        llm_client_factory=no_llm,
    )
    return ctx, calls


def run_stages(ctx, records):
    for i, stage in enumerate(StageBuilder(ctx.config, ROOT).build()):
        stage_ctx = replace(ctx, stage_name=stage.name, work_dir=ctx.work_dir / str(i))
        records = stage.run(records, stage_ctx)
    return records


def test_panel_populates_judges_and_inverse_critic_remains_separate(tmp_path):
    pipeline = [
        {"type": "decision_panel"},
        {
            "type": "labeling_engine",
            "backend": "decision",
            "rubric_path": "rubrics/sft_quality.yaml",
            "min_overall_score": 3,
        },
        {"type": "complexity_elo", "backend": "decision"},
        {"type": "reward_model", "backend": "decision"},
        {"type": "double_critic", "backend": "decision"},
    ]
    ctx, calls = context(tmp_path, pipeline, {"double_critic.is_incorrect": 0.1})
    out = run_stages(ctx, [record(ifd=0.7, quality_per_dim={"old": 2})])
    assert len(out) == 1
    scores = out[0].scores
    assert scores.ifd == 0.7
    assert scores.quality_per_dim["old"] == 2
    assert scores.quality == 5
    assert scores.reward_model == 9
    assert scores.quality_per_dim["complexity"] == 5
    assert scores.quality_per_dim["complexity_elo"] == 1250
    assert scores.decisions["double_critic.is_incorrect"].probability == 0.1
    assert len(calls) == 2
    assert [q["name"] for q in calls[1]["questions"]] == ["double_critic.is_incorrect"]
    assert run_stages(ctx, [record(ifd=0.7)])
    assert len(calls) == 2
    serialized = pl.read_parquet(tmp_path / "4" / "dropped.parquet")
    assert serialized.height == 0


@pytest.mark.parametrize(
    "probability,policy,kept,escalated",
    [
        (0.9, "drop", True, False),
        (0.1, "keep", False, False),
        (0.5, "escalate", True, True),
        (0.5, "drop", False, False),
        (0.5, "keep", True, False),
    ],
)
def test_double_critic_bands_and_no_fallback(
    tmp_path, probability, policy, kept, escalated
):
    ctx, _ = context(
        tmp_path,
        [{"type": "double_critic", "backend": "decision", "on_uncertain": policy}],
        {
            "double_critic.is_correct": probability,
            "double_critic.is_incorrect": 1 - probability,
        },
    )
    out = run_stages(ctx, [record()])
    assert bool(out) is kept
    if out:
        assert out[0].is_escalated() is escalated
    stats = json.loads((tmp_path / "0" / "stats.json").read_text())
    assert stats["decision_usage"]["requests"] == 2


def test_rubric_questions_interpolate_missing_levels(tmp_path):
    ctx, calls = context(
        tmp_path,
        [
            {
                "type": "labeling_engine",
                "backend": "decision",
                "rubric_path": "rubrics/sft_quality.yaml",
            }
        ],
    )
    assert run_stages(ctx, [record()])
    questions = calls[0]["questions"]
    assert all(len(q["levels"]) == 5 for q in questions)
    assert any("between" in q["levels"][1]["description"].lower() for q in questions)


def test_low_confidence_reward_escalates_even_with_high_score(tmp_path):
    ctx, _ = context(
        tmp_path,
        [
            {
                "type": "reward_model",
                "backend": "decision",
                "min_confidence": 0.5,
                "on_uncertain": "escalate",
            }
        ],
        {"reward_model.helpfulness.confidence": 0.2},
    )
    out = run_stages(ctx, [record()])
    assert out[0].scores.reward_model == 9
    assert out[0].is_escalated()


def test_seed_gate_drops_bad_seeds_before_generation(tmp_path):
    ctx, _ = context(tmp_path, [{"type": "seed_gate"}], {"seed_gate.usable": 0.1})
    assert run_stages(ctx, [record()]) == []
    stats = json.loads((tmp_path / "0" / "stats.json").read_text())
    assert stats["drop_reasons"] == {"seed_unusable": 1}


def test_language_checks_both_fields_and_ignores_code(tmp_path):
    pipeline = [{"type": "language", "backend": "decision", "allowed": ["en", "de"]}]
    ctx, _ = context(
        tmp_path,
        pipeline,
        {"language.language_instruction": "en", "language.language_response": "de"},
    )
    assert run_stages(ctx, [record()]) == []
    assert json.loads((tmp_path / "0" / "stats.json").read_text())["drop_reasons"] == {
        "language_field_disagreement": 1
    }
    ctx, _ = context(
        tmp_path / "code",
        pipeline,
        {"language.language_instruction": "en", "language.language_response": "none"},
    )
    assert run_stages(ctx, [record()])


def test_identity_decision_catches_paraphrases(tmp_path):
    ctx, _ = context(
        tmp_path,
        [{"type": "identity_claim", "backend": "regex_then_decision"}],
        {
            "identity_claim.claims_foreign_identity": 0.9,
            "identity_claim.claimed_identity": "google",
        },
    )
    assert run_stages(ctx, [record()]) == []
    assert json.loads((tmp_path / "0" / "stats.json").read_text())["drop_reasons"] == {
        "identity_claim_paraphrase": 1
    }


@pytest.mark.parametrize(
    "pipeline",
    [
        [
            {
                "type": "reward_model",
                "backend": "decision",
                "reject_at": 0.9,
                "accept_at": 0.8,
            }
        ],
        [{"type": "prompt_based_generator"}, {"type": "seed_gate"}],
        [{"type": "labeling_engine", "backend": "decision"}],
    ],
)
def test_invalid_decision_config_rejected(tmp_path, pipeline):
    with pytest.raises(ConfigValidationError):
        context(tmp_path, pipeline)


def test_provider_batch_rejected():
    with pytest.raises(ValueError, match="provider_batch"):
        from arka.config.models import ExecutorConfig

        ExecutorConfig(mode="provider_batch")


def test_select_resolves_normalizes_and_weights_decision_confidence(tmp_path):
    from arka.decisions.models import ScoreAnswer

    confident = record(
        "confident",
        decisions={
            "reward_model.helpfulness": ScoreAnswer(
                score=8,
                probabilities={i: float(i == 8) for i in range(10)},
                confidence=0.9,
            )
        },
    )
    unsure = record(
        "unsure",
        decisions={
            "reward_model.helpfulness": ScoreAnswer(
                score=9,
                probabilities={i: float(i == 9) for i in range(10)},
                confidence=0.4,
            )
        },
    )
    ctx, calls = context(
        tmp_path,
        [
            {
                "type": "select",
                "target_count": 1,
                "weights": {"decisions.reward_model.helpfulness": 1},
                "confidence_weighting": True,
            }
        ],
    )
    assert [r.id for r in run_stages(ctx, [unsure, confident])] == ["confident"]
    assert calls == []


@pytest.mark.parametrize(
    "preferred,reason",
    [("B", "pairwise_not_preferred"), ("A", "pairwise_position_inconsistent")],
)
def test_pairwise_check_swaps_order(tmp_path, preferred, reason):
    from arka.pipeline.scoring_stages import PairDeltaFilterStage

    choice = (
        (
            lambda payload: (
                "A" if json.loads(payload["input"])["A"] == "Parent response" else "B"
            )
        )
        if preferred == "B"
        else "A"
    )
    ctx, calls = context(
        tmp_path,
        [{"type": "pair_delta", "pairwise_check": {"enabled": True}}],
        {"pair_delta.better_response": choice},
    )
    parent = record("parent", quality=1)
    parent = parent.model_copy(
        update={
            "payload": parent.payload.model_copy(update={"response": "Parent response"})
        }
    )
    child = record("child", quality=2)
    child = child.model_copy(
        update={"lineage": child.lineage.model_copy(update={"parent_ids": [parent.id]})}
    )
    stage = PairDeltaFilterStage(config=ctx.config.pipeline[0])
    assert [r.id for r in stage.run([parent, child], ctx)] == ["parent"]
    assert json.loads((tmp_path / "stats.json").read_text())["drop_reasons"] == {
        reason: 1
    }
    assert len(calls) == 2


def test_semantic_band_uses_decisions_only_inside_band(tmp_path, monkeypatch):
    import numpy as np

    class Embedder:
        def __init__(self, config):
            pass

        def embed(self, texts, **kwargs):
            return (
                np.array([[1, 0]])
                if len(texts) == 1
                else np.array([[0.92, (1 - 0.92**2) ** 0.5], [0, 1]])
            )

    monkeypatch.setattr("arka.pipeline.filter_stages.Embedder", Embedder)
    ctx, calls = context(
        tmp_path,
        [{"type": "semantic_similarity", "decision_band": {"enabled": True}}],
        {
            "semantic_similarity.copies_seed": 0.1,
            "semantic_similarity.leaks_specifics": 0.1,
        },
    )
    children = [record("a"), record("b")]
    children = [
        r.model_copy(update={"source": RecordSource(type="generated")})
        for r in children
    ]
    assert len(run_stages(ctx, [record("seed"), *children])) == 3
    assert len(calls) == 1


def test_pdf_front_matter_decision_check_preserves_provenance(tmp_path):
    from arka.pipeline.decision_checks import filter_pdf_chunks
    from arka.records.models import GroundedChunkPayload, GroundedChunkRecord

    ctx, _ = context(
        tmp_path,
        [
            {
                "type": "pdf_source",
                "path": "missing.pdf",
                "decision_filter": {"enabled": True},
            }
        ],
        {"pdf_source.is_front_or_back_matter": 0.9},
    )
    base = record().model_dump(exclude={"payload"})
    chunk = GroundedChunkRecord(
        **base,
        payload=GroundedChunkPayload(
            text="Table of contents",
            doc_id="doc",
            chunk_idx=0,
            page_start=1,
            page_end=1,
            char_start=0,
            char_end=17,
            word_count=3,
            chunk_strategy="fixed",
        ),
    )
    kept, dropped, _ = filter_pdf_chunks(
        [chunk], ctx, ctx.config.pipeline[0].decision_filter, "pdf"
    )
    assert kept == []
    assert dropped[0].payload.page_start == 1
    assert dropped[0].stage_events[-1].reason_code == "pdf_front_back_matter"


def test_best_of_n_keeps_one_candidate_per_parent(tmp_path):
    from arka.pipeline.decision_checks import best_of_n

    ctx, _ = context(
        tmp_path,
        [
            {
                "type": "prompt_based_generator",
                "best_of_n": {"enabled": True, "keep_per_seed": 1},
            }
        ],
    )
    records = [record(str(i)) for i in range(3)]
    records = [
        r.model_copy(
            update={"lineage": r.lineage.model_copy(update={"parent_ids": ["seed"]})}
        )
        for r in records
    ]
    kept, dropped, _ = best_of_n(
        records, ctx, ctx.config.pipeline[0].best_of_n, "generator"
    )
    assert [r.id for r in kept] == ["0"]
    assert len(dropped) == 2
    assert all(
        r.stage_events[-1].reason_code == "best_of_n_not_selected" for r in dropped
    )


def test_taxonomy_fidelity_removes_unmet_nodes(tmp_path):
    from arka.pipeline.decision_checks import taxonomy_fidelity

    ctx, _ = context(
        tmp_path,
        [
            {
                "type": "taxonomy_generator",
                "taxonomy_path": "missing.yaml",
                "fidelity_check": {"enabled": True, "relabel": True},
            }
        ],
        {"taxonomy_generator.reflects_topic": 0.1},
    )
    rec = record(quality_per_dim={"taxonomy_nodes": {"topic": ["finance"]}})
    kept, dropped, _ = taxonomy_fidelity(
        [rec], ctx, ctx.config.pipeline[0].fidelity_check, "taxonomy"
    )
    assert not dropped
    assert kept[0].scores.quality_per_dim["taxonomy_nodes"] == {}
    assert kept[0].scores.quality_per_dim["taxonomy_nodes_requested"] == {
        "topic": ["finance"]
    }


def test_evolution_checks_refusal_on_response(tmp_path):
    from arka.pipeline.decision_checks import check_evolution

    ctx, _ = context(
        tmp_path,
        [
            {
                "type": "evol_instruct_generator",
                "rounds": 1,
                "branching_factor": 1,
                "operators": ["deepen"],
                "filter": {"backend": "decision"},
            }
        ],
        {"evol_instruct_generator.is_refusal": 0.9},
    )
    updated, reason, _ = check_evolution(
        record("child"), record("parent"), ctx, ctx.config.pipeline[0], "evol"
    )
    assert reason == "evol_refusal"
    assert (
        updated.scores.decisions["evol_instruct_generator.is_refusal"].probability
        == 0.9
    )


def test_reasoning_language_decides_low_confidence_fields(tmp_path):
    from arka.pipeline.reasoning_language import (
        LanguagePrediction,
        ReasoningLanguageGateStage,
    )

    class Detector:
        def predict(self, text):
            return LanguagePrediction("en", 0.1 if text == "短文本" else 0.99)

    ctx, calls = context(
        tmp_path,
        [{"type": "reasoning_language_gate", "low_confidence_backend": "decision"}],
        {"reasoning_language_gate.same_language_reasoning": 0.1},
    )
    rec = record()
    rec = rec.model_copy(
        update={"payload": rec.payload.model_copy(update={"reasoning": "短文本"})}
    )
    out = ReasoningLanguageGateStage(
        config=ctx.config.pipeline[0], detector=Detector()
    ).run([rec], ctx)
    assert out[0].payload.reasoning is None
    assert len(calls) == 1


def test_escalation_output_policy_and_manifest(tmp_path):
    from arka.pipeline.runner import PipelineRunner

    ctx, _ = context(
        tmp_path,
        [{"type": "double_critic", "backend": "decision", "on_uncertain": "escalate"}],
        {"double_critic.is_correct": 0.5, "double_critic.is_incorrect": 0.5},
    )
    config = ctx.config.model_copy(
        update={
            "output": ctx.config.output.model_copy(
                update={"on_unresolved_escalation": "drop"}
            )
        }
    )
    from arka.pipeline.stages import Stage

    class Source(Stage):
        name = "source"

        def run(self, records, ctx):
            return [record()]

    stages = [Source(), *StageBuilder(config, ROOT).build()]
    runner = PipelineRunner(
        project_root=tmp_path, decision_client_factory=ctx.decision_client_factory
    )
    result = runner.run(config=config, run_id="escalate", stages=stages)
    assert result.final_count == 0
    queue = pl.read_parquet(tmp_path / "runs" / "escalate" / "escalations.parquet")
    assert queue.height == 1
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["decision_usage"]["input_tokens"] == 20
    assert manifest["escalation_counts"]["unresolved"] == 1


def test_select_labels_seed_taxonomy_before_reserving(tmp_path):
    taxonomy = tmp_path / "taxonomy.yaml"
    taxonomy.write_text(
        "factors:\n  - factor: topic\n    root:\n      name: topic\n      children:\n        - name: physics\n        - name: finance\n"
    )
    ctx, calls = context(
        tmp_path,
        [
            {
                "type": "select",
                "target_count": 1,
                "coverage": [{"factor": "topic", "path": ["physics"], "min_count": 1}],
                "label_taxonomy": {"enabled": True, "taxonomy_path": str(taxonomy)},
            }
        ],
        {"select.taxonomy_topic": "physics"},
    )
    out = run_stages(
        ctx,
        [
            record("unlabeled"),
            record(
                "labeled", quality_per_dim={"taxonomy_nodes": {"topic": ["finance"]}}
            ),
        ],
    )
    assert [r.id for r in out] == ["unlabeled"]
    assert out[0].scores.quality_per_dim["taxonomy_labeled_by"] == "decision"
    assert len(calls) == 1


def test_decontamination_decides_ngram_near_miss(tmp_path, monkeypatch):
    from arka.pipeline.decontamination import BenchmarkIndex
    from arka.records.models import DecontaminationMatch

    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        '{"question": "A train travels fifty miles in one hour. What is its speed?"}\n'
    )
    monkeypatch.setattr(
        BenchmarkIndex,
        "match",
        lambda self, text, field, window_factor: DecontaminationMatch(
            benchmark="benchmark.jsonl",
            eval_id=1,
            score=0.25,
            field=field,
            window_start=0,
            window_end=5,
        ),
    )
    ctx, calls = context(
        tmp_path,
        [
            {
                "type": "decontaminate",
                "eval_paths": [str(benchmark)],
                "threshold": 0.45,
                "paraphrase_check": {"enabled": True},
            }
        ],
        {"decontaminate.same_problem": 0.9},
    )
    assert run_stages(ctx, [record()]) == []
    assert json.loads((tmp_path / "0" / "stats.json").read_text())["drop_reasons"] == {
        "benchmark_paraphrase": 1
    }
    assert len(calls) == 1


def test_parallel_generation_preserves_raw_order_and_resumes(tmp_path):
    import threading
    import time

    from arka.llm.models import LLMOutput, TokenUsage
    from arka.pipeline.generator_stages import PromptBasedGeneratorStage

    active = peak = calls = 0
    lock = threading.Lock()

    class Generator:
        def complete_structured(self, messages, schema, **kwargs):
            nonlocal active, peak, calls
            with lock:
                calls += 1
                active += 1
                peak = max(peak, active)
                index = calls
            time.sleep(0.01)
            with lock:
                active -= 1
            parsed = schema(instruction=f"Generated {index}", response="Answer")
            return LLMOutput(
                text=parsed.model_dump_json(),
                parsed=parsed,
                usage=TokenUsage(),
                model="generator",
                provider="openai",
                latency_ms=1,
            )

    ctx, _ = context(tmp_path, [{"type": "prompt_based_generator", "target_count": 4}])
    stage = PromptBasedGeneratorStage(
        config=ctx.config.pipeline[0], llm_client=Generator()
    )
    out = stage.run([record()], ctx)
    assert len(out) == 4
    assert peak == 2
    assert [
        json.loads(line)["plan_index"]
        for line in (tmp_path / "raw_responses.jsonl").read_text().splitlines()
    ] == [0, 1, 2, 3]
    assert len(stage.run([record()], ctx)) == 4
    assert calls == 4


def test_fasttext_language_backend_needs_no_decision_client(tmp_path, monkeypatch):
    from arka.pipeline.reasoning_language import LanguagePrediction

    class Detector:
        def __init__(self, path):
            pass

        def predict(self, text):
            return LanguagePrediction("fr", 0.99)

    monkeypatch.setattr("arka.pipeline.reasoning_language.FastTextDetector", Detector)
    ctx, calls = context(
        tmp_path,
        [
            {
                "type": "language",
                "backend": "fasttext",
                "lid_model_path": "lid.bin",
                "allowed": ["en"],
            }
        ],
    )
    assert run_stages(ctx, [record()]) == []
    assert calls == []
