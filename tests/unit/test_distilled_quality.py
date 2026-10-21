"""Training must not leak related records or overwrite teacher judgments.

Failure cases: absent or stale labels, one-class training, overlapping sources,
missing/nonfinite embeddings, incompatible metadata, changed external artifacts,
and inference that silently retrains or changes record order.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from arka.config.loader import ConfigLoader
from arka.labeling.rubric import RubricLoader
from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.pipeline.stage_builder import StageBuilder
from arka.records.identity import content_hash
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


def make_record(name, good=True, source=None):
    payload = ConversationPayload(
        instruction=f"Task {name}", response="GOOD" if good else "BAD"
    )
    return ConversationRecord(
        id=name,
        content_hash=content_hash(payload),
        source=RecordSource(type="seed", seed_file_hash=source),
        lineage=RecordLineage(root_id=name, parent_ids=[]),
        payload=payload,
        scores=RecordScores(),
        config_hash="fixture",
        created_at="2026-10-19",
    )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text(
        yaml.safe_dump(
            {
                "version": "1",
                "description": "Quality",
                "dimensions": [
                    {
                        "name": "correctness",
                        "description": "Correct answer",
                        "scale_min": 0,
                        "scale_max": 4,
                        "criteria": {0: "Bad", 4: "Good"},
                    }
                ],
                "overall_weights": {"correctness": 1},
            }
        )
    )
    rubric_hash = RubricLoader().load(rubric).hash
    records = []
    for group in range(8):
        for good in (False, True):
            r = make_record(f"{group}-{good}", good, f"source-{group}")
            records.append(
                r.model_copy(
                    update={
                        "scores": RecordScores(
                            quality=4 if good else 0,
                            quality_per_dim={"correctness": 4 if good else 0},
                            rubric_hash=rubric_hash,
                            judge_model="teacher",
                            judge_prompt_hash="prompt",
                        )
                    }
                )
            )
    data = tmp_path / "labeled.parquet"
    OutputWriter().write_parquet(records, Path(data))
    calls = []

    def embed(self, texts, **kwargs):
        calls.extend(texts)
        return np.array([[1.0, 0.0] if '"GOOD"' in t else [0.0, 1.0] for t in texts])

    monkeypatch.setattr("arka.embeddings.embedder.Embedder.embed", embed)

    def context(stage_config, directory="train", resume=False):
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
                "embeddings": {
                    "provider": "openai",
                    "model": "fixture-embedding",
                    "base_url": "http://localhost:9000/v1",
                },
                "pipeline": [stage_config],
                "output": {"format": "jsonl", "path": "output.jsonl"},
            }
        )
        stage = StageBuilder(cfg, tmp_path).build()[0]
        return stage, StageContext(
            run_id=directory,
            stage_name=stage.name,
            work_dir=tmp_path / directory,
            config=cfg,
            executor_mode="threadpool",
            max_workers=1,
            resume=resume,
        )

    train_config = {
        "type": "distilled_quality_train",
        "training_paths": [str(data)],
        "rubric_path": str(rubric),
        "min_scores": {"correctness": 3},
    }
    return context, train_config, records, calls


def train(setup):
    context, config, records, calls = setup
    stage, ctx = context(config)
    result = stage.run([], ctx)
    return context, config, records, calls, stage, ctx, result


def test_training_separates_sources_and_scores_without_teacher_mutation(setup):
    context, config, records, calls, stage, ctx, result = train(setup)
    model = json.loads((ctx.work_dir / "model.json").read_text())
    assert set(model["train_groups"]).isdisjoint(model["holdout_groups"])
    assert model["evaluation"]["useful_record_recall"] == 1
    assert model["evaluation"]["false_rejections"] == 0
    assert len(calls) == len(records)
    assert result.kept == []
    score, score_ctx = context(
        {
            "type": "distilled_quality_score",
            "model_path": str(ctx.work_dir / "model.json"),
            "rubric_path": config["rubric_path"],
            "min_scores": config["min_scores"],
        },
        "score",
    )
    scored = score.run(records, score_ctx)
    assert [r.id for r in scored.kept] == [r.id for r in records]
    for old, new in zip(records, scored.kept, strict=True):
        assert new.payload == old.payload
        assert new.scores.quality == old.scores.quality
        assert new.scores.quality_per_dim == old.scores.quality_per_dim
        assert (new.scores.distilled_quality >= 0.5) == (old.payload.response == "GOOD")
        assert new.scores.annotations["distilled_quality"]["model_hash"]
    assert not (score_ctx.work_dir / "model.json").exists()


def test_explicit_prefilter_and_independent_evaluation(setup):
    context, config, records, _, _, ctx, _ = train(setup)
    stage, score_ctx = context(
        {
            "type": "distilled_quality_score",
            "model_path": str(ctx.work_dir / "model.json"),
            "rubric_path": config["rubric_path"],
            "min_scores": config["min_scores"],
            "min_probability": 0.5,
            "evaluate": True,
        },
        "score",
    )
    result = stage.run(records, score_ctx)
    assert len(result.kept) == len(result.dropped) == 8
    assert all(r.payload.response == "GOOD" for r in result.kept)
    assert all(
        r.stage_events[-1].reason_code == "distilled_quality_below_min"
        for r in result.dropped
    )
    assert result.report.model_dump()["evaluation"]["useful_record_recall"] == 1
    assert result.report.model_dump()["evaluation_independent"] is False


@pytest.mark.parametrize(
    "failure",
    [
        "missing_labels",
        "rubric",
        "single_class",
        "one_source",
        "nonfinite",
        "missing_embeddings",
        "changed_text",
    ],
)
def test_invalid_training_fails_without_a_model(setup, monkeypatch, failure):
    context, config, records, calls = setup
    data = Path(config["training_paths"][0])
    if failure == "missing_labels":
        records[0] = records[0].model_copy(update={"scores": RecordScores()})
    elif failure == "rubric":
        records[0] = records[0].model_copy(
            update={
                "scores": records[0].scores.model_copy(update={"rubric_hash": "stale"})
            }
        )
    elif failure == "single_class":
        records = [r for r in records if r.payload.response == "GOOD"]
    elif failure == "one_source":
        records = [
            r.model_copy(
                update={"source": RecordSource(type="seed", seed_file_hash="same")}
            )
            for r in records
        ]
    elif failure == "changed_text":
        records[0] = records[0].model_copy(
            update={
                "payload": records[0].payload.model_copy(
                    update={"response": "Changed after judging"}
                )
            }
        )
    else:
        monkeypatch.setattr(
            "arka.embeddings.embedder.Embedder.embed",
            lambda *a, **k: (
                None
                if failure == "missing_embeddings"
                else np.full((len(records), 2), np.nan)
            ),
        )
    OutputWriter().write_parquet(records, data)
    stage, ctx = context(config)
    with pytest.raises(ValueError):
        stage.run([], ctx)
    assert not (ctx.work_dir / "model.json").exists()
    if failure in {
        "missing_labels",
        "rubric",
        "single_class",
        "one_source",
        "changed_text",
    }:
        assert not calls


def test_decision_teacher_labels_keep_actual_confidence_and_provenance(setup):
    from arka.decisions.models import ScoreAnswer

    context, config, records, calls = setup
    for i, record in enumerate(records):
        value = int(record.scores.quality_per_dim["correctness"])
        records[i] = record.model_copy(
            update={
                "scores": record.scores.model_copy(
                    update={
                        "judge_prompt_hash": None,
                        "decisions": {
                            "labeling_engine.correctness": ScoreAnswer(
                                score=value,
                                confidence=1,
                                probabilities={j: float(j == value) for j in range(5)},
                            )
                        },
                    }
                )
            }
        )
    OutputWriter().write_parquet(records, Path(config["training_paths"][0]))
    stage, ctx = context(config)
    stage.run([], ctx)
    assert len(calls) == len(records)
    records[0] = records[0].model_copy(
        update={
            "scores": records[0].scores.model_copy(
                update={
                    "decisions": {
                        "labeling_engine.correctness": records[0]
                        .scores.decisions["labeling_engine.correctness"]
                        .model_copy(update={"confidence": 0.1})
                    }
                }
            )
        }
    )
    OutputWriter().write_parquet(records, Path(config["training_paths"][0]))
    stage, ctx = context(config, "uncertain")
    with pytest.raises(ValueError, match="confidence"):
        stage.run([], ctx)


@pytest.mark.parametrize(
    "change", ["rubric", "embedding", "model", "dimension", "privacy"]
)
def test_scoring_rejects_incompatible_model_before_embedding(setup, change):
    context, config, records, calls, _, ctx, _ = train(setup)
    calls.clear()
    options = {
        "type": "distilled_quality_score",
        "model_path": str(ctx.work_dir / "model.json"),
        "rubric_path": config["rubric_path"],
        "min_scores": config["min_scores"],
    }
    stage, score_ctx = context(options, "score")
    path = ctx.work_dir / "model.json"
    model = json.loads(path.read_text())
    if change == "rubric":
        stage.config = stage.config.model_copy(
            update={"min_scores": {"correctness": 2}}
        )
    elif change == "embedding":
        score_ctx = replace(
            score_ctx,
            config=score_ctx.config.model_copy(
                update={
                    "embeddings": score_ctx.config.embeddings.model_copy(
                        update={"model": "different"}
                    )
                }
            ),
        )
    else:
        model[
            {
                "model": "version",
                "dimension": "dimension",
                "privacy": "privacy_fingerprint",
            }[change]
        ] = {"model": 99, "dimension": 9, "privacy": "changed"}[change]
        path.write_text(json.dumps(model))
    with pytest.raises(ValueError):
        stage.run(records, score_ctx)
    assert not calls


def test_training_resume_rejects_changed_external_labels(setup):
    _, config, records, _, stage, ctx, _ = train(setup)
    stage.validate_resume(replace(ctx, resume=True))
    OutputWriter().write_parquet(records[:-1], Path(config["training_paths"][0]))
    with pytest.raises(ValueError, match="new run ID"):
        stage.validate_resume(replace(ctx, resume=True))


def test_training_and_holdout_keep_shared_lineage_together(setup):
    context, config, records, _ = setup
    # Different source identifiers must not split the same root across partitions.
    records[0] = records[0].model_copy(update={"lineage": records[2].lineage})
    OutputWriter().write_parquet(records, Path(config["training_paths"][0]))
    stage, ctx = context(config)
    stage.run([], ctx)
    model = json.loads((ctx.work_dir / "model.json").read_text())
    assert set(model["train_roots"]).isdisjoint(model["holdout_roots"])
