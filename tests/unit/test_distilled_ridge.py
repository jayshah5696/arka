"""Failure cases: leaked source/lineage/duplicate text across splits, missing or
mixed rubric labels, constant training targets, invalid features or coefficients,
incompatible embedding identity, changed model on resume, and lost record metadata.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from click.testing import CliRunner

from arka.cli import cli, train_scorer
from arka.config.loader import ConfigLoader
from arka.distillation import (
    DistilledModel,
    embedding_identity,
    features_for_records,
    record_text,
    train_model,
)
from arka.embeddings.embedder import Embedder
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


def records(count=30):
    return [
        ConversationRecord(
            id=str(i),
            content_hash=content_hash(
                ConversationPayload(instruction=f"Question {i}", response=f"Answer {i}")
            ),
            source=RecordSource(type="generated", doc_id=f"doc-{i // 2}"),
            lineage=RecordLineage(root_id=f"root-{i // 2}", parent_ids=[]),
            payload=ConversationPayload(
                instruction=f"Question {i}", response=f"Answer {i}"
            ),
            scores=RecordScores(
                quality=float(i),
                rubric_hash="rubric-a",
                judge_model="fixture-judge",
                ifd=0.7,
                quality_per_dim={"taxonomy_nodes": {"topic": ["test"]}},
            ),
            config_hash="cfg",
            created_at="2026-10-21T00:00:00Z",
        )
        for i in range(count)
    ]


def config_dict():
    return {
        "version": "1",
        "llm": {
            "provider": "openai",
            "model": "unused",
            "api_key": "test",
            "base_url": "https://api.openai.com/v1",
        },
        "executor": {"mode": "threadpool", "max_workers": 1},
        "pipeline": [],
        "output": {"format": "jsonl", "path": "output.jsonl"},
    }


def features(recs):
    return np.array([[float(r.id), 1.0] for r in recs])


def fitted(recs=None):
    recs = recs if recs is not None else records()
    cfg = ConfigLoader().load_dict(config_dict())
    return train_model(recs, features(recs), embedding_identity(cfg), alpha=0.01)


def test_train_split_roundtrip_and_ranking(tmp_path):
    recs = records()
    # These two otherwise separate groups share a source and must stay together.
    recs[2].source.doc_id = recs[0].source.doc_id
    model = fitted(recs)
    train_ids = set(model.training_ids)
    test_ids = set(model.validation_ids)
    assert train_ids and test_ids and not train_ids & test_ids
    assert train_ids | test_ids == {r.id for r in recs}
    for attribute in (lambda r: r.source.doc_id, lambda r: r.lineage.root_id):
        assert not {attribute(r) for r in recs if r.id in train_ids} & {
            attribute(r) for r in recs if r.id in test_ids
        }
    assert model.evaluation.mae < 0.01
    assert model.evaluation.spearman == pytest.approx(1.0)
    path = tmp_path / "model.json"
    path.write_text(model.model_dump_json())
    loaded = DistilledModel.model_validate_json(path.read_text())
    predictions = loaded.predict(features(recs))
    assert np.argsort(predictions).tolist() == list(range(len(recs)))
    assert "Question" not in path.read_text() and "Answer" not in path.read_text()
    assert record_text(recs[0]) != record_text(
        recs[0].model_copy(
            update={
                "payload": ConversationPayload(
                    instruction="Question 0", response="Changed"
                )
            }
        )
    )


@pytest.mark.parametrize(
    "failure",
    [
        "missing_label",
        "mixed_rubric",
        "one_group",
        "constant",
        "nan",
        "shape",
        "duplicate_id",
    ],
)
def test_training_rejects_invalid_inputs(failure):
    recs = records()
    vectors = features(recs)
    if failure == "missing_label":
        recs[0].scores.quality = None
    elif failure == "mixed_rubric":
        recs[0].scores.rubric_hash = "different"
    elif failure == "one_group":
        for r in recs:
            r.source.doc_id = "same"
    elif failure == "constant":
        for r in recs:
            r.scores.quality = 1.0
    elif failure == "nan":
        vectors[0, 0] = np.nan
    elif failure == "shape":
        vectors = vectors[:-1]
    elif failure == "duplicate_id":
        recs[1].id = recs[0].id
    with pytest.raises(ValueError):
        train_model(
            recs, vectors, embedding_identity(ConfigLoader().load_dict(config_dict()))
        )


def test_identical_text_cannot_leak_into_holdout():
    recs = records()
    for r in recs:
        r.payload = recs[0].payload
        r.content_hash = content_hash(r.payload)
    with pytest.raises(ValueError, match="group"):
        fitted(recs)


@pytest.mark.parametrize(
    "change", [{"version": 99}, {"coefficients": [float("inf")]}, {"coefficients": []}]
)
def test_invalid_artifact_rejected(change):
    data = fitted().model_dump()
    data.update(change)
    with pytest.raises(ValueError):
        DistilledModel.model_validate(data)


def test_prediction_validates_features():
    model = fitted()
    for vectors in (np.ones((2, 3)), np.array([[np.nan, 1]]), None):
        with pytest.raises(ValueError):
            model.predict(vectors)


def test_pair_features_use_separate_texts_and_reject_failed_embeddings(monkeypatch):
    recs = records(2)
    cfg = ConfigLoader().load_dict(config_dict())
    captured = []

    def embed(self, texts, **kwargs):
        captured.extend(texts)
        # Instructions, then responses. First pair aligns, second pair opposes.
        return np.array([[1.0, 0], [0, 1.0], [1.0, 0], [0, -1.0]])

    monkeypatch.setattr(Embedder, "embed", embed)
    vectors = features_for_records(recs, cfg, feature_format="pair_quality_v2")
    assert captured == [r.payload.instruction for r in recs] + [
        r.payload.response for r in recs
    ]
    assert vectors.shape == (2, 12)
    assert vectors[:, -1].tolist() == [1.0, -1.0]
    assert np.isfinite(vectors).all()
    for invalid in (None, np.ones((3, 2)), np.zeros((4, 2))):
        monkeypatch.setattr(
            Embedder, "embed", lambda *args, result=invalid, **kwargs: result
        )
        with pytest.raises(ValueError):
            features_for_records(recs, cfg, feature_format="pair_quality_v2")


def test_v2_scaling_uses_only_training_records_and_survives_loading():
    recs = records()
    vectors = np.zeros((len(recs), 12))
    vectors[:, 0] = np.arange(len(recs)) * 1000
    vectors[:, 1] = 7
    cfg = ConfigLoader().load_dict(config_dict())
    model = train_model(
        recs,
        vectors,
        embedding_identity(cfg),
        alpha=0.01,
        feature_format="pair_quality_v2",
    )
    training = [int(i) for i in model.training_ids]
    assert model.version == 2
    assert model.feature_mean == pytest.approx(vectors[training].mean(axis=0))
    assert model.feature_scale[1] == 1
    predicted = model.predict(vectors)
    assert np.mean(abs(predicted - np.arange(len(recs)))) < 0.01
    loaded = DistilledModel.model_validate_json(model.model_dump_json())
    assert loaded.predict(vectors) == pytest.approx(predicted)
    for field, value in [
        ("feature_scale", [0] * 12),
        ("feature_mean", [1]),
        ("feature_scale", None),
    ]:
        data = model.model_dump()
        data[field] = value
        with pytest.raises(ValueError):
            DistilledModel.model_validate(data)


def test_cli_training_scoring_selection_resume(tmp_path, monkeypatch):
    """Exercise real Parquet, CLI, model file, stage artifacts, selection and resume.

    Only the embedding provider boundary is replaced with deterministic vectors.
    """
    import yaml

    recs = records()
    OutputWriter().write_parquet(recs, tmp_path / "judged.parquet")
    (tmp_path / "seeds.jsonl").write_text(
        "".join(json.dumps(r.export_payload()) + "\n" for r in recs)
    )
    cfg = config_dict()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    calls = []

    def embed(self, texts, **kwargs):
        if texts[0].startswith("{"):
            calls.append(texts)
        return np.array(
            [
                [
                    float(
                        (
                            json.loads(t)["instruction"] if t.startswith("{") else t
                        ).split()[-1]
                    ),
                    1.0,
                ]
                for t in texts
            ]
        )

    monkeypatch.setattr(Embedder, "embed", embed)
    runner = CliRunner()
    result = runner.invoke(
        train_scorer,
        [
            "--config",
            str(config_path),
            "--records",
            str(tmp_path / "judged.parquet"),
            "--output",
            str(tmp_path / "model.json"),
            "--features",
            "instruction_response_json_v1",
        ],
    )
    assert result.exit_code == 0, result.output
    digest = hashlib.sha256((tmp_path / "model.json").read_bytes()).hexdigest()
    assert digest in result.output
    cfg["pipeline"] = [
        {"type": "seed_source", "path": "seeds.jsonl"},
        {"type": "normalize_conversation"},
        {
            "type": "distilled_ridge",
            "model_path": "model.json",
            "model_sha256": digest,
        },
        {"type": "select", "target_count": 3, "weights": {"distilled_ridge": 1.0}},
    ]
    config_path.write_text(yaml.safe_dump(cfg))
    result = runner.invoke(cli, ["--config", str(config_path), "--run-id", "proof"])
    assert result.exit_code == 0, result.output
    scored_path = tmp_path / "runs/proof/stages/02s_distilled_ridge/data.parquet"
    scored = OutputWriter().read_parquet(scored_path)
    assert len(scored) == len(recs)
    assert [r.payload.instruction for r in scored] == [
        r.payload.instruction for r in recs
    ]
    assert all(
        r.scores.distilled_ridge_prediction.model_sha256 == digest for r in scored
    )
    assert all(r.scores.distilled_ridge_prediction.input_hash for r in scored)
    output = [
        json.loads(line)
        for line in (tmp_path / "output.jsonl").read_text().splitlines()
    ]
    assert [r["instruction"] for r in output] == [
        "Question 29",
        "Question 28",
        "Question 27",
    ]
    count = len(calls)
    result = runner.invoke(
        cli, ["--config", str(config_path), "--run-id", "proof", "--resume"]
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == count
    # Changing the pin must not allow earlier prediction artifacts to resume.
    model = DistilledModel.model_validate_json((tmp_path / "model.json").read_text())
    model.intercept += 1
    (tmp_path / "model.json").write_text(model.model_dump_json())
    result = runner.invoke(
        cli, ["--config", str(config_path), "--run-id", "proof", "--resume"]
    )
    assert result.exit_code != 0
    cfg["pipeline"][2]["model_sha256"] = hashlib.sha256(
        (tmp_path / "model.json").read_bytes()
    ).hexdigest()
    config_path.write_text(yaml.safe_dump(cfg))
    result = runner.invoke(
        cli, ["--config", str(config_path), "--run-id", "proof", "--resume"]
    )
    assert result.exit_code != 0
    assert "model" in result.output.lower()


def test_stage_preserves_judge_scores_and_rejects_embedding_mismatch(
    tmp_path, monkeypatch
):
    from arka.pipeline.models import StageContext

    recs = records()
    model = fitted(recs)
    path = tmp_path / "model.json"
    path.write_text(model.model_dump_json())
    cfg = config_dict()
    cfg["pipeline"] = [
        {
            "type": "distilled_ridge",
            "model_path": "model.json",
            "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    ]
    resolved = ConfigLoader().load_dict(cfg)
    stage = StageBuilder(resolved, tmp_path).build()[0]
    monkeypatch.setattr(Embedder, "embed", lambda *args, **kwargs: features(recs))
    ctx = StageContext(
        run_id="test",
        stage_name=stage.name,
        work_dir=tmp_path / "stage",
        config=resolved,
        executor_mode="threadpool",
        max_workers=1,
    )
    out = stage.run(recs, ctx)
    assert [r.id for r in out] == [r.id for r in recs]
    for old, new in zip(recs, out, strict=True):
        assert new.scores.quality == old.scores.quality
        assert new.scores.quality_per_dim == old.scores.quality_per_dim
        assert new.scores.ifd == old.scores.ifd
        assert old.scores.distilled_ridge is None
    from arka.decisions.scores import score_value

    assert score_value(out[0], "distilled_ridge") == out[0].scores.distilled_ridge
    out[0].payload = ConversationPayload(instruction="Changed", response="Changed")
    with pytest.raises(ValueError, match="stale"):
        score_value(out[0], "distilled_ridge")
    out[1].payload.turns = [{"role": "user", "content": "A later turn"}]
    with pytest.raises(ValueError, match="stale"):
        score_value(out[1], "distilled_ridge")
    cfg["embeddings"] = {"model": "different"}
    with pytest.raises(ValueError, match="embedding"):
        StageBuilder(ConfigLoader().load_dict(cfg), tmp_path).build()


def test_private_training_rejects_text_before_embedding(tmp_path, monkeypatch):
    import yaml

    recs = records()
    recs[0].payload.response = "Contact person@example.com"
    recs[0].content_hash = content_hash(recs[0].payload)
    OutputWriter().write_parquet(recs, tmp_path / "judged.parquet")
    cfg = config_dict()
    cfg["pii"] = {"backend": "regex", "languages": ["en"], "entities": ["email"]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))

    def forbidden(*args, **kwargs):
        pytest.fail("Unsafe training text reached embedding provider")

    monkeypatch.setattr(Embedder, "embed", forbidden)
    result = CliRunner().invoke(
        train_scorer,
        [
            "--config",
            str(path),
            "--records",
            str(tmp_path / "judged.parquet"),
            "--output",
            str(tmp_path / "model.json"),
        ],
    )
    assert result.exit_code != 0
    assert "privacy" in result.output.lower()
    assert not (tmp_path / "model.json").exists()


def test_ridge_rejects_model_trained_under_a_different_privacy_policy(tmp_path):
    from arka.pipeline.models import StageContext
    from arka.privacy.runtime import build_policy

    path = tmp_path / "model.json"
    path.write_text(fitted().model_dump_json())
    cfg = config_dict()
    cfg["pii"] = {"backend": "regex", "languages": ["en"], "entities": ["email"]}
    cfg["pipeline"] = [
        {
            "type": "distilled_ridge",
            "model_path": "model.json",
            "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    ]
    resolved = ConfigLoader().load_dict(cfg)
    stage = StageBuilder(resolved, tmp_path).build()[0]
    ctx = StageContext(
        run_id="private",
        stage_name=stage.name,
        work_dir=tmp_path / "private",
        config=resolved,
        executor_mode="threadpool",
        max_workers=1,
        privacy=build_policy(resolved.pii, tmp_path),
    )
    with pytest.raises(ValueError, match="privacy"):
        stage.validate_resume(ctx)
