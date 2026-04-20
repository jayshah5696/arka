from __future__ import annotations

import json

import numpy as np
import pytest

from arka.config.models import CanaryFilterConfig, SemanticSimilarityFilterConfig
from arka.pipeline.filter_stages import CanaryFilterStage, SemanticSimilarityFilterStage
from arka.pipeline.models import StageContext
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


def _record(
    id: str, instruction: str, response: str, source_type: str = "generated"
) -> ConversationRecord:
    return ConversationRecord(
        id=id,
        content_hash="hash",
        source=RecordSource(type=source_type),
        lineage=RecordLineage(root_id=id, parent_ids=[]),
        payload=ConversationPayload(instruction=instruction, response=response),
        scores=RecordScores(),
        config_hash="cfg",
        created_at="2026-01-01T00:00:00Z",
    )


@pytest.fixture()
def ctx(tmp_path, _base_config):
    return StageContext(
        run_id="test",
        stage_name="test_stage",
        work_dir=tmp_path / "stage_dir",
        config=_base_config,
        executor_mode="threadpool",
        max_workers=1,
    )


@pytest.fixture()
def _base_config():
    from unittest.mock import MagicMock

    config = MagicMock()
    config.filters.canary = CanaryFilterConfig(enabled=False)
    config.filters.semantic_similarity = SemanticSimilarityFilterConfig(enabled=False)
    return config


# ── Canary filter ──────────────────────────────────────────


def test_canary_filter_disabled_returns_all(ctx) -> None:
    ctx.config.filters.canary = CanaryFilterConfig(enabled=False)
    records = [_record("1", "hello", "world")]
    assert CanaryFilterStage().run(records, ctx) == records


def test_canary_filter_drops_matching_phrase(ctx) -> None:
    ctx.config.filters.canary = CanaryFilterConfig(enabled=True, phrases=["SECRET"])
    records = [
        _record("1", "hello", "world"),
        _record("2", "leak", "this is SECRET data"),
    ]
    result = CanaryFilterStage().run(records, ctx)
    assert len(result) == 1
    assert result[0].id == "1"

    # Check artifacts
    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["dropped_count"] == 1
    assert stats["drop_reasons"]["canary_leak"] == 1


def test_canary_filter_no_match_keeps_all(ctx) -> None:
    ctx.config.filters.canary = CanaryFilterConfig(enabled=True, phrases=["NOPE"])
    records = [_record("1", "hello", "world")]
    assert len(CanaryFilterStage().run(records, ctx)) == 1


# ── Semantic similarity filter ─────────────────────────────


def test_semantic_similarity_filter_disabled_returns_all(ctx) -> None:
    ctx.config.filters.semantic_similarity = SemanticSimilarityFilterConfig(
        enabled=False
    )
    records = [_record("1", "hello", "world")]
    assert SemanticSimilarityFilterStage().run(records, ctx) == records


def test_semantic_similarity_filter_drops_high_similarity(ctx, monkeypatch) -> None:
    ctx.config.filters.semantic_similarity = SemanticSimilarityFilterConfig(
        enabled=True, threshold=0.9
    )

    seed = _record(
        "s1", "What is Python?", "A programming language.", source_type="seed"
    )
    gen_similar = _record(
        "g1", "What is Python?", "A programming language.", source_type="generated"
    )
    gen_different = _record(
        "g2", "What is Rust?", "A systems language.", source_type="generated"
    )

    # Mock embeddings: seed and gen_similar get identical vectors, gen_different gets orthogonal
    seed_vec = np.array([1.0, 0.0, 0.0])
    diff_vec = np.array([0.0, 1.0, 0.0])

    def fake_embed(self, *, config, texts):
        vecs = []
        for text in texts:
            if "Rust" in text:
                vecs.append(diff_vec)
            else:
                vecs.append(seed_vec)
        return np.array(vecs)

    # The stage imports PipelineRunner lazily; mock at the class level
    from arka.pipeline.runner import PipelineRunner

    monkeypatch.setattr(PipelineRunner, "_embed_texts", fake_embed)

    records = [seed, gen_similar, gen_different]
    result = SemanticSimilarityFilterStage().run(records, ctx)

    # seed always kept, gen_similar dropped (cosine=1.0 > 0.9), gen_different kept
    result_ids = {r.id for r in result}
    assert "s1" in result_ids
    assert "g2" in result_ids
    assert "g1" not in result_ids
