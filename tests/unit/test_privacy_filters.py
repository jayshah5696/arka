import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from arka.pipeline.filter_stages import CanaryFilterStage, SemanticSimilarityFilterStage
from arka.pipeline.models import StageContext
from arka.records.models import ConversationRecord, ConversationPayload, RecordLineage, RecordSource, RecordScores
from arka.config.models import ResolvedConfig, FiltersConfig, CanaryFilterConfig, SemanticSimilarityFilterConfig

def build_test_record(id: str, instruction: str, response: str, source_type: str) -> ConversationRecord:
    return ConversationRecord(
        id=id,
        content_hash="hash",
        source=RecordSource(type=source_type),
        lineage=RecordLineage(root_id=id, parent_ids=[]),
        payload=ConversationPayload(instruction=instruction, response=response),
        scores=RecordScores(),
        config_hash="config",
        created_at="2023-01-01T00:00:00Z"
    )

@pytest.fixture
def base_config():
    config_mock = MagicMock(spec=ResolvedConfig)
    config_mock.filters = MagicMock()
    config_mock.filters.canary = CanaryFilterConfig(enabled=False)
    config_mock.filters.semantic_similarity = SemanticSimilarityFilterConfig(enabled=False)
    return config_mock

@pytest.fixture
def mock_ctx(tmp_path, base_config):
    ctx = MagicMock(spec=StageContext)
    ctx.work_dir = tmp_path / "stage_dir"
    ctx.config = base_config
    return ctx


def test_canary_filter_stage_disabled(mock_ctx):
    stage = CanaryFilterStage()
    records = [build_test_record("1", "inst", "resp", "generated")]

    result = stage.run(records, mock_ctx)
    assert len(result) == 1


def test_canary_filter_stage_enabled_match(mock_ctx):
    mock_ctx.config.filters.canary = CanaryFilterConfig(enabled=True, phrases=["leak_me", "secret"])
    stage = CanaryFilterStage()

    records = [
        build_test_record("1", "Hello", "world", "generated"),
        build_test_record("2", "leak_me instruction", "clean response", "generated"),
        build_test_record("3", "clean inst", "a secret is here", "generated"),
    ]

    result = stage.run(records, mock_ctx)
    assert len(result) == 1
    assert result[0].id == "1"

    # Check stats
    stats_file = mock_ctx.work_dir / "stats.json"
    assert stats_file.exists()
    stats = json.loads(stats_file.read_text())
    assert stats["dropped_count"] == 2
    assert stats["drop_reasons"]["canary_leak"] == 2


def test_semantic_similarity_filter_disabled(mock_ctx):
    stage = SemanticSimilarityFilterStage()
    records = [build_test_record("1", "inst", "resp", "generated")]

    result = stage.run(records, mock_ctx)
    assert len(result) == 1

@patch.object(SemanticSimilarityFilterStage, "_embed_texts")
def test_semantic_similarity_filter_enabled_drop(mock_embed_texts, mock_ctx):
    mock_ctx.config.filters.semantic_similarity = SemanticSimilarityFilterConfig(enabled=True, threshold=0.9)
    stage = SemanticSimilarityFilterStage()

    records = [
        build_test_record("seed_1", "What is Python?", "A language.", "seed"),
        build_test_record("gen_1", "What is Python?", "A language.", "generated"), # very similar
        build_test_record("gen_2", "Different", "Thing", "generated"), # dissimilar
    ]

    # Mock embeddings to return specific vectors
    def mock_embed(config, texts):
        embeddings = []
        for text in texts:
            if "Python" in text:
                embeddings.append([1.0, 0.0])
            else:
                embeddings.append([0.0, 1.0])
        return np.array(embeddings)

    mock_embed_texts.side_effect = mock_embed

    result = stage.run(records, mock_ctx)

    # Expect gen_1 to be dropped, gen_2 and seed_1 kept
    assert len(result) == 2
    ids = [r.id for r in result]
    assert "seed_1" in ids
    assert "gen_2" in ids
    assert "gen_1" not in ids

    stats_file = mock_ctx.work_dir / "stats.json"
    assert stats_file.exists()
    stats = json.loads(stats_file.read_text())
    assert stats["dropped_count"] == 1
    assert stats["drop_reasons"]["high_semantic_similarity"] == 1


@patch.object(SemanticSimilarityFilterStage, "_embed_texts")
def test_semantic_similarity_filter_fail_closed(mock_embed_texts, mock_ctx):
    mock_ctx.config.filters.semantic_similarity = SemanticSimilarityFilterConfig(enabled=True, threshold=0.9)
    stage = SemanticSimilarityFilterStage()

    records = [
        build_test_record("seed_1", "A", "B", "seed"),
        build_test_record("gen_1", "C", "D", "generated"),
    ]

    # Mock to throw exception
    mock_embed_texts.side_effect = Exception("API error")

    with pytest.raises(RuntimeError, match="Semantic similarity embedding failed: API error"):
        stage.run(records, mock_ctx)
