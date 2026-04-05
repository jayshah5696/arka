from __future__ import annotations

from pathlib import Path

import pytest

from arka.config.models import ResolvedConfig
from arka.pipeline.cheap_filters import LanguageFilterStage, LengthFilterStage
from arka.pipeline.dedup_stages import ExactDedupStage, NearDedupStage
from arka.pipeline.evol_generator_stage import EvolInstructRoundStage
from arka.pipeline.filter_stages import LabelingQualityFilterStage
from arka.pipeline.generator_stages import PromptBasedGeneratorStage
from arka.pipeline.ifd_stage import IFDFilterStage
from arka.pipeline.source_stages import PDFSourceStage, SeedSourceStage
from arka.pipeline.stage_builder import StageBuilder
from arka.pipeline.transforms import NormalizeConversationStage


def _base_config(**overrides) -> ResolvedConfig:
    data = {
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
        "dedup": {"exact": {"enabled": False}, "near": {"enabled": False}},
        "filters": {"target_count": 2},
        "embeddings": {"provider": "huggingface", "model": "all-MiniLM-L6-v2"},
        "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
        **overrides,
    }
    return ResolvedConfig(**data)


def test_seeds_without_labeling_builds_source_normalize_and_generator(
    tmp_path: Path,
) -> None:
    config = _base_config()
    stages = StageBuilder(config=config, project_root=tmp_path).build()

    assert len(stages) == 3
    assert isinstance(stages[0], SeedSourceStage)
    assert isinstance(stages[1], NormalizeConversationStage)
    assert isinstance(stages[2], PromptBasedGeneratorStage)


def test_seeds_with_labeling_appends_quality_filter(tmp_path: Path) -> None:
    config = _base_config(
        filters={
            "target_count": 2,
            "labeling_engine": {
                "enabled": True,
                "rubric_path": "rubrics/quality.yaml",
                "min_overall_score": 3.0,
            },
        }
    )
    stages = StageBuilder(config=config, project_root=tmp_path).build()

    assert len(stages) == 4
    assert isinstance(stages[0], SeedSourceStage)
    assert isinstance(stages[1], NormalizeConversationStage)
    assert isinstance(stages[2], PromptBasedGeneratorStage)
    assert isinstance(stages[3], LabelingQualityFilterStage)


def test_labeling_disabled_skips_quality_filter(tmp_path: Path) -> None:
    config = _base_config(
        filters={
            "target_count": 2,
            "labeling_engine": {
                "enabled": False,
                "rubric_path": "rubrics/quality.yaml",
            },
        }
    )
    stages = StageBuilder(config=config, project_root=tmp_path).build()

    assert len(stages) == 3
    assert any(isinstance(s, PromptBasedGeneratorStage) for s in stages)
    assert not any(isinstance(s, LabelingQualityFilterStage) for s in stages)


def test_pdf_source_skips_normalize_and_uses_pdf_source_stage(tmp_path: Path) -> None:
    config = _base_config(
        data_source={
            "type": "pdf",
            "path": "./source.pdf",
            "chunk_strategy": "fixed",
            "chunk_size_chars": 3000,
            "chunk_overlap_chars": 300,
        }
    )
    stages = StageBuilder(config=config, project_root=tmp_path).build()

    assert len(stages) == 2
    assert isinstance(stages[0], PDFSourceStage)
    assert isinstance(stages[1], PromptBasedGeneratorStage)
    assert not any(isinstance(stage, NormalizeConversationStage) for stage in stages)


def test_unsupported_data_source_type_raises(tmp_path: Path) -> None:
    config = _base_config(data_source={"type": "unknown_type"})

    with pytest.raises(ValueError, match="Unsupported data_source.type"):
        StageBuilder(config=config, project_root=tmp_path).build()


def test_unsupported_generator_type_raises(tmp_path: Path) -> None:
    config = _base_config(
        generator={
            "type": "unknown_type",
            "target_count": 2,
            "generation_multiplier": 1,
        }
    )

    with pytest.raises(ValueError, match="Unsupported generator.type"):
        StageBuilder(config=config, project_root=tmp_path).build()


def test_project_root_propagated_to_stages(tmp_path: Path) -> None:
    config = _base_config(
        filters={
            "target_count": 2,
            "labeling_engine": {"enabled": True, "rubric_path": "rubric.yaml"},
        }
    )
    stages = StageBuilder(config=config, project_root=tmp_path).build()

    seed_stage = stages[0]
    assert isinstance(seed_stage, SeedSourceStage)
    assert seed_stage.project_root == tmp_path

    generator_stage = stages[2]
    assert isinstance(generator_stage, PromptBasedGeneratorStage)
    assert generator_stage._project_root == tmp_path
    assert generator_stage._checkpoint is not None

    filter_stage = stages[3]
    assert isinstance(filter_stage, LabelingQualityFilterStage)
    assert filter_stage.project_root == tmp_path


def test_exact_dedup_included_when_enabled(tmp_path: Path) -> None:
    config = _base_config(dedup={"exact": {"enabled": True}})
    stages = StageBuilder(config=config, project_root=tmp_path).build()

    assert len(stages) == 4
    assert isinstance(stages[2], PromptBasedGeneratorStage)
    assert isinstance(stages[3], ExactDedupStage)


def test_near_dedup_included_when_enabled(tmp_path: Path) -> None:
    config = _base_config(
        dedup={"exact": {"enabled": False}, "near": {"enabled": True}}
    )
    stages = StageBuilder(config=config, project_root=tmp_path).build()

    assert len(stages) == 4
    assert isinstance(stages[2], PromptBasedGeneratorStage)
    assert isinstance(stages[3], NearDedupStage)


def test_length_filter_included_when_enabled(tmp_path: Path) -> None:
    config = _base_config(filters={"target_count": 2, "length": {"enabled": True}})
    stages = StageBuilder(config=config, project_root=tmp_path).build()

    assert len(stages) == 4
    assert isinstance(stages[2], PromptBasedGeneratorStage)
    assert isinstance(stages[3], LengthFilterStage)


def test_language_filter_included_when_enabled(tmp_path: Path) -> None:
    config = _base_config(filters={"target_count": 2, "language": {"enabled": True}})
    stages = StageBuilder(config=config, project_root=tmp_path).build()

    assert len(stages) == 4
    assert isinstance(stages[2], PromptBasedGeneratorStage)
    assert isinstance(stages[3], LanguageFilterStage)


def test_evol_instruct_builds_one_stage_per_round_and_preserves_order(
    tmp_path: Path,
) -> None:
    config = _base_config(
        generator={
            "type": "evol_instruct",
            "target_count": 2,
            "generation_multiplier": 1,
            "rounds": 3,
            "branching_factor": 1,
            "operators": ["deepen"],
        },
        dedup={"exact": {"enabled": True}, "near": {"enabled": True}},
        filters={
            "target_count": 2,
            "length": {"enabled": True},
            "language": {"enabled": True},
            "labeling_engine": {"enabled": True, "rubric_path": "rubric.yaml"},
        },
    )
    stages = StageBuilder(config=config, project_root=tmp_path).build()

    assert len(stages) == 10
    assert isinstance(stages[0], SeedSourceStage)
    assert isinstance(stages[1], NormalizeConversationStage)
    assert all(isinstance(stages[index], EvolInstructRoundStage) for index in [2, 3, 4])
    assert stages[2].name == "03_evol_round_01"
    assert stages[3].name == "04_evol_round_02"
    assert stages[4].name == "05_evol_round_03"
    assert isinstance(stages[5], ExactDedupStage)
    assert isinstance(stages[6], NearDedupStage)
    assert isinstance(stages[7], LengthFilterStage)
    assert isinstance(stages[8], LanguageFilterStage)
    assert isinstance(stages[9], LabelingQualityFilterStage)


def test_ifd_enabled_inserts_stage_before_label_quality(tmp_path: Path) -> None:
    config = _base_config(
        filters={
            "target_count": 2,
            "length": {"enabled": True},
            "language": {"enabled": True},
            "ifd": {"enabled": True, "min_score": 0.2},
            "labeling_engine": {"enabled": True, "rubric_path": "rubric.yaml"},
        }
    )

    with pytest.raises(
        ValueError, match="IFD requires provider/model response-scoring capability"
    ):
        StageBuilder(config=config, project_root=tmp_path).build()


def test_all_filters_ordering_without_ifd(tmp_path: Path) -> None:
    config = _base_config(
        dedup={"exact": {"enabled": True}, "near": {"enabled": True}},
        filters={
            "target_count": 2,
            "length": {"enabled": True},
            "language": {"enabled": True},
            "labeling_engine": {
                "enabled": True,
                "rubric_path": "rubric.yaml",
            },
        },
    )
    stages = StageBuilder(config=config, project_root=tmp_path).build()

    assert len(stages) == 8
    assert isinstance(stages[0], SeedSourceStage)
    assert isinstance(stages[1], NormalizeConversationStage)
    assert isinstance(stages[2], PromptBasedGeneratorStage)
    assert isinstance(stages[3], ExactDedupStage)
    assert isinstance(stages[4], NearDedupStage)
    assert isinstance(stages[5], LengthFilterStage)
    assert isinstance(stages[6], LanguageFilterStage)
    assert isinstance(stages[7], LabelingQualityFilterStage)


def test_ifd_stage_position_when_capability_check_is_stubbed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "arka.pipeline.stage_builder.validate_ifd_capability", lambda ctx: None
    )
    config = _base_config(
        filters={
            "target_count": 2,
            "length": {"enabled": True},
            "language": {"enabled": True},
            "ifd": {"enabled": True, "min_score": 0.2},
            "labeling_engine": {
                "enabled": True,
                "rubric_path": "rubric.yaml",
            },
        },
    )
    stages = StageBuilder(config=config, project_root=tmp_path).build()

    assert len(stages) == 7
    assert isinstance(stages[0], SeedSourceStage)
    assert isinstance(stages[1], NormalizeConversationStage)
    assert isinstance(stages[2], PromptBasedGeneratorStage)
    assert isinstance(stages[3], LengthFilterStage)
    assert isinstance(stages[4], LanguageFilterStage)
    assert isinstance(stages[5], IFDFilterStage)
    assert isinstance(stages[6], LabelingQualityFilterStage)
