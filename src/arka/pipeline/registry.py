from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arka.pipeline.stages import Stage


_STAGE_MAPPING: dict[str, str] = {
    "seed_source": "arka.pipeline.source_stages.SeedSourceStage",
    "pdf_source": "arka.pipeline.source_stages.PDFSourceStage",
    "normalize_conversation": "arka.pipeline.transforms.NormalizeConversationStage",
    "prompt_based_generator": "arka.pipeline.generator_stages.PromptBasedGeneratorStage",
    "transform_generator": "arka.pipeline.generator_stages.TransformGeneratorStage",
    "evol_instruct_generator": "arka.pipeline.evol_generator_stage.EvolInstructRoundStage",
    "taxonomy_generator": "arka.pipeline.taxonomy_generator.TaxonomyGeneratorStage",
    "exact": "arka.pipeline.dedup_stages.ExactDedupStage",
    "near": "arka.pipeline.dedup_stages.NearDedupStage",
    "length": "arka.pipeline.cheap_filters.LengthFilterStage",
    "language": "arka.pipeline.cheap_filters.LanguageFilterStage",
    "sentence_variance": "arka.pipeline.cheap_filters.SentenceVarianceFilterStage",
    "ifd": "arka.pipeline.ifd_stage.IFDFilterStage",
    "labeling_engine": "arka.pipeline.filter_stages.LabelingQualityFilterStage",
    "reward_model": "arka.pipeline.scoring_stages.RewardModelScoringStage",
    "pair_delta": "arka.pipeline.scoring_stages.PairDeltaFilterStage",
    "select": "arka.pipeline.scoring_stages.CompositeSelectStage",
    "semantic_similarity": "arka.pipeline.filter_stages.SemanticSimilarityFilterStage",
    "canary": "arka.pipeline.filter_stages.CanaryFilterStage",
    "double_critic": "arka.pipeline.double_critic_stage.DoubleCriticFilterStage",
    "complexity_elo": "arka.pipeline.complexity_elo_stage.ComplexityEloScoringStage",
}


def get_stage_class(config_type: str) -> type[Stage]:
    class_path = _STAGE_MAPPING.get(config_type)
    if class_path is None:
        raise ValueError(f"Unknown pipeline stage type: {config_type}")

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
