from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arka.pipeline.stages import Stage


@dataclass(frozen=True)
class StageSpec:
    stage_tag: str
    config_cls: str
    stage_cls: str
    requires_decisions: bool | Callable[[Any], bool] = False
    requires_sequence_scoring: bool = False
    requires_embeddings: bool = False

    def get_stage_class(self) -> type[Stage]:
        module_path, class_name = self.stage_cls.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def check_requires_decisions(self, config: Any) -> bool:
        if callable(self.requires_decisions):
            return bool(self.requires_decisions(config))
        if self.requires_decisions:
            return True
        if getattr(config, "backend", None) in ("decision", "regex_then_decision"):
            return True
        if getattr(config, "low_confidence_backend", None) == "decision":
            return True
        if getattr(config, "operator_selection", None) == "decision":
            return True
        if getattr(getattr(config, "filter", None), "backend", None) == "decision":
            return True
        for field in (
            "best_of_n",
            "fidelity_check",
            "decision_filter",
            "decision_band",
            "paraphrase_check",
            "pairwise_check",
            "label_taxonomy",
        ):
            sub = getattr(config, field, None)
            if getattr(sub, "enabled", False):
                return True
        return False


_STAGE_SPECS: dict[str, StageSpec] = {
    "distilled_ridge": StageSpec(
        stage_tag="distilled_ridge",
        config_cls="arka.config.models.DistilledRidgeConfig",
        stage_cls="arka.pipeline.distilled_ridge.DistilledRidgeStage",
        requires_embeddings=True,
    ),
    "source_audit": StageSpec(
        stage_tag="source_audit",
        config_cls="arka.config.models.SourceAuditConfig",
        stage_cls="arka.pipeline.source_audit.SourceAuditStage",
    ),
    "distilled_quality_train": StageSpec(
        stage_tag="distilled_quality_train",
        config_cls="arka.config.models.DistilledQualityTrainConfig",
        stage_cls="arka.pipeline.distilled_quality.DistilledQualityTrainStage",
        requires_embeddings=True,
    ),
    "distilled_quality_score": StageSpec(
        stage_tag="distilled_quality_score",
        config_cls="arka.config.models.DistilledQualityScoreConfig",
        stage_cls="arka.pipeline.distilled_quality.DistilledQualityScoreStage",
        requires_embeddings=True,
    ),
    "quality_repair": StageSpec(
        stage_tag="quality_repair",
        config_cls="arka.config.models.QualityRepairConfig",
        stage_cls="arka.pipeline.quality_repair.QualityRepairStage",
    ),
    "decision_panel": StageSpec(
        stage_tag="decision_panel",
        config_cls="arka.config.models.DecisionPanelConfig",
        stage_cls="arka.pipeline.decision_panel.DecisionPanelStage",
        requires_decisions=True,
    ),
    "seed_gate": StageSpec(
        stage_tag="seed_gate",
        config_cls="arka.config.models.SeedGateConfig",
        stage_cls="arka.pipeline.decision_stages.SeedGateStage",
        requires_decisions=True,
    ),
    "seed_source": StageSpec(
        stage_tag="seed_source",
        config_cls="arka.config.models.SeedSourceConfig",
        stage_cls="arka.pipeline.source_stages.SeedSourceStage",
    ),
    "pdf_source": StageSpec(
        stage_tag="pdf_source",
        config_cls="arka.config.models.PDFSourceConfig",
        stage_cls="arka.pipeline.source_stages.PDFSourceStage",
    ),
    "normalize_conversation": StageSpec(
        stage_tag="normalize_conversation",
        config_cls="arka.config.models.NormalizeConversationConfig",
        stage_cls="arka.pipeline.transforms.NormalizeConversationStage",
    ),
    "prompt_based_generator": StageSpec(
        stage_tag="prompt_based_generator",
        config_cls="arka.config.models.PromptBasedGeneratorConfig",
        stage_cls="arka.pipeline.generator_stages.PromptBasedGeneratorStage",
    ),
    "transform_generator": StageSpec(
        stage_tag="transform_generator",
        config_cls="arka.config.models.TransformGeneratorConfig",
        stage_cls="arka.pipeline.generator_stages.TransformGeneratorStage",
    ),
    "evol_instruct_generator": StageSpec(
        stage_tag="evol_instruct_generator",
        config_cls="arka.config.models.EvolInstructGeneratorConfig",
        stage_cls="arka.pipeline.evol_generator_stage.EvolInstructRoundStage",
    ),
    "taxonomy_generator": StageSpec(
        stage_tag="taxonomy_generator",
        config_cls="arka.config.models.TaxonomyGeneratorConfig",
        stage_cls="arka.pipeline.taxonomy_generator.TaxonomyGeneratorStage",
    ),
    "exact": StageSpec(
        stage_tag="exact",
        config_cls="arka.config.models.ExactDedupConfig",
        stage_cls="arka.pipeline.dedup_stages.ExactDedupStage",
    ),
    "near": StageSpec(
        stage_tag="near",
        config_cls="arka.config.models.NearDedupConfig",
        stage_cls="arka.pipeline.dedup_stages.NearDedupStage",
    ),
    "length": StageSpec(
        stage_tag="length",
        config_cls="arka.config.models.LengthFilterConfig",
        stage_cls="arka.pipeline.cheap_filters.LengthFilterStage",
    ),
    "language": StageSpec(
        stage_tag="language",
        config_cls="arka.config.models.LanguageFilterConfig",
        stage_cls="arka.pipeline.cheap_filters.LanguageFilterStage",
    ),
    "repetition": StageSpec(
        stage_tag="repetition",
        config_cls="arka.config.models.RepetitionFilterConfig",
        stage_cls="arka.pipeline.cheap_filters.RepetitionFilterStage",
    ),
    "identity_claim": StageSpec(
        stage_tag="identity_claim",
        config_cls="arka.config.models.IdentityClaimFilterConfig",
        stage_cls="arka.pipeline.cheap_filters.IdentityClaimFilterStage",
    ),
    "artifact_filter": StageSpec(
        stage_tag="artifact_filter",
        config_cls="arka.config.models.ArtifactFilterConfig",
        stage_cls="arka.pipeline.artifact_filter.ArtifactFilterStage",
    ),
    "decontaminate": StageSpec(
        stage_tag="decontaminate",
        config_cls="arka.config.models.DecontaminateConfig",
        stage_cls="arka.pipeline.decontamination.DecontaminateStage",
    ),
    "reasoning_language_gate": StageSpec(
        stage_tag="reasoning_language_gate",
        config_cls="arka.config.models.ReasoningLanguageGateConfig",
        stage_cls="arka.pipeline.reasoning_language.ReasoningLanguageGateStage",
    ),
    "sentence_variance": StageSpec(
        stage_tag="sentence_variance",
        config_cls="arka.config.models.SentenceVarianceFilterConfig",
        stage_cls="arka.pipeline.cheap_filters.SentenceVarianceFilterStage",
    ),
    "ifd": StageSpec(
        stage_tag="ifd",
        config_cls="arka.config.models.IFDFilterConfig",
        stage_cls="arka.pipeline.ifd_stage.IFDFilterStage",
        requires_sequence_scoring=True,
    ),
    "labeling_engine": StageSpec(
        stage_tag="labeling_engine",
        config_cls="arka.config.models.LabelingFilterConfig",
        stage_cls="arka.pipeline.filter_stages.LabelingQualityFilterStage",
    ),
    "labeling_score": StageSpec(
        stage_tag="labeling_score",
        config_cls="arka.config.models.LabelingScoreConfig",
        stage_cls="arka.pipeline.scoring_stages.LabelingScoreStage",
    ),
    "reward_model": StageSpec(
        stage_tag="reward_model",
        config_cls="arka.config.models.RewardModelFilterConfig",
        stage_cls="arka.pipeline.scoring_stages.RewardModelScoringStage",
    ),
    "pair_delta": StageSpec(
        stage_tag="pair_delta",
        config_cls="arka.config.models.PairDeltaFilterConfig",
        stage_cls="arka.pipeline.scoring_stages.PairDeltaFilterStage",
    ),
    "select": StageSpec(
        stage_tag="select",
        config_cls="arka.config.models.CompositeSelectConfig",
        stage_cls="arka.pipeline.scoring_stages.CompositeSelectStage",
    ),
    "semantic_similarity": StageSpec(
        stage_tag="semantic_similarity",
        config_cls="arka.config.models.SemanticSimilarityFilterConfig",
        stage_cls="arka.pipeline.filter_stages.SemanticSimilarityFilterStage",
        requires_embeddings=True,
    ),
    "canary": StageSpec(
        stage_tag="canary",
        config_cls="arka.config.models.CanaryFilterConfig",
        stage_cls="arka.pipeline.filter_stages.CanaryFilterStage",
    ),
    "double_critic": StageSpec(
        stage_tag="double_critic",
        config_cls="arka.config.models.DoubleCriticFilterConfig",
        stage_cls="arka.pipeline.double_critic_stage.DoubleCriticFilterStage",
    ),
    "complexity_elo": StageSpec(
        stage_tag="complexity_elo",
        config_cls="arka.config.models.ComplexityEloConfig",
        stage_cls="arka.pipeline.complexity_elo_stage.ComplexityEloScoringStage",
    ),
}

_STAGE_MAPPING: dict[str, str] = {
    tag: spec.stage_cls for tag, spec in _STAGE_SPECS.items()
}


def register_stage_spec(spec: StageSpec) -> None:
    _STAGE_SPECS[spec.stage_tag] = spec
    _STAGE_MAPPING[spec.stage_tag] = spec.stage_cls


def get_stage_spec(stage_tag: str) -> StageSpec | None:
    return _STAGE_SPECS.get(stage_tag)


def get_stage_class(config_type: str) -> type[Stage]:
    spec = get_stage_spec(config_type)
    if spec is not None:
        return spec.get_stage_class()
    class_path = _STAGE_MAPPING.get(config_type)
    if class_path is None:
        raise ValueError(f"Unknown pipeline stage type: {config_type}")

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
