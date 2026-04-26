from __future__ import annotations

from pathlib import Path

from arka.config.models import (
    CanaryFilterConfig,
    DoubleCriticFilterConfig,
    IFDFilterConfig,
    LabelingFilterConfig,
    LanguageFilterConfig,
    LengthFilterConfig,
    ResolvedConfig,
    SemanticSimilarityFilterConfig,
)
from arka.pipeline.cheap_filters import LanguageFilterStage, LengthFilterStage
from arka.pipeline.dedup_stages import ExactDedupStage, NearDedupStage
from arka.pipeline.double_critic_stage import DoubleCriticFilterStage
from arka.pipeline.evol_generator_stage import EvolInstructRoundStage
from arka.pipeline.filter_stages import (
    CanaryFilterStage,
    LabelingQualityFilterStage,
    SemanticSimilarityFilterStage,
    validate_ifd_capability,
)
from arka.pipeline.generator_stages import (
    PromptBasedGeneratorStage,
    TransformGeneratorStage,
)
from arka.pipeline.ifd_stage import IFDFilterStage
from arka.pipeline.models import StageContext
from arka.pipeline.source_stages import PDFSourceStage, SeedSourceStage
from arka.pipeline.stages import Stage
from arka.pipeline.taxonomy_generator import TaxonomyGeneratorStage
from arka.pipeline.transforms import NormalizeConversationStage

# Registry: config type → stage factory
_DEDUP_REGISTRY: dict[str, type[Stage]] = {
    "exact": ExactDedupStage,
    "near": NearDedupStage,
}


def _build_filter_stage(
    cfg: object, project_root: Path, config: ResolvedConfig
) -> Stage:
    """Instantiate a filter stage from its config object."""
    if isinstance(cfg, LengthFilterConfig):
        return LengthFilterStage()
    if isinstance(cfg, LanguageFilterConfig):
        return LanguageFilterStage()
    if isinstance(cfg, IFDFilterConfig):
        validate_ifd_capability(
            StageContext(
                run_id="validation",
                stage_name="02e_ifd_filter",
                work_dir=project_root
                / "runs"
                / "validation"
                / "stages"
                / "02e_ifd_filter",
                config=config,
                executor_mode=config.executor.mode,
                max_workers=config.executor.max_workers,
            )
        )
        return IFDFilterStage(project_root=project_root)
    if isinstance(cfg, CanaryFilterConfig):
        return CanaryFilterStage()
    if isinstance(cfg, SemanticSimilarityFilterConfig):
        return SemanticSimilarityFilterStage()
    if isinstance(cfg, LabelingFilterConfig):
        return LabelingQualityFilterStage(project_root=project_root)
    if isinstance(cfg, DoubleCriticFilterConfig):
        return DoubleCriticFilterStage()
    # SentenceVariance, RewardModel, PairDelta, CompositeSelect
    from arka.config.models import (
        CompositeSelectConfig,
        PairDeltaFilterConfig,
        RewardModelFilterConfig,
        SentenceVarianceFilterConfig,
    )
    from arka.pipeline.cheap_filters import SentenceVarianceFilterStage
    from arka.pipeline.scoring_stages import (
        CompositeSelectStage,
        PairDeltaFilterStage,
        RewardModelScoringStage,
    )

    if isinstance(cfg, SentenceVarianceFilterConfig):
        return SentenceVarianceFilterStage()
    if isinstance(cfg, RewardModelFilterConfig):
        return RewardModelScoringStage()
    if isinstance(cfg, PairDeltaFilterConfig):
        return PairDeltaFilterStage()
    if isinstance(cfg, CompositeSelectConfig):
        return CompositeSelectStage()
    raise ValueError(f"Unknown filter config type: {type(cfg).__name__}")


class StageBuilder:
    """Build the ordered list of pipeline stages from a resolved config."""

    def __init__(self, config: ResolvedConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root

    def build(self) -> list[Stage]:
        stages: list[Stage] = []
        stages.extend(self._source_stages())
        stages.extend(self._generator_stages())
        stages.extend(self._dedup_stages())
        stages.extend(self._filter_stages())
        return stages

    def _source_stages(self) -> list[Stage]:
        if self.config.data_source.type == "seeds":
            return [
                SeedSourceStage(project_root=self.project_root),
                NormalizeConversationStage(),
            ]
        if self.config.data_source.type == "pdf":
            return [PDFSourceStage(project_root=self.project_root)]
        raise ValueError(
            f"Unsupported data_source.type: {self.config.data_source.type!r}"
        )

    def _generator_stages(self) -> list[Stage]:
        if self.config.generator.type == "prompt_based":
            return [
                PromptBasedGeneratorStage(
                    project_root=self.project_root,
                )
            ]
        if self.config.generator.type == "transform":
            return [
                TransformGeneratorStage(
                    project_root=self.project_root,
                )
            ]
        if self.config.generator.type == "evol_instruct":
            rounds = self.config.generator.rounds or 0
            return [
                EvolInstructRoundStage(
                    round_number=round_number,
                    project_root=self.project_root,
                )
                for round_number in range(1, rounds + 1)
            ]
        if self.config.generator.type == "taxonomy_prompt":
            return [TaxonomyGeneratorStage(project_root=self.project_root)]
        raise ValueError(f"Unsupported generator.type: {self.config.generator.type!r}")

    def _dedup_stages(self) -> list[Stage]:
        return [_DEDUP_REGISTRY[cfg.type]() for cfg in self.config.dedup]

    def _filter_stages(self) -> list[Stage]:
        return [
            _build_filter_stage(cfg, self.project_root, self.config)
            for cfg in self.config.filters.stages
        ]
