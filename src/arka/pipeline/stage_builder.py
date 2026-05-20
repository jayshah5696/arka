from __future__ import annotations

from pathlib import Path

from arka.config.models import (
    CanaryFilterConfig,
    ComplexityEloFilterConfig,
    CompositeSelectConfig,
    DoubleCriticFilterConfig,
    EvolInstructGeneratorConfig,
    ExactDedupConfig,
    IFDFilterConfig,
    LabelingFilterConfig,
    LanguageFilterConfig,
    LengthFilterConfig,
    NearDedupConfig,
    NormalizeConversationConfig,
    PairDeltaFilterConfig,
    PDFSourceConfig,
    PromptBasedGeneratorConfig,
    ResolvedConfig,
    RewardModelFilterConfig,
    SeedSourceConfig,
    SemanticSimilarityFilterConfig,
    SentenceVarianceFilterConfig,
    TaxonomyGeneratorConfig,
    TransformGeneratorConfig,
)
from arka.pipeline.cheap_filters import (
    LanguageFilterStage,
    LengthFilterStage,
    SentenceVarianceFilterStage,
)
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
from arka.pipeline.scoring_stages import (
    CompositeSelectStage,
    PairDeltaFilterStage,
    RewardModelScoringStage,
)
from arka.pipeline.source_stages import PDFSourceStage, SeedSourceStage
from arka.pipeline.stages import Stage
from arka.pipeline.taxonomy_generator import TaxonomyGeneratorStage
from arka.pipeline.transforms import NormalizeConversationStage


class StageBuilder:
    """Build the ordered list of pipeline stages from a resolved config."""

    def __init__(self, config: ResolvedConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root

    def build(self) -> list[Stage]:
        stages: list[Stage] = []
        for cfg in self.config.pipeline:
            # Source stages
            if isinstance(cfg, SeedSourceConfig):
                stages.append(
                    SeedSourceStage(config=cfg, project_root=self.project_root)
                )
            elif isinstance(cfg, PDFSourceConfig):
                stages.append(
                    PDFSourceStage(config=cfg, project_root=self.project_root)
                )
            elif isinstance(cfg, NormalizeConversationConfig):
                stages.append(NormalizeConversationStage())
            # Generator stages
            elif isinstance(cfg, PromptBasedGeneratorConfig):
                stages.append(
                    PromptBasedGeneratorStage(
                        config=cfg, project_root=self.project_root
                    )
                )
            elif isinstance(cfg, TransformGeneratorConfig):
                stages.append(
                    TransformGeneratorStage(config=cfg, project_root=self.project_root)
                )
            elif isinstance(cfg, EvolInstructGeneratorConfig):
                rounds = cfg.rounds or 0
                for round_number in range(1, rounds + 1):
                    stages.append(
                        EvolInstructRoundStage(
                            round_number=round_number,
                            config=cfg,
                            project_root=self.project_root,
                        )
                    )
            elif isinstance(cfg, TaxonomyGeneratorConfig):
                stages.append(
                    TaxonomyGeneratorStage(config=cfg, project_root=self.project_root)
                )
            # Dedup stages
            elif isinstance(cfg, ExactDedupConfig):
                stages.append(ExactDedupStage(config=cfg))
            elif isinstance(cfg, NearDedupConfig):
                stages.append(NearDedupStage(config=cfg))
            # Cheap filter stages
            elif isinstance(cfg, LengthFilterConfig):
                stages.append(LengthFilterStage(config=cfg))
            elif isinstance(cfg, LanguageFilterConfig):
                stages.append(LanguageFilterStage(config=cfg))
            elif isinstance(cfg, SentenceVarianceFilterConfig):
                stages.append(SentenceVarianceFilterStage(config=cfg))
            # Complex filter / scoring / selection stages
            elif isinstance(cfg, IFDFilterConfig):
                validate_ifd_capability(
                    cfg,
                    StageContext(
                        run_id="validation",
                        stage_name="02e_ifd_filter",
                        work_dir=self.project_root
                        / "runs"
                        / "validation"
                        / "stages"
                        / "02e_ifd_filter",
                        config=self.config,
                        executor_mode=self.config.executor.mode,
                        max_workers=self.config.executor.max_workers,
                    ),
                )
                stages.append(
                    IFDFilterStage(config=cfg, project_root=self.project_root)
                )
            elif isinstance(cfg, LabelingFilterConfig):
                stages.append(
                    LabelingQualityFilterStage(
                        config=cfg, project_root=self.project_root
                    )
                )
            elif isinstance(cfg, RewardModelFilterConfig):
                stages.append(RewardModelScoringStage(config=cfg))
            elif isinstance(cfg, PairDeltaFilterConfig):
                stages.append(PairDeltaFilterStage(config=cfg))
            elif isinstance(cfg, CompositeSelectConfig):
                stages.append(CompositeSelectStage(config=cfg))
            elif isinstance(cfg, SemanticSimilarityFilterConfig):
                stages.append(SemanticSimilarityFilterStage(config=cfg))
            elif isinstance(cfg, CanaryFilterConfig):
                stages.append(CanaryFilterStage(config=cfg))
            elif isinstance(cfg, DoubleCriticFilterConfig):
                stages.append(DoubleCriticFilterStage(config=cfg))
            elif isinstance(cfg, ComplexityEloFilterConfig):
                from arka.pipeline.complexity_elo_stage import ComplexityEloScoringStage

                stages.append(ComplexityEloScoringStage(config=cfg))
            else:
                raise ValueError(
                    f"Unknown pipeline stage config type: {type(cfg).__name__}"
                )
        return stages
