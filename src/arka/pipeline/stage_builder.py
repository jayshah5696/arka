from __future__ import annotations

from pathlib import Path

from arka.config.models import ResolvedConfig
from arka.pipeline.cheap_filters import LanguageFilterStage, LengthFilterStage
from arka.pipeline.checkpoint import CheckpointManager
from arka.pipeline.dedup_stages import ExactDedupStage, NearDedupStage
from arka.pipeline.evol_generator_stage import EvolInstructRoundStage
from arka.pipeline.filter_stages import (
    LabelingQualityFilterStage,
    validate_ifd_capability,
)
from arka.pipeline.generator_stages import PromptBasedGeneratorStage
from arka.pipeline.ifd_stage import IFDFilterStage
from arka.pipeline.models import StageContext
from arka.pipeline.source_stages import PDFSourceStage, SeedSourceStage
from arka.pipeline.stages import Stage
from arka.pipeline.transforms import NormalizeConversationStage


class StageBuilder:
    """Build the ordered list of pipeline stages from a resolved config."""

    def __init__(self, config: ResolvedConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        self._checkpoint_manager = CheckpointManager(project_root / "state.db")

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
                    checkpoint_manager=self._checkpoint_manager,
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
        raise ValueError(f"Unsupported generator.type: {self.config.generator.type!r}")

    def _dedup_stages(self) -> list[Stage]:
        stages: list[Stage] = []
        if self.config.dedup.exact.enabled:
            stages.append(ExactDedupStage())
        if self.config.dedup.near.enabled:
            stages.append(NearDedupStage())
        return stages

    def _filter_stages(self) -> list[Stage]:
        stages: list[Stage] = []
        if self.config.filters.length.enabled:
            stages.append(LengthFilterStage())
        if self.config.filters.language.enabled:
            stages.append(LanguageFilterStage())
        if self.config.filters.ifd.enabled:
            validate_ifd_capability(
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
                )
            )
            stages.append(IFDFilterStage(project_root=self.project_root))
        if self.config.filters.labeling_engine.enabled:
            stages.append(LabelingQualityFilterStage(project_root=self.project_root))
        return stages
