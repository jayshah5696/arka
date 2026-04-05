from __future__ import annotations

from pathlib import Path

from arka.config.models import ResolvedConfig
from arka.pipeline.filter_stages import LabelingQualityFilterStage
from arka.pipeline.source_stages import SeedSourceStage
from arka.pipeline.stages import Stage
from arka.pipeline.transforms import NormalizeConversationStage


class StageBuilder:
    """Build the ordered list of pipeline stages from a resolved config."""

    def __init__(self, config: ResolvedConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root

    def build(self) -> list[Stage]:
        stages: list[Stage] = []
        stages.extend(self._source_stages())
        stages.extend(self._filter_stages())
        return stages

    def _source_stages(self) -> list[Stage]:
        if self.config.data_source.type == "seeds":
            return [
                SeedSourceStage(project_root=self.project_root),
                NormalizeConversationStage(),
            ]
        raise ValueError(
            f"Unsupported data_source.type: {self.config.data_source.type!r}"
        )

    def _filter_stages(self) -> list[Stage]:
        stages: list[Stage] = []
        if self.config.filters.labeling_engine.enabled:
            stages.append(LabelingQualityFilterStage(project_root=self.project_root))
        return stages
