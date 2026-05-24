from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from arka.pipeline.registry import get_stage_class

if TYPE_CHECKING:
    from arka.config.models import ResolvedConfig
    from arka.pipeline.stages import Stage


class StageBuilder:
    """Build the ordered list of pipeline stages from a resolved config."""

    def __init__(self, config: ResolvedConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root

    def build(self) -> list[Stage]:
        stages: list[Stage] = []
        for cfg in self.config.pipeline:
            try:
                stage_cls = get_stage_class(cfg.type)
            except ValueError as e:
                raise ValueError(
                    f"Unknown pipeline stage config type: {type(cfg).__name__}"
                ) from e
            built = stage_cls.from_config(
                config=cfg,
                project_root=self.project_root,
                resolved_config=self.config,
            )
            if isinstance(built, list):
                stages.extend(built)
            else:
                stages.append(built)
        return stages
