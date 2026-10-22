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
        for index, cfg in enumerate(self.config.pipeline):
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
            instances = built if isinstance(built, list) else [built]
            for part, stage in enumerate(instances, 1):
                if getattr(stage, "config", None) is None:
                    stage.config = cfg
                if self.config._stage_ids:
                    identity = self.config._stage_ids[index]
                    stage.name = (
                        identity if len(instances) == 1 else f"{identity}_{part:02d}"
                    )
                    stage.llm_config = self.config._stage_models[index]
                stages.append(stage)
        names = [stage.name for stage in stages]
        if self.config._stage_ids and len(names) != len(set(names)):
            raise ValueError("Stage names collide; use version 2 with unique stage IDs")
        return stages
