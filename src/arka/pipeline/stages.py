from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from arka.pipeline.models import StageContext
from arka.records.models import Record

if TYPE_CHECKING:
    from pathlib import Path

    from arka.config.models import PipelineStageConfig, ResolvedConfig
    from arka.records.models import ConversationRecord


class Stage(ABC):
    name: str
    stage_action: str = "transformed"

    @abstractmethod
    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        """Transform records for a pipeline stage."""

    @classmethod
    def from_config(
        cls,
        config: PipelineStageConfig,
        project_root: Path,
        resolved_config: ResolvedConfig,
    ) -> Stage | list[Stage]:
        import inspect

        sig = inspect.signature(cls.__init__)
        kwargs = {}
        if "config" in sig.parameters:
            kwargs["config"] = config
        if "project_root" in sig.parameters:
            kwargs["project_root"] = project_root
        return cls(**kwargs)


class BaseFilterStage(Stage):
    config_type: str
    stage_action: str = "filtered"

    def __init__(self, config: Any | None = None) -> None:
        self.config = config

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        from arka.records.models import ConversationRecord

        cfg = self.config
        if cfg is None and ctx.config is not None:
            if hasattr(ctx.config, "filters") and ctx.config.filters is not None:
                cfg = ctx.config.filters.get_stage_config(self.config_type)
            if cfg is None and hasattr(ctx.config, "get_stage_config"):
                cfg = ctx.config.get_stage_config(self.config_type)

        if cfg is None:
            return records

        if not self._is_active(cfg):
            return records

        kept: list[Record] = []
        dropped: list[Record] = []
        drop_reasons: dict[str, int] = {}

        for record in records:
            if not isinstance(record, ConversationRecord):
                kept.append(record)
                continue

            check_result = self._check_record(record, cfg)
            if check_result is None:
                kept.append(record)
            else:
                reason = check_result[0]
                details = check_result[1] if len(check_result) > 1 else None
                dropped.append(record.dropped_by(self.name, reason, details))
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

        self._write_artifacts(ctx, len(records), len(kept), dropped, drop_reasons)
        return kept

    def _is_active(self, config: Any) -> bool:
        return True

    def _check_record(
        self, record: ConversationRecord, config: Any
    ) -> tuple[str] | tuple[str, str] | None:
        """Return None if record is kept, or a tuple of (reason, [details]) if dropped."""
        raise NotImplementedError

    def _write_artifacts(
        self,
        ctx: StageContext,
        count_in: int,
        count_out: int,
        dropped: list[Record],
        drop_reasons: dict[str, int],
    ) -> None:
        from arka.pipeline.artifacts import StageArtifacts, StageReport

        StageArtifacts(ctx).write(
            report=StageReport(
                stage=self.name,
                count_in=count_in,
                count_out=count_out,
                dropped_count=len(dropped),
                drop_reasons=drop_reasons,
            ),
            # Preserve historical behavior: skip writing dropped.parquet when empty
            dropped=dropped if dropped else None,
        )
