from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from arka.common.models import StrictModel
from arka.pipeline.models import StageContext, StageResult
from arka.records.models import Record

T = TypeVar("T", bound=StrictModel)

if TYPE_CHECKING:
    from pathlib import Path

    from arka.config.models import PipelineStageConfig, ResolvedConfig
    from arka.records.models import ConversationRecord


class Stage(ABC):
    name: str
    stage_action: str = "transformed"
    config: Any = None

    def validate_resume(
        self, ctx: StageContext, records: list[Record] | None = None
    ) -> None:
        """Validate external inputs before the runner reuses completed artifacts."""
        return None

    def __init__(self, config: Any = None, **kwargs: Any) -> None:
        self.config = config

    @abstractmethod
    def run(
        self, records: list[Record], ctx: StageContext
    ) -> list[Record] | StageResult:
        """Transform records for a pipeline stage."""

    @staticmethod
    def partition_conversations(
        records: list[Record],
    ) -> tuple[list[ConversationRecord], list[Record]]:
        """Partition records into conversation records and passthrough records."""
        return StageResult.partition_conversations(records)

    def get_stage_config(
        self,
        ctx: StageContext,
        config_cls: type[T],
        legacy_field: str | None = None,
        default_val: T | None = None,
    ) -> T | None:
        """Prefer the builder's typed config. Bare stages can use a resolved pipeline."""
        if isinstance(self.config, config_cls):
            return self.config
        if isinstance(self.config, dict):
            return config_cls.model_validate(self.config)
        if ctx.config is not None:
            # Direct Python callers may still supply explicit legacy overrides.
            overrides = vars(ctx.config)
            filters = overrides.get("filters")
            if filters is not None and legacy_field:
                override = filters.get_stage_config(legacy_field)
                if isinstance(override, config_cls):
                    return override
            if legacy_field == "data_source" and legacy_field in overrides:
                source = overrides[legacy_field]
                if not source.path:
                    return default_val
                return config_cls.model_validate(
                    {
                        key: getattr(source, key)
                        for key in config_cls.model_fields
                        if key != "type" and hasattr(source, key)
                    }
                )
            return next(
                (item for item in ctx.config.pipeline if isinstance(item, config_cls)),
                default_val,
            )
        if default_val is None and all(
            not field.is_required() for field in config_cls.model_fields.values()
        ):
            return config_cls()
        return default_val

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
        instance = cls(**kwargs)
        if isinstance(instance, Stage) and getattr(instance, "config", None) is None:
            instance.config = config
        elif isinstance(instance, list):
            for item in instance:
                if isinstance(item, Stage) and getattr(item, "config", None) is None:
                    item.config = config
        return instance


class BaseFilterStage(Stage):
    config_type: str
    config_class: type[Any]
    stage_action: str = "filtered"

    def __init__(self, config: Any | None = None) -> None:
        super().__init__(config=config)

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        from arka.records.models import ConversationRecord

        self.config = self.get_stage_config(
            ctx,
            config_cls=self.config_class,
            legacy_field=self.config_type,
        )
        cfg = self.config
        if cfg is None:
            return records

        if getattr(cfg, "backend", None) in (
            "decision",
            "regex_then_decision",
            "fasttext",
        ):
            from arka.pipeline.decision_stages import run_decision_stage

            return run_decision_stage(self, records, ctx, cfg)

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
