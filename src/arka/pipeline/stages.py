from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from arka.common.models import StrictModel
from arka.pipeline.models import StageContext
from arka.records.models import Record

T = TypeVar("T", bound=StrictModel)

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

    def get_stage_config(
        self,
        ctx: StageContext,
        config_cls: type[T],
        legacy_field: str | None = None,
        default_val: T | None = None,
    ) -> T | None:
        """Resolve a stage's configuration from context config or defaults."""
        # 1. Use explicit self.config if already populated and correct type
        current = getattr(self, "config", None)
        if current is not None and isinstance(current, config_cls):
            return current

        # 2. Try to resolve from context config
        if ctx.config is not None:
            # 2a. Check legacy config properties/namespaces first for compatibility
            if legacy_field is not None:
                legacy_cfg = None
                if hasattr(ctx.config, "filters"):
                    filters_obj = ctx.config.filters
                    if hasattr(filters_obj, "get_stage_config"):
                        legacy_cfg = filters_obj.get_stage_config(legacy_field)

                if legacy_cfg is None and hasattr(ctx.config, legacy_field):
                    legacy_cfg = getattr(ctx.config, legacy_field)

                if legacy_cfg is not None:
                    if isinstance(legacy_cfg, config_cls):
                        return legacy_cfg
                    if hasattr(legacy_cfg, "dict"):
                        dct = legacy_cfg.dict()
                        if "type" in dct:
                            t_map = {
                                "seeds": "seed_source",
                                "pdf": "pdf_source",
                                "prompt_based": "prompt_based_generator",
                                "transform": "transform_generator",
                                "evol_instruct": "evol_instruct_generator",
                                "taxonomy_prompt": "taxonomy_generator",
                            }
                            if dct["type"] in t_map:
                                dct["type"] = t_map[dct["type"]]
                        valid_fields = config_cls.model_fields.keys()
                        filtered_dct = {
                            k: v
                            for k, v in dct.items()
                            if k in valid_fields and v is not None
                        }
                        try:
                            return config_cls.model_validate(filtered_dct)
                        except Exception:
                            full_dct = {
                                f: filtered_dct.get(f, None)
                                for f in config_cls.model_fields.keys()
                            }
                            return config_cls.model_construct(**full_dct)

            # 2b. Loop over modern pipeline config list
            if hasattr(ctx.config, "pipeline"):
                for stage_cfg in ctx.config.pipeline:
                    if isinstance(stage_cfg, config_cls):
                        return stage_cfg

        # 3. Fallback to default
        if default_val is not None:
            return default_val
        if ctx.config is None:
            try:
                return config_cls()
            except Exception:
                return None
        return None


class BaseFilterStage(Stage):
    config_type: str
    config_class: type[Any]
    stage_action: str = "filtered"

    def __init__(self, config: Any | None = None) -> None:
        self.config = config

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
