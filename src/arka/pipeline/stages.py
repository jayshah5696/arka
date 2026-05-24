from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar, Any

from arka.pipeline.models import StageContext
from arka.records.models import Record
from arka.common.models import StrictModel

T = TypeVar("T", bound=StrictModel)


class Stage(ABC):
    name: str
    stage_action: str = "transformed"

    @abstractmethod
    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        """Transform records for a pipeline stage."""

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




