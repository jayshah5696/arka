from __future__ import annotations

import warnings
from typing import Any


def _recursive_dump(val: Any, mode: str = "python") -> Any:
    if hasattr(val, "model_dump"):
        return val.model_dump(mode=mode)
    if hasattr(val, "dict"):
        return val.dict()
    if isinstance(val, dict):
        return {k: _recursive_dump(v, mode=mode) for k, v in val.items()}
    if isinstance(val, list):
        return [_recursive_dump(v, mode=mode) for v in val]
    return val


class LegacyConfigNamespace:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def model_copy(self, update: dict[str, Any] | None = None) -> LegacyConfigNamespace:
        import copy

        new_obj = copy.copy(self)
        if update:
            for k, v in update.items():
                setattr(new_obj, k, v)
        return new_obj

    def dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raw = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return _recursive_dump(raw, mode="python")

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        mode = kwargs.get("mode", "python")
        raw = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return _recursive_dump(raw, mode=mode)

    def get_stage_config(self, stage_type: str) -> Any:
        stages = getattr(self, "stages", [])
        for s in stages:
            if getattr(s, "type", None) == stage_type:
                return s
        return None

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.dict().items())
        return f"LegacyConfigNamespace({attrs})"


def migrate_old_config(data: Any, *, warn: bool = True) -> Any:
    """Translate legacy 4-section config (data_source, generator, dedup, filters) into pipeline format."""
    if not isinstance(data, dict):
        return data
    if "pipeline" in data:
        return data

    if warn:
        warnings.warn(
            "Legacy configuration format detected. Please migrate to the unified sequential pipeline config format. "
            "You can use the offline migration script `scripts/migrate_config.py` to automatically update your files.",
            DeprecationWarning,
            stacklevel=2,
        )

    pipeline: list[dict[str, Any]] = []

    # 1. data_source
    data_source = data.get("data_source")
    if data_source:
        ds_type = data_source.get("type")
        if ds_type not in ("seeds", "pdf"):
            raise ValueError(f"Unsupported data_source.type: {ds_type}")
        ds_cfg = {k: v for k, v in data_source.items() if v is not None}
        if ds_type == "seeds":
            ds_cfg["type"] = "seed_source"
            pipeline.append(ds_cfg)
            pipeline.append({"type": "normalize_conversation"})
        elif ds_type == "pdf":
            ds_cfg["type"] = "pdf_source"
            pipeline.append(ds_cfg)

    # 2. generator
    generator = data.get("generator")
    if generator:
        gen_type = generator.get("type")
        if gen_type not in (
            "prompt_based",
            "transform",
            "evol_instruct",
            "taxonomy_prompt",
        ):
            raise ValueError(f"Unsupported generator.type: {gen_type}")
        target_count = (
            generator.get("target_count")
            or (data.get("filters") or {}).get("target_count")
            or 100
        )

        g_cfg = {k: v for k, v in generator.items() if v is not None}

        if gen_type == "prompt_based":
            g_cfg.setdefault("target_count", target_count)
            g_cfg["type"] = "prompt_based_generator"
            pipeline.append(g_cfg)
        elif gen_type == "transform":
            g_cfg.pop("target_count", None)
            g_cfg["type"] = "transform_generator"
            pipeline.append(g_cfg)
        elif gen_type == "evol_instruct":
            g_cfg.setdefault("target_count", target_count)
            g_cfg["type"] = "evol_instruct_generator"
            pipeline.append(g_cfg)
        elif gen_type == "taxonomy_prompt":
            g_cfg.setdefault("target_count", target_count)
            g_cfg["type"] = "taxonomy_generator"
            pipeline.append(g_cfg)

    # 3. dedup
    dedup = data.get("dedup") or []
    for d in dedup:
        pipeline.append(d)

    # 4. filters
    filters = data.get("filters") or {}
    stages = filters.get("stages") or []
    for stage in stages:
        pipeline.append(stage)

    new_data = {
        k: v
        for k, v in data.items()
        if k not in ("data_source", "generator", "dedup", "filters")
    }
    new_data["pipeline"] = pipeline
    return new_data


def get_legacy_data_source(resolved: Any) -> Any:
    if "data_source" in resolved.__dict__:
        return resolved.__dict__["data_source"]
    for stage in getattr(resolved, "pipeline", []):
        if stage.type == "seed_source":
            return LegacyConfigNamespace(type="seeds", path=stage.path)
        elif stage.type == "pdf_source":
            return LegacyConfigNamespace(
                type="pdf",
                path=stage.path,
                chunk_strategy=getattr(stage, "chunk_strategy", "fixed"),
                chunk_size_chars=getattr(stage, "chunk_size_chars", 3000),
                chunk_overlap_chars=getattr(stage, "chunk_overlap_chars", 300),
                max_chunks=getattr(stage, "max_chunks", None),
            )
    return None


def get_legacy_generator(resolved: Any) -> Any:
    if "generator" in resolved.__dict__:
        return resolved.__dict__["generator"]
    for stage in getattr(resolved, "pipeline", []):
        if stage.type == "prompt_based_generator":
            return LegacyConfigNamespace(
                type="prompt_based",
                target_count=stage.target_count,
                generation_multiplier=stage.generation_multiplier,
                prompt_template=getattr(stage, "prompt_template", None),
                temperature=getattr(stage, "temperature", 0.7),
                max_tokens=getattr(stage, "max_tokens", 512),
                llm_override=getattr(stage, "llm_override", None),
            )
        elif stage.type == "transform_generator":
            return LegacyConfigNamespace(
                type="transform",
                target_count=stage.target_count,
                generation_multiplier=stage.generation_multiplier,
                prompt_template=getattr(stage, "prompt_template", None),
                system_prompt=getattr(stage, "system_prompt", None),
                temperature=getattr(stage, "temperature", 0.7),
                max_tokens=getattr(stage, "max_tokens", 512),
                input_field=getattr(stage, "input_field", "payload.instruction"),
                output_field=getattr(stage, "output_field", "payload.response"),
                preserve_original=getattr(stage, "preserve_original", False),
                llm_override=getattr(stage, "llm_override", None),
            )
        elif stage.type == "evol_instruct_generator":
            return LegacyConfigNamespace(
                type="evol_instruct",
                rounds=stage.rounds,
                branching_factor=getattr(stage, "branching_factor", 1),
                operators=getattr(stage, "operators", []),
                filter=getattr(stage, "filter", None),
                temperature=getattr(stage, "temperature", 0.7),
                max_tokens=getattr(stage, "max_tokens", 512),
                target_count=stage.target_count,
                prompt_template=getattr(stage, "prompt_template", None),
                llm_override=getattr(stage, "llm_override", None),
            )
        elif stage.type == "taxonomy_generator":
            return LegacyConfigNamespace(
                type="taxonomy_prompt",
                target_count=stage.target_count,
                taxonomy_path=getattr(stage, "taxonomy_path", None),
                temperature=getattr(stage, "temperature", 0.7),
                max_tokens=getattr(stage, "max_tokens", 512),
                llm_override=getattr(stage, "llm_override", None),
            )
    return None


def get_legacy_dedup(resolved: Any) -> list[Any]:
    if "dedup" in resolved.__dict__:
        return resolved.__dict__["dedup"]
    return [
        stage
        for stage in getattr(resolved, "pipeline", [])
        if stage.type in ("exact", "near")
    ]


def get_legacy_filters(resolved: Any) -> Any:
    if "filters" in resolved.__dict__:
        return resolved.__dict__["filters"]
    stages = [
        stage
        for stage in getattr(resolved, "pipeline", [])
        if stage.type
        not in (
            "seed_source",
            "pdf_source",
            "normalize_conversation",
            "prompt_based_generator",
            "transform_generator",
            "evol_instruct_generator",
            "taxonomy_generator",
            "exact",
            "near",
        )
    ]
    target_count = 100
    for stage in getattr(resolved, "pipeline", []):
        if hasattr(stage, "target_count") and stage.target_count is not None:
            target_count = stage.target_count
            break
    return LegacyConfigNamespace(target_count=target_count, stages=stages)


def inject_legacy_dict(
    resolved: Any, data: dict[str, Any], mode: str = "python"
) -> dict[str, Any]:
    if "data_source" not in data:
        try:
            ds = get_legacy_data_source(resolved)
            if ds is not None:
                data["data_source"] = _recursive_dump(ds, mode=mode)
        except Exception:
            pass
    if "generator" not in data:
        try:
            g = get_legacy_generator(resolved)
            if g is not None:
                data["generator"] = _recursive_dump(g, mode=mode)
        except Exception:
            pass
    if "dedup" not in data:
        try:
            d = get_legacy_dedup(resolved)
            if d is not None:
                data["dedup"] = [_recursive_dump(x, mode=mode) for x in d]
        except Exception:
            pass
    if "filters" not in data:
        try:
            f = get_legacy_filters(resolved)
            if f is not None:
                data["filters"] = _recursive_dump(f, mode=mode)
        except Exception:
            pass
    return data
