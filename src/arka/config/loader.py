from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from arka.config.models import ResolvedConfig

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


class ConfigValidationError(ValueError):
    """Raised when config loading or validation fails."""


class ConfigLoader:
    def load(self, path: Path) -> ResolvedConfig:
        raw_text = path.read_text()
        resolved_text = self._resolve_env_vars(raw_text)

        try:
            # DX: Use a named StringIO to give the yaml parser a filename for better line hints
            stream = io.StringIO(resolved_text)
            stream.name = str(path)
            data = yaml.safe_load(stream) or {}
            return ResolvedConfig.model_validate(data)
        except ValidationError as exc:
            raise ConfigValidationError(
                self._format_validation_error(exc, data)
            ) from exc
        except yaml.YAMLError as exc:
            raise ConfigValidationError(str(exc)) from exc

    def load_dict(self, data: dict[str, Any]) -> ResolvedConfig:
        try:
            return ResolvedConfig.model_validate(data)
        except ValidationError as exc:
            raise ConfigValidationError(
                self._format_validation_error(exc, data)
            ) from exc

    @staticmethod
    def _format_validation_error(
        exc: ValidationError, data: dict[str, Any] | None = None
    ) -> str:
        is_legacy = data is not None and "pipeline" not in data

        legacy_missing = []
        if is_legacy:
            for field in [
                "llm",
                "executor",
                "data_source",
                "generator",
                "filters",
                "output",
            ]:
                if field not in data:
                    legacy_missing.append((field, "Field required"))

        lines = ["Configuration is invalid:"]
        reported = set()

        for error in exc.errors():
            loc = error["loc"]
            msg = error["msg"]

            if is_legacy and loc and loc[0] == "pipeline" and len(loc) > 1:
                idx = loc[1]
                has_ds = "data_source" in data
                ds_type = data.get("data_source", {}).get("type") if has_ds else None
                has_normalize = has_ds and ds_type == "seeds"
                has_gen = "generator" in data
                dedup_stages = data.get("dedup", [])
                if not isinstance(dedup_stages, list):
                    dedup_stages = []
                num_dedup = len(dedup_stages)

                ds_idx = 0 if has_ds else -1
                norm_idx = 1 if (has_ds and has_normalize) else -1

                gen_idx = -1
                if has_gen:
                    gen_idx = 0
                    if has_ds:
                        gen_idx += 1
                        if has_normalize:
                            gen_idx += 1

                dedup_start = (
                    gen_idx + 1
                    if has_gen
                    else (
                        norm_idx + 1
                        if norm_idx != -1
                        else (ds_idx + 1 if ds_idx != -1 else 0)
                    )
                )
                dedup_indices = list(range(dedup_start, dedup_start + num_dedup))

                filters_start = dedup_start + num_dedup

                legacy_field = ""
                sub_path = []
                if idx == ds_idx:
                    legacy_field = "data_source"
                elif idx == norm_idx:
                    legacy_field = "data_source"
                elif idx == gen_idx:
                    legacy_field = "generator"
                    for item in loc[2:]:
                        if item not in (
                            "evol_instruct_generator",
                            "prompt_based_generator",
                            "transform_generator",
                            "taxonomy_generator",
                        ):
                            sub_path.append(str(item))
                elif idx in dedup_indices:
                    dedup_idx = idx - dedup_start
                    legacy_field = f"dedup.{dedup_idx}"
                    for item in loc[2:]:
                        sub_path.append(str(item))
                elif idx >= filters_start:
                    legacy_field = "filters"
                    for item in loc[2:]:
                        if item not in ("stages",):
                            sub_path.append(str(item))

                if legacy_field:
                    path = ".".join([legacy_field] + sub_path)
                else:
                    path = ".".join(str(loc) for loc in loc)
            else:
                path = ".".join(str(loc) for loc in loc)

            err_type = error.get("type")
            if err_type == "missing":
                err_msg = f"Missing required field: '{path}'"
            elif err_type == "extra_forbidden":
                err_msg = f"Unknown field: '{path}' (this key is not allowed here)"
            else:
                err_msg = f"Invalid value for '{path}': {msg}"

            reported.add(path)
            lines.append(f"  - {err_msg}")

        if is_legacy:
            for field, _ in legacy_missing:
                if field not in reported:
                    lines.append(f"  - Missing required field: '{field}'")
                    reported.add(field)

        return "\n".join(lines)

    def _resolve_env_vars(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            env_var = match.group(1)
            value = os.getenv(env_var)
            if value is None:
                raise ConfigValidationError(f"Missing environment variable: {env_var}")
            return value

        return _ENV_VAR_PATTERN.sub(replace, text)
