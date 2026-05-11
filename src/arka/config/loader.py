from __future__ import annotations

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
            data = yaml.safe_load(resolved_text) or {}
            return ResolvedConfig.model_validate(data)
        except ValidationError as exc:
            raise ConfigValidationError(self._format_validation_error(exc)) from exc
        except yaml.YAMLError as exc:
            raise ConfigValidationError(str(exc)) from exc

    def load_dict(self, data: dict[str, Any]) -> ResolvedConfig:
        try:
            return ResolvedConfig.model_validate(data)
        except ValidationError as exc:
            raise ConfigValidationError(self._format_validation_error(exc)) from exc

    @staticmethod
    def _format_validation_error(exc: ValidationError) -> str:
        # DX: Make pydantic errors more human-readable and actionable
        lines = ["Configuration is invalid:"]
        for error in exc.errors():
            path = ".".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            err_type = error["type"]
            if err_type == "missing":
                lines.append(f"  - Missing required field: '{path}'")
            elif err_type == "extra_forbidden":
                lines.append(f"  - Unknown field: '{path}' (this key is not allowed here)")
            else:
                lines.append(f"  - Invalid value for '{path}': {msg}")
        return "\n".join(lines)

    def _resolve_env_vars(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            env_var = match.group(1)
            value = os.getenv(env_var)
            if value is None:
                raise ConfigValidationError(f"Missing environment variable: {env_var}")
            return value

        return _ENV_VAR_PATTERN.sub(replace, text)
