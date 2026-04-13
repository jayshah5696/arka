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

    # DX: Formats Pydantic errors to clearly name missing/invalid keys instead of dumping raw validation internals
    def _format_validation_error(self, exc: ValidationError) -> str:
        messages = []
        for error in exc.errors():
            loc = ".".join(str(loc) for loc in error["loc"])
            messages.append(f"  - {loc}: {error['msg']}")
        return "Configuration is invalid:\n" + "\n".join(messages)

    def _resolve_env_vars(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            env_var = match.group(1)
            value = os.getenv(env_var)
            if value is None:
                raise ConfigValidationError(f"Missing environment variable: {env_var}")
            return value

        return _ENV_VAR_PATTERN.sub(replace, text)
