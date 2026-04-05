from __future__ import annotations

from openai import OpenAI

from arka.config.models import LLMConfig


def build_openai_client(config: LLMConfig) -> OpenAI:
    default_headers: dict[str, str] = {}
    if config.openai_compatible is not None:
        if config.openai_compatible.referer is not None:
            default_headers["HTTP-Referer"] = str(config.openai_compatible.referer)
        if config.openai_compatible.title is not None:
            default_headers["X-Title"] = config.openai_compatible.title
    return OpenAI(
        api_key=config.api_key,
        base_url=str(config.base_url),
        timeout=config.timeout_seconds,
        max_retries=0,
        default_headers=default_headers or None,
    )

