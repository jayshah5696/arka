from __future__ import annotations

import httpx
from openai import OpenAI

from arka.config.models import LLMConfig

# PERF: Connection pooling for openai client across batch calls. Reuses the underlying httpx.Client instead of instantiating new connections per request, avoiding TLS handshake overhead on batch calls.
_SHARED_HTTP_CLIENT = httpx.Client()


def build_openai_client(config: LLMConfig) -> OpenAI:
    default_headers: dict[str, str] = {}
    if config.openai_compatible is not None:
        if config.openai_compatible.referer is not None:
            default_headers["HTTP-Referer"] = str(config.openai_compatible.referer)
        if config.openai_compatible.title is not None:
            default_headers["X-Title"] = config.openai_compatible.title
    return OpenAI(
        api_key=config.api_key.get_secret_value(),
        base_url=str(config.base_url),
        timeout=config.timeout_seconds,
        max_retries=0,
        default_headers=default_headers or None,
        http_client=_SHARED_HTTP_CLIENT,
    )
