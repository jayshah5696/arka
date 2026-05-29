from __future__ import annotations

import threading
from typing import Any

from openai import OpenAI

from arka.config.models import LLMConfig

_client_cache: dict[tuple[Any, ...], OpenAI] = {}
_cache_lock = threading.Lock()


def build_openai_client(config: LLMConfig) -> OpenAI:
    default_headers: dict[str, str] = {}
    if config.openai_compatible is not None:
        if config.openai_compatible.referer is not None:
            default_headers["HTTP-Referer"] = str(config.openai_compatible.referer)
        if config.openai_compatible.title is not None:
            default_headers["X-Title"] = config.openai_compatible.title

    headers_tuple = tuple(sorted(default_headers.items())) if default_headers else None
    cache_key = (
        config.api_key.get_secret_value(),
        str(config.base_url),
        config.timeout_seconds,
        headers_tuple,
    )

    with _cache_lock:
        if cache_key in _client_cache:
            return _client_cache[cache_key]

        # PERF: Connection pooling for OpenAI client across batch calls to avoid TLS handshake overhead for every stage processing
        client = OpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=str(config.base_url),
            timeout=config.timeout_seconds,
            max_retries=0,
            default_headers=default_headers or None,
        )
        _client_cache[cache_key] = client
        return client
