from __future__ import annotations

from arka.config.models import LLMConfig
from arka.llm.openai_client import build_openai_client


def test_build_openai_client_uses_configured_base_url() -> None:
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
    )

    client = build_openai_client(config)

    assert str(client.base_url) == "https://openrouter.ai/api/v1/"


def test_build_openai_client_sets_openrouter_compatible_headers() -> None:
    config = LLMConfig(
        provider="openai",
        model="google/gemini-3.1-flash-lite-preview",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        openai_compatible={
            "referer": "https://example.com",
            "title": "arka",
        },
    )

    client = build_openai_client(config)

    assert client.default_headers["HTTP-Referer"] == "https://example.com/"
    assert client.default_headers["X-Title"] == "arka"
