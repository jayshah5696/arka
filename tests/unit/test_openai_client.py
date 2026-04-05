from __future__ import annotations

from arka.config.models import LLMConfig
from arka.llm.openai_client import OpenAIClientFactory, build_openai_client


def test_openai_client_factory_uses_configured_base_url() -> None:
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
    )

    client = OpenAIClientFactory(config=config).build()

    assert str(client.base_url) == "https://openrouter.ai/api/v1/"


def test_openai_client_factory_sets_openrouter_compatible_headers() -> None:
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

    client = OpenAIClientFactory(config=config).build()

    assert client.default_headers["HTTP-Referer"] == "https://example.com/"
    assert client.default_headers["X-Title"] == "arka"


def test_build_openai_client_matches_factory_output() -> None:
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        openai_compatible={
            "referer": "https://example.com",
            "title": "arka",
        },
    )

    direct_client = build_openai_client(config)
    factory_client = OpenAIClientFactory(config=config).build()

    assert str(direct_client.base_url) == str(factory_client.base_url)
    assert (
        direct_client.default_headers["HTTP-Referer"]
        == factory_client.default_headers["HTTP-Referer"]
    )
    assert (
        direct_client.default_headers["X-Title"]
        == factory_client.default_headers["X-Title"]
    )
