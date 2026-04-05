from __future__ import annotations

from pathlib import Path

import pytest

from arka.config.loader import ConfigLoader


@pytest.mark.parametrize(
    ("config_name", "env_name", "env_value"),
    [
        ("config.example.yaml", "OPENAI_API_KEY", "openai-test-key"),
        ("config.openrouter.yaml", "OPENROUTER_API_KEY", "openrouter-test-key"),
        (
            "config.examples.verify-openrouter.yaml",
            "OPENROUTER_API_KEY",
            "openrouter-test-key",
        ),
        (
            "config.examples.resume-openrouter.yaml",
            "OPENROUTER_API_KEY",
            "openrouter-test-key",
        ),
        (
            "config.examples.csv-seeds.yaml",
            "OPENAI_API_KEY",
            "openai-test-key",
        ),
        (
            "config.examples.dedup-quality.yaml",
            "OPENAI_API_KEY",
            "openai-test-key",
        ),
    ],
)
def test_example_configs_load(
    config_name: str, env_name: str, env_value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(env_name, env_value)

    resolved = ConfigLoader().load(Path(config_name))

    assert resolved.llm.api_key == env_value


def test_smoke_config_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")

    resolved = ConfigLoader().load(Path("config.smoke.yaml"))

    assert resolved.data_source.type == "seeds"
    assert resolved.dedup.exact.enabled is False
