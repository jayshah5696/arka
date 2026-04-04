from __future__ import annotations

from pathlib import Path

import pytest

from arka.config.loader import ConfigLoader, ConfigValidationError

CONFIG_YAML = """
version: "1"
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
  base_url: https://api.openai.com/v1
executor:
  mode: threadpool
  max_workers: 4
data_source:
  type: seeds
  path: ./seeds.jsonl
generator:
  type: prompt_based
  target_count: 5
  generation_multiplier: 2
filters:
  target_count: 5
output:
  format: chatml
  path: ./output/dataset.jsonl
"""


OPENROUTER_CONFIG_YAML = """
version: "1"
llm:
  provider: openai
  model: google/gemini-3.1-flash-lite-preview
  api_key: ${OPENROUTER_API_KEY}
  base_url: https://openrouter.ai/api/v1
  openai_compatible:
    referer: https://example.com
    title: arka
executor:
  mode: threadpool
  max_workers: 4
data_source:
  type: seeds
  path: ./seeds.jsonl
generator:
  type: prompt_based
  target_count: 5
  generation_multiplier: 2
filters:
  target_count: 5
output:
  format: chatml
  path: ./output/dataset.jsonl
"""


def test_load_config_resolves_env_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_YAML)

    resolved = ConfigLoader().load(config_path)

    assert resolved.llm.provider == "openai"
    assert resolved.llm.api_key == "test-key"
    assert str(resolved.llm.base_url) == "https://api.openai.com/v1"
    assert resolved.executor.max_workers == 4


def test_load_config_rejects_unknown_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_YAML + "unexpected_key: true\n")

    with pytest.raises(ConfigValidationError):
        ConfigLoader().load(config_path)


def test_load_config_requires_declared_env_vars(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_YAML)

    with pytest.raises(ConfigValidationError, match="OPENAI_API_KEY"):
        ConfigLoader().load(config_path)


def test_load_config_supports_openrouter_style_openai_compatible_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(OPENROUTER_CONFIG_YAML)

    resolved = ConfigLoader().load(config_path)

    assert resolved.llm.model == "google/gemini-3.1-flash-lite-preview"
    assert resolved.llm.api_key == "openrouter-key"
    assert str(resolved.llm.base_url) == "https://openrouter.ai/api/v1"
    assert resolved.llm.openai_compatible is not None
    assert str(resolved.llm.openai_compatible.referer) == "https://example.com/"
    assert resolved.llm.openai_compatible.title == "arka"
