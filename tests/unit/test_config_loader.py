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
dedup:
  exact:
    enabled: false
  near:
    enabled: false
filters:
  target_count: 5
embeddings:
  provider: huggingface
  model: all-MiniLM-L6-v2
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
dedup:
  exact:
    enabled: false
  near:
    enabled: false
filters:
  target_count: 5
  labeling_engine:
    enabled: true
    rubric_path: ./rubrics/sft_quality.yaml
    min_overall_score: 3.5
labeling_engine:
  rubric_path: ./rubrics/sft_quality.yaml
  mode: single
embeddings:
  provider: openai
  model: text-embedding-3-small
  api_key: ${OPENROUTER_API_KEY}
  base_url: https://openrouter.ai/api/v1
  openai_compatible:
    referer: https://example.com
    title: arka
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
    assert resolved.embeddings.provider == "huggingface"
    assert resolved.embeddings.model == "all-MiniLM-L6-v2"


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
    assert resolved.filters.labeling_engine.enabled is True
    assert resolved.labeling_engine.rubric_path == "./rubrics/sft_quality.yaml"
    assert resolved.embeddings.provider == "openai"
    assert resolved.embeddings.model == "text-embedding-3-small"
    assert resolved.embeddings.api_key == "openrouter-key"


def test_load_config_accepts_valid_evol_instruct_config() -> None:
    resolved = ConfigLoader().load_dict(
        {
            "version": "1",
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "test-key",
                "base_url": "https://api.openai.com/v1",
            },
            "executor": {"mode": "threadpool", "max_workers": 1},
            "data_source": {"type": "seeds", "path": "./seeds.jsonl"},
            "generator": {
                "type": "evol_instruct",
                "target_count": 2,
                "generation_multiplier": 1,
                "rounds": 2,
                "branching_factor": 2,
                "operators": [
                    "add_constraints",
                    "deepen",
                    "increase_reasoning_steps",
                    "breadth_mutation",
                ],
            },
            "dedup": {"exact": {"enabled": False}, "near": {"enabled": False}},
            "filters": {"target_count": 2},
            "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
        }
    )

    assert resolved.generator.type == "evol_instruct"
    assert resolved.generator.rounds == 2
    assert resolved.generator.branching_factor == 2


def test_load_config_rejects_unknown_evol_operator() -> None:
    with pytest.raises(ConfigValidationError, match="unsupported names"):
        ConfigLoader().load_dict(
            {
                "version": "1",
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "test-key",
                    "base_url": "https://api.openai.com/v1",
                },
                "executor": {"mode": "threadpool", "max_workers": 1},
                "data_source": {"type": "seeds", "path": "./seeds.jsonl"},
                "generator": {
                    "type": "evol_instruct",
                    "target_count": 2,
                    "generation_multiplier": 1,
                    "rounds": 1,
                    "branching_factor": 1,
                    "operators": ["unknown_operator"],
                },
                "dedup": {"exact": {"enabled": False}, "near": {"enabled": False}},
                "filters": {"target_count": 2},
                "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
            }
        )


def test_load_config_rejects_zero_evol_rounds_or_branching() -> None:
    with pytest.raises(ConfigValidationError, match="generator.rounds"):
        ConfigLoader().load_dict(
            {
                "version": "1",
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "test-key",
                    "base_url": "https://api.openai.com/v1",
                },
                "executor": {"mode": "threadpool", "max_workers": 1},
                "data_source": {"type": "seeds", "path": "./seeds.jsonl"},
                "generator": {
                    "type": "evol_instruct",
                    "target_count": 2,
                    "generation_multiplier": 1,
                    "rounds": 0,
                    "branching_factor": 1,
                    "operators": ["deepen"],
                },
                "dedup": {"exact": {"enabled": False}, "near": {"enabled": False}},
                "filters": {"target_count": 2},
                "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
            }
        )

    with pytest.raises(ConfigValidationError, match="generator.branching_factor"):
        ConfigLoader().load_dict(
            {
                "version": "1",
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "test-key",
                    "base_url": "https://api.openai.com/v1",
                },
                "executor": {"mode": "threadpool", "max_workers": 1},
                "data_source": {"type": "seeds", "path": "./seeds.jsonl"},
                "generator": {
                    "type": "evol_instruct",
                    "target_count": 2,
                    "generation_multiplier": 1,
                    "rounds": 1,
                    "branching_factor": 0,
                    "operators": ["deepen"],
                },
                "dedup": {"exact": {"enabled": False}, "near": {"enabled": False}},
                "filters": {"target_count": 2},
                "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
            }
        )
