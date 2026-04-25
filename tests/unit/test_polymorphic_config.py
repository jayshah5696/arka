"""RED tests for polymorphic list-based config design.

The new config shape replaces enabled: bool flags with presence-in-list semantics:
- filters.stages: list of discriminated union configs (presence = enabled)
- dedup: list of discriminated union configs (presence = enabled)
- Order in list = execution order
"""
from __future__ import annotations

import pytest

from arka.config.loader import ConfigLoader, ConfigValidationError
from arka.config.models import (
    CanaryFilterConfig,
    ExactDedupConfig,
    FiltersConfig,
    LanguageFilterConfig,
    LengthFilterConfig,
    NearDedupConfig,
    ResolvedConfig,
    SemanticSimilarityFilterConfig,
)


def _minimal_config(**overrides) -> dict:
    """Base config dict with only required fields."""
    data = {
        "version": "1",
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
        },
        "executor": {"mode": "threadpool", "max_workers": 2},
        "data_source": {"type": "seeds", "path": "./seeds.jsonl"},
        "generator": {
            "type": "prompt_based",
            "target_count": 5,
            "generation_multiplier": 1,
        },
        "filters": {"target_count": 5},
        "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
    }
    data.update(overrides)
    return data


# ── Filters: list-based config ─────────────────────────────


class TestFiltersListConfig:
    def test_no_stages_key_means_no_filter_stages(self) -> None:
        """When filters has no stages key, stages defaults to empty list."""
        config = ResolvedConfig(**_minimal_config())
        assert config.filters.stages == []

    def test_empty_stages_list_is_valid(self) -> None:
        config = ResolvedConfig(**_minimal_config(
            filters={"target_count": 5, "stages": []}
        ))
        assert config.filters.stages == []

    def test_length_filter_parsed_from_list(self) -> None:
        config = ResolvedConfig(**_minimal_config(
            filters={
                "target_count": 5,
                "stages": [
                    {"type": "length", "min_instruction_chars": 40, "max_response_chars": 8000},
                ],
            }
        ))
        assert len(config.filters.stages) == 1
        stage = config.filters.stages[0]
        assert isinstance(stage, LengthFilterConfig)
        assert stage.type == "length"
        assert stage.min_instruction_chars == 40
        assert stage.max_response_chars == 8000

    def test_language_filter_parsed_from_list(self) -> None:
        config = ResolvedConfig(**_minimal_config(
            filters={
                "target_count": 5,
                "stages": [
                    {"type": "language", "allowed": ["en", "fr"]},
                ],
            }
        ))
        stage = config.filters.stages[0]
        assert isinstance(stage, LanguageFilterConfig)
        assert stage.allowed == ["en", "fr"]

    def test_canary_filter_parsed_from_list(self) -> None:
        config = ResolvedConfig(**_minimal_config(
            filters={
                "target_count": 5,
                "stages": [
                    {"type": "canary", "phrases": ["SECRET", "CLASSIFIED"]},
                ],
            }
        ))
        stage = config.filters.stages[0]
        assert isinstance(stage, CanaryFilterConfig)
        assert stage.phrases == ["SECRET", "CLASSIFIED"]

    def test_semantic_similarity_parsed_from_list(self) -> None:
        config = ResolvedConfig(**_minimal_config(
            filters={
                "target_count": 5,
                "stages": [
                    {"type": "semantic_similarity", "threshold": 0.85},
                ],
            }
        ))
        stage = config.filters.stages[0]
        assert isinstance(stage, SemanticSimilarityFilterConfig)
        assert stage.threshold == 0.85

    def test_multiple_filters_preserve_order(self) -> None:
        """Order in the list = execution order."""
        config = ResolvedConfig(**_minimal_config(
            filters={
                "target_count": 5,
                "stages": [
                    {"type": "canary", "phrases": ["SECRET"]},
                    {"type": "length", "min_instruction_chars": 20},
                    {"type": "language", "allowed": ["en"]},
                    {"type": "semantic_similarity", "threshold": 0.9},
                ],
            }
        ))
        types = [s.type for s in config.filters.stages]
        assert types == ["canary", "length", "language", "semantic_similarity"]

    def test_unknown_filter_type_rejected(self) -> None:
        with pytest.raises((ConfigValidationError, Exception)):
            ResolvedConfig(**_minimal_config(
                filters={
                    "target_count": 5,
                    "stages": [{"type": "nonexistent_filter"}],
                }
            ))

    def test_filter_with_wrong_params_rejected(self) -> None:
        """Length filter should not accept canary-specific params."""
        with pytest.raises((ConfigValidationError, Exception)):
            ResolvedConfig(**_minimal_config(
                filters={
                    "target_count": 5,
                    "stages": [{"type": "length", "phrases": ["SECRET"]}],
                }
            ))

    def test_filters_no_enabled_field(self) -> None:
        """The enabled field must not exist on filter configs."""
        config = ResolvedConfig(**_minimal_config(
            filters={
                "target_count": 5,
                "stages": [{"type": "length"}],
            }
        ))
        stage = config.filters.stages[0]
        assert not hasattr(stage, "enabled")

    def test_default_values_applied_when_only_type_given(self) -> None:
        """Length filter with only type should use all defaults."""
        config = ResolvedConfig(**_minimal_config(
            filters={
                "target_count": 5,
                "stages": [{"type": "length"}],
            }
        ))
        stage = config.filters.stages[0]
        assert isinstance(stage, LengthFilterConfig)
        assert stage.min_instruction_chars == 10
        assert stage.max_instruction_chars == 4096


# ── Dedup: list-based config ──────────────────────────────


class TestDedupListConfig:
    def test_no_dedup_key_means_no_dedup_stages(self) -> None:
        """When dedup is absent from config, defaults to empty list."""
        config = ResolvedConfig(**_minimal_config())
        assert config.dedup == []

    def test_empty_dedup_list_is_valid(self) -> None:
        config = ResolvedConfig(**_minimal_config(dedup=[]))
        assert config.dedup == []

    def test_exact_dedup_parsed(self) -> None:
        config = ResolvedConfig(**_minimal_config(
            dedup=[{"type": "exact"}]
        ))
        assert len(config.dedup) == 1
        stage = config.dedup[0]
        assert isinstance(stage, ExactDedupConfig)
        assert stage.type == "exact"

    def test_near_dedup_parsed_with_params(self) -> None:
        config = ResolvedConfig(**_minimal_config(
            dedup=[{"type": "near", "lsh_bands": 32, "jaccard_threshold": 0.8}]
        ))
        stage = config.dedup[0]
        assert isinstance(stage, NearDedupConfig)
        assert stage.lsh_bands == 32
        assert stage.jaccard_threshold == 0.8

    def test_both_dedup_types_preserve_order(self) -> None:
        config = ResolvedConfig(**_minimal_config(
            dedup=[
                {"type": "exact"},
                {"type": "near", "lsh_bands": 16},
            ]
        ))
        types = [s.type for s in config.dedup]
        assert types == ["exact", "near"]

    def test_unknown_dedup_type_rejected(self) -> None:
        with pytest.raises((ConfigValidationError, Exception)):
            ResolvedConfig(**_minimal_config(
                dedup=[{"type": "fuzzy"}]
            ))

    def test_dedup_no_enabled_field(self) -> None:
        """The enabled field must not exist on dedup configs."""
        config = ResolvedConfig(**_minimal_config(
            dedup=[{"type": "exact"}]
        ))
        stage = config.dedup[0]
        assert not hasattr(stage, "enabled")

    def test_near_dedup_defaults(self) -> None:
        config = ResolvedConfig(**_minimal_config(
            dedup=[{"type": "near"}]
        ))
        stage = config.dedup[0]
        assert isinstance(stage, NearDedupConfig)
        assert stage.shingle_size == 5
        assert stage.num_hashes == 128
        assert stage.lsh_bands == 16
        assert stage.jaccard_threshold == 0.7


# ── ConfigLoader YAML round-trip ─────────────────────────


class TestConfigLoaderPolymorphic:
    def test_load_yaml_with_filter_stages_list(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
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
dedup:
  - type: exact
  - type: near
    lsh_bands: 32
filters:
  target_count: 5
  stages:
    - type: length
      min_instruction_chars: 40
    - type: language
      allowed: [en]
    - type: canary
      phrases: ["SECRET"]
output:
  format: jsonl
  path: ./output/dataset.jsonl
""")

        resolved = ConfigLoader().load(config_path)

        assert len(resolved.dedup) == 2
        assert isinstance(resolved.dedup[0], ExactDedupConfig)
        assert isinstance(resolved.dedup[1], NearDedupConfig)
        assert resolved.dedup[1].lsh_bands == 32

        assert len(resolved.filters.stages) == 3
        assert isinstance(resolved.filters.stages[0], LengthFilterConfig)
        assert isinstance(resolved.filters.stages[1], LanguageFilterConfig)
        assert isinstance(resolved.filters.stages[2], CanaryFilterConfig)

    def test_load_yaml_minimal_no_dedup_no_filter_stages(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
version: "1"
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
  base_url: https://api.openai.com/v1
executor:
  mode: threadpool
  max_workers: 2
data_source:
  type: seeds
  path: ./seeds.jsonl
generator:
  type: prompt_based
  target_count: 10
filters:
  target_count: 10
output:
  format: jsonl
  path: ./output/dataset.jsonl
""")

        resolved = ConfigLoader().load(config_path)

        assert resolved.dedup == []
        assert resolved.filters.stages == []
