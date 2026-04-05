from __future__ import annotations

from pathlib import Path

from arka.config.loader import ConfigLoader


def test_config_loader_supports_labeling_engine_section(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
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
  labeling_engine:
    enabled: true
    rubric_path: ./rubrics/sft_quality.yaml
    min_overall_score: 3.5
labeling_engine:
  rubric_path: ./rubrics/sft_quality.yaml
  mode: single
embeddings:
  provider: huggingface
  model: all-MiniLM-L6-v2
output:
  format: chatml
  path: ./output/dataset.jsonl
"""
    )

    resolved = ConfigLoader().load(config_path)

    assert resolved.filters.labeling_engine.enabled is True
    assert resolved.filters.labeling_engine.rubric_path == "./rubrics/sft_quality.yaml"
    assert resolved.labeling_engine.mode == "single"
