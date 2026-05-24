import sys
from pathlib import Path
import warnings
import yaml
import pytest

# Ensure scripts directory can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.migrate_config import migrate_file
from arka.config.loader import ConfigLoader


def test_offline_migration_utility(tmp_path: Path) -> None:
    legacy_yaml = """
version: "1"
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: test-key
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
    input_file = tmp_path / "legacy.yaml"
    output_file = tmp_path / "modern.yaml"
    input_file.write_text(legacy_yaml.strip())

    # Run the offline migration utility
    migrate_file(input_file, output_file)

    assert output_file.exists()

    # Load the migrated config and assert it has 'pipeline' instead of legacy properties
    migrated_content = output_file.read_text()
    migrated_data = yaml.safe_load(migrated_content)

    assert "pipeline" in migrated_data
    assert "data_source" not in migrated_data
    assert "generator" not in migrated_data
    assert "filters" not in migrated_data

    # Verify ConfigLoader loads it without any deprecation warnings
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        loader = ConfigLoader()
        resolved = loader.load(output_file)

    # Filter out external package warnings (like pydantic json_encoders warnings)
    deprecation_warnings = [
        w for w in captured
        if issubclass(w.category, DeprecationWarning) and "Legacy configuration format detected" in str(w.message)
    ]
    assert len(deprecation_warnings) == 0

    assert resolved.executor.max_workers == 4
    assert len(resolved.pipeline) == 3
    assert resolved.pipeline[0].type == "seed_source"
    assert resolved.pipeline[1].type == "normalize_conversation"
