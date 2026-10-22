"""Experiment boundary failures: invalid references, ambiguous IDs, and unsafe resume."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from arka.config.loader import ConfigLoader, ConfigValidationError
from arka.pipeline.stage_builder import StageBuilder
from arka.records.identity import config_hash


def experiment() -> dict:
    return {
        "version": "2",
        "models": {
            "writer": {
                "provider": "openai",
                "model": "writer-model",
                "api_key": "secret",
                "base_url": "http://localhost:1234/v1",
            },
            "judge": {
                "provider": "openai",
                "model": "judge-model",
                "api_key": "judge-secret",
                "base_url": "http://localhost:4321/v1",
            },
        },
        "default_model": "writer",
        "execution": {"max_workers": 1, "timeout_seconds": 30, "max_retries": 2},
        "pipeline": [
            {"id": "source", "config": {"type": "seed_source", "path": "seeds.jsonl"}},
            {"id": "normalize", "config": {"type": "normalize_conversation"}},
            {
                "id": "generate",
                "model": "judge",
                "config": {"type": "prompt_based_generator", "target_count": 2},
            },
            {"id": "short", "config": {"type": "length", "min_response_chars": 5}},
            {"id": "long", "config": {"type": "length", "max_response_chars": 50}},
        ],
        "output": {"format": "jsonl", "path": "dataset.jsonl"},
    }


def test_named_model_and_repeated_stage_instances(tmp_path: Path) -> None:
    config = ConfigLoader().load_dict(experiment())
    stages = StageBuilder(config, tmp_path).build()
    assert [stage.name for stage in stages] == [
        "source",
        "normalize",
        "generate",
        "short",
        "long",
    ]
    assert stages[2].llm_config.model == "judge-model"
    assert stages[3].config.min_response_chars == 5
    assert stages[4].config.max_response_chars == 50


@pytest.mark.parametrize(
    "failure",
    [
        "unknown_model",
        "duplicate_id",
        "unsafe_id",
        "unsupported_version",
        "inline_connection",
    ],
)
def test_experiment_rejects_ambiguous_settings(failure: str) -> None:
    data = experiment()
    if failure == "unknown_model":
        data["pipeline"][2]["model"] = "missing"
    elif failure == "duplicate_id":
        data["pipeline"][3]["id"] = "source"
    elif failure == "unsafe_id":
        data["pipeline"][0]["id"] = "../escape"
    elif failure == "unsupported_version":
        data["version"] = "999"
    else:
        data["pipeline"][2]["config"]["llm_override"] = {"model": "untracked"}
    with pytest.raises(ConfigValidationError):
        ConfigLoader().load_dict(data)


def test_operational_changes_preserve_experiment_identity() -> None:
    data = experiment()
    original = config_hash(ConfigLoader().load_dict(data))
    changed = deepcopy(data)
    changed["execution"] = {"max_workers": 4, "timeout_seconds": 90, "max_retries": 5}
    changed["models"]["judge"]["api_key"] = "rotated-secret"
    assert config_hash(ConfigLoader().load_dict(changed)) == original
    changed["pipeline"][2]["config"]["temperature"] = 0.1
    assert config_hash(ConfigLoader().load_dict(changed)) != original


def test_environment_values_cannot_change_yaml_structure(
    tmp_path: Path, monkeypatch
) -> None:
    data = experiment()
    data["models"]["writer"]["api_key"] = "${EXPERIMENT_KEY}"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    value = "a: b\noutput: [malformed # quote'"
    monkeypatch.setenv("EXPERIMENT_KEY", value)
    config = ConfigLoader().load(path)
    assert config.llm.api_key.get_secret_value() == value
    assert config.output.path == "dataset.jsonl"
