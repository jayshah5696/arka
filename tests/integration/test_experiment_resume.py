"""Resume must preserve completed work and reject changes before writing run state."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from arka.cli import cli


def config_data() -> dict:
    return {
        "version": "2",
        "models": {
            "writer": {
                "provider": "openai",
                "model": "unused",
                "api_key": "secret",
                "base_url": "http://127.0.0.1:1/v1",
            }
        },
        "default_model": "writer",
        "execution": {"max_workers": 1},
        "pipeline": [
            {"id": "source", "config": {"type": "seed_source", "path": "seeds.jsonl"}},
            {"id": "normalize", "config": {"type": "normalize_conversation"}},
        ],
        "output": {"format": "jsonl", "path": "dataset.jsonl"},
    }


def invoke(path: Path, *args: str):
    return CliRunner().invoke(
        cli, ["--config", str(path), "--run-id", "experiment", *args]
    )


@pytest.mark.parametrize("change", ["model", "source", "pipeline", "output"])
def test_changed_experiment_is_rejected_without_mutating_run(
    tmp_path: Path, change: str
) -> None:
    data = config_data()
    path = tmp_path / "config.yaml"
    seed = tmp_path / "seeds.jsonl"
    seed.write_text('{"instruction":"Explain water","response":"Water is a liquid."}\n')
    path.write_text(yaml.safe_dump(data))
    initial = invoke(path)
    assert initial.exit_code == 0, initial.output
    run = tmp_path / "runs/experiment"
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    if change == "source":
        seed.write_text('{"instruction":"Changed","response":"Changed answer"}\n')
    elif change == "model":
        data["models"]["writer"]["model"] = "changed"
    elif change == "pipeline":
        data["pipeline"].reverse()
    else:
        data["output"]["format"] = "chatml"
    path.write_text(yaml.safe_dump(data))
    rejected = invoke(path, "--resume")
    assert rejected.exit_code != 0
    assert "new run" in rejected.output.lower(), rejected.output
    assert {
        p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()
    } == before


def test_operational_resume_retains_original_config_and_records_history(
    tmp_path: Path,
) -> None:
    data = config_data()
    path = tmp_path / "config.yaml"
    (tmp_path / "seeds.jsonl").write_text(
        '{"instruction":"Explain water","response":"Water is a liquid."}\n'
    )
    path.write_text(yaml.safe_dump(data))
    result = invoke(path)
    assert result.exit_code == 0, result.output
    run = tmp_path / "runs/experiment"
    original_config = (run / "config.resolved.yaml").read_bytes()
    original_data = (tmp_path / "dataset.jsonl").read_bytes()
    data = deepcopy(data)
    data["execution"] = {"max_workers": 3, "timeout_seconds": 60, "max_retries": 4}
    path.write_text(yaml.safe_dump(data))
    resumed = invoke(path, "--resume")
    assert resumed.exit_code == 0, resumed.output
    assert "Skipping stage" in resumed.output
    assert (run / "config.resolved.yaml").read_bytes() == original_config
    assert (tmp_path / "dataset.jsonl").read_bytes() == original_data
    history = [
        json.loads(line)
        for line in (run / "execution.history.jsonl").read_text().splitlines()
    ]
    assert len(history) == 2
    assert history[0]["settings"]["executor"]["max_workers"] == 1
    assert history[1]["settings"]["executor"]["max_workers"] == 3
    assert "secret" not in (run / "plan.json").read_text()


def test_preset_generates_editable_config_without_credentials(tmp_path: Path) -> None:
    from arka.config.presets import init_config

    path = tmp_path / "experiment.yaml"
    result = CliRunner().invoke(
        init_config,
        [
            "--preset",
            "sft",
            "--output",
            str(path),
            "--source",
            "seeds.jsonl",
            "--model",
            "local-model",
            "--base-url",
            "http://localhost:1234/v1",
        ],
    )
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(path.read_text())
    assert data["version"] == "2"
    assert "preset" not in data
    assert data["models"]["writer"]["api_key"] == "${ARKA_API_KEY}"
    assert data["pipeline"][0]["config"]["path"] == "seeds.jsonl"
    original = path.read_bytes()
    repeated = CliRunner().invoke(
        init_config,
        [
            "--preset",
            "sft",
            "--output",
            str(path),
            "--source",
            "other.jsonl",
            "--model",
            "local-model",
            "--base-url",
            "http://localhost:1234/v1",
        ],
    )
    assert repeated.exit_code != 0
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "bad_input",
    [
        "yaml_list",
        "duplicate_yaml_key",
        "ambiguous_model",
        "transport_in_model",
        "empty_reference",
        "unhashable_yaml_key",
    ],
)
def test_invalid_authoring_fails_cleanly(tmp_path: Path, bad_input: str) -> None:
    data = config_data()
    (tmp_path / "seeds.jsonl").write_text(
        '{"instruction":"Question","response":"Answer"}\n'
    )
    (tmp_path / "rubric.yaml").write_text(
        'version: "1"\ndescription: Quality\ndimensions:\n  - name: quality\n    description: Quality\n    scale_min: 1\n    scale_max: 5\n    criteria: {1: poor, 5: good}\noverall_weights: {quality: 1}\n'
    )
    if bad_input == "ambiguous_model":
        data["pipeline"].append(
            {
                "id": "audit",
                "config": {
                    "type": "source_audit",
                    "rubric_path": "rubric.yaml",
                    "min_scores": {"quality": 3},
                    "sample_size": 1,
                    "judge_llm_override": {"model": "hidden-model"},
                },
            }
        )
    elif bad_input == "transport_in_model":
        data["models"]["writer"]["timeout_seconds"] = 100
    elif bad_input == "empty_reference":
        data["pipeline"][0]["model"] = ""
    text = yaml.safe_dump(data)
    if bad_input == "yaml_list":
        text = "- not-a-config\n"
    elif bad_input == "duplicate_yaml_key":
        text += "default_model: writer\n"
    elif bad_input == "unhashable_yaml_key":
        text = "? [invalid, key]\n: value\n"
    path = tmp_path / "invalid.yaml"
    path.write_text(text)
    result = invoke(path, "--validate-config")
    assert result.exit_code != 0
    assert (
        "Configuration is invalid" in result.output
        or "YAML syntax error" in result.output
    )
    assert not (tmp_path / "runs").exists()


def test_resume_fingerprints_rubric_contents(tmp_path: Path) -> None:
    data = config_data()
    # A configured rubric is a dependency even when this pipeline makes no judge calls.
    data["labeling_engine"] = {"rubric_path": "rubric.yaml"}
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text("original rubric")
    (tmp_path / "seeds.jsonl").write_text(
        '{"instruction":"Question","response":"Answer"}\n'
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    first = invoke(path)
    assert first.exit_code == 0, first.output
    rubric.write_text("changed rubric")
    rejected = invoke(path, "--resume")
    assert rejected.exit_code != 0
    assert "dependencies" in rejected.output
