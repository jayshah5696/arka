from __future__ import annotations

from pathlib import Path

import pytest

from arka.examples_validation import (
    ALLOWED_COSTS,
    HEADER_FIELDS,
    example_yaml_paths,
    header_values,
    validate_examples,
)


@pytest.fixture
def examples_root() -> Path:
    return Path("examples")


@pytest.fixture
def example_paths(examples_root: Path) -> list[Path]:
    return example_yaml_paths(Path("."))


@pytest.fixture(autouse=True)
def _example_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")


@pytest.mark.parametrize("config_path", example_yaml_paths(Path(".")))
def test_all_example_yaml_files_parse_without_schema_errors(config_path: Path) -> None:
    from arka.examples_validation import load_example_config

    resolved = load_example_config(config_path)

    assert resolved.llm.api_key


@pytest.mark.parametrize("config_path", example_yaml_paths(Path(".")))
def test_all_examples_have_required_header_fields(config_path: Path) -> None:
    headers = header_values(config_path)

    for field in HEADER_FIELDS:
        assert field in headers, f"{config_path} missing {field}"


@pytest.mark.parametrize("config_path", example_yaml_paths(Path(".")))
def test_all_examples_use_valid_cost_values(config_path: Path) -> None:
    headers = header_values(config_path)

    assert headers["COST"] in ALLOWED_COSTS


@pytest.mark.parametrize("config_path", example_yaml_paths(Path(".")))
def test_openrouter_configs_reference_openrouter_api_key(config_path: Path) -> None:
    raw_text = config_path.read_text(encoding="utf-8")
    if "openrouter.ai" not in raw_text:
        return

    assert "${OPENROUTER_API_KEY}" in raw_text
    assert "${OPENAI_API_KEY}" not in raw_text


@pytest.mark.parametrize("config_path", example_yaml_paths(Path(".")))
def test_example_output_paths_are_relative(config_path: Path) -> None:
    from arka.examples_validation import load_example_config

    resolved = load_example_config(config_path)

    assert resolved.output.path.startswith("./")


@pytest.mark.parametrize("config_path", example_yaml_paths(Path(".")))
def test_example_seed_references_exist_when_using_examples_seed_dir(
    config_path: Path,
) -> None:
    from arka.examples_validation import load_example_config

    resolved = load_example_config(config_path)
    seed_path = resolved.data_source.path
    if seed_path is None or "/seeds/" not in seed_path:
        return

    assert (config_path.parent / seed_path).resolve().exists()


@pytest.mark.parametrize(
    "config_path",
    [path for path in example_yaml_paths(Path(".")) if path.parent.name == "future"],
)
def test_future_examples_have_todo_comment_with_target_slice(config_path: Path) -> None:
    raw_text = config_path.read_text(encoding="utf-8")

    assert "# TODO:" in raw_text
    assert "slice" in raw_text.lower() or "milestone" in raw_text.lower()


def test_validate_examples_reports_zero_errors() -> None:
    assert validate_examples(Path(".")) == []
