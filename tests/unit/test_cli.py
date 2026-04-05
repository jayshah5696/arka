from __future__ import annotations

import json
import re
from pathlib import Path

from arka.cli import _resolve_run_id, main

CONFIG_TEXT = """
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
  target_count: 1
  generation_multiplier: 1
filters:
  target_count: 1
output:
  format: jsonl
  path: ./output/dataset.jsonl
"""


def test_cli_auto_generates_run_id_when_not_provided(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    (tmp_path / "seeds.jsonl").write_text('{"instruction":"Hello?","response":"Hi."}\n')
    (tmp_path / "config.yaml").write_text(CONFIG_TEXT)

    main([])

    assert (tmp_path / "output" / "dataset.jsonl").exists()
    # Should create a run dir with a UUID-like name, not 'manual-run'
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir_name = run_dirs[0].name
    # UUID4 format: 8-4-4-4-12 hex
    assert re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        run_dir_name,
    )
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text())
    assert manifest["run_id"] == run_dir_name


def test_resolve_run_id_prefers_cli_over_config() -> None:
    assert _resolve_run_id("cli-run", "config-run") == "cli-run"


def test_resolve_run_id_falls_back_to_config() -> None:
    assert _resolve_run_id(None, "config-run") == "config-run"


def test_resolve_run_id_auto_generates_uuid_when_both_none() -> None:
    run_id = _resolve_run_id(None, None)
    assert re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        run_id,
    )


def test_cli_supports_explicit_config_run_id_and_resume(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "custom-config.yaml"
    config_path.write_text(CONFIG_TEXT)
    (tmp_path / "seeds.jsonl").write_text('{"instruction":"Hello?","response":"Hi."}\n')
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    main(["--config", str(config_path), "--run-id", "custom-run"])
    main(["--config", str(config_path), "--run-id", "custom-run", "--resume"])

    assert (tmp_path / "runs" / "custom-run" / "manifest.json").exists()
