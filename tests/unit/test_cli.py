from __future__ import annotations

from pathlib import Path

from arka.cli import main

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


def test_cli_runs_with_local_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    (tmp_path / "config.yaml").write_text(CONFIG_TEXT)

    main([])

    assert (tmp_path / "output" / "dataset.jsonl").exists()
    assert (tmp_path / "runs" / "manual-run" / "manifest.json").exists()


def test_cli_supports_explicit_config_run_id_and_resume(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "custom-config.yaml"
    config_path.write_text(CONFIG_TEXT)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    main(["--config", str(config_path), "--run-id", "custom-run"])
    main(["--config", str(config_path), "--run-id", "custom-run", "--resume"])

    assert (tmp_path / "runs" / "custom-run" / "manifest.json").exists()
