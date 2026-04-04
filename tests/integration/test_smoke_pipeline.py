from __future__ import annotations

from pathlib import Path

from arka.cli import main


def test_smoke_pipeline_runs_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    (tmp_path / "seeds.jsonl").write_text(
        '{"instruction":"  Say hello  ","response":"  Hello there  "}\n'
    )
    (tmp_path / "config.smoke.yaml").write_text(
        """
version: "1"
run_id: null
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
  path: ./output/smoke-dataset.jsonl
"""
    )

    main(["--config", "config.smoke.yaml", "--run-id", "smoke-run"])

    dataset_path = tmp_path / "output" / "smoke-dataset.jsonl"
    report_path = tmp_path / "runs" / "smoke-run" / "report" / "run_report.json"
    stage_path = (
        tmp_path / "runs" / "smoke-run" / "stages" / "01_source" / "data.parquet"
    )

    assert dataset_path.exists()
    assert report_path.exists()
    assert stage_path.exists()
    assert (
        dataset_path.read_text().strip()
        == '{"instruction":"Say hello","response":"Hello there","system":null,"turns":null}'
    )
