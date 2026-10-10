from __future__ import annotations

import json
from pathlib import Path

from arka.cli import main
from arka.pipeline.output import OutputWriter


def test_repetition_pipeline_filters_seed_records_without_llm_calls(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    seeds = [
        {"instruction": "Explain", "response": "hello world " * 60},
        {"instruction": "Explain", "response": "A useful answer."},
    ]
    (tmp_path / "seeds.jsonl").write_text("\n".join(json.dumps(row) for row in seeds))
    config = {
        "version": "1",
        "llm": {
            "provider": "openai",
            "model": "test",
            "api_key": "test",
            "base_url": "https://api.openai.com/v1",
        },
        "pipeline": [
            {"type": "seed_source", "path": "seeds.jsonl"},
            {"type": "repetition"},
        ],
        "executor": {"mode": "threadpool", "max_workers": 1},
        "output": {"format": "chatml", "path": "dataset.jsonl"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    main(["--config", "config.json", "--run-id", "repetition"])

    rows = [
        json.loads(line)
        for line in (tmp_path / "dataset.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["messages"][-1]["content"] == "A useful answer."
    stage_dir = tmp_path / "runs/repetition/stages/02c_repetition_filter"
    stats = json.loads((stage_dir / "stats.json").read_text())
    assert stats["count_in"] == 2
    assert stats["count_out"] == 1
    assert stats["drop_reasons"] == {"repetitive_top_ngram": 1}
    dropped = OutputWriter().read_parquet(stage_dir / "dropped.parquet")
    assert dropped[0].payload.response == ("hello world " * 60).strip()
