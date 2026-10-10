import json
from pathlib import Path

from arka.cli import main
from arka.pipeline.output import OutputWriter


def test_artifact_strip_precedes_exact_dedup_and_survives_resume(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    seeds = [
        {"instruction": "Explain", "response": "Certainly! A useful answer."},
        {"instruction": "Explain", "response": "A useful answer."},
        {"instruction": "Bad", "response": "Actual [Your Name]"},
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
        "executor": {"mode": "threadpool", "max_workers": 1},
        "pipeline": [
            {"type": "seed_source", "path": "seeds.jsonl"},
            {"type": "artifact_filter"},
            {"type": "exact"},
        ],
        "output": {"format": "chatml", "path": "dataset.jsonl"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    main(["--config", "config.json", "--run-id", "artifacts"])
    rows = [
        json.loads(line)
        for line in (tmp_path / "dataset.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["messages"][-1]["content"] == "A useful answer."
    directory = tmp_path / "runs/artifacts/stages/02i_artifact_filter"
    stripped = OutputWriter().read_parquet(directory / "data.parquet")
    assert stripped[0].stage_events[-2].action == "stripped"
    stats = json.loads((directory / "stats.json").read_text())
    assert stats["stripped_count"] == 1
    assert stats["dropped_count"] == 1
    before = (tmp_path / "dataset.jsonl").read_bytes()
    main(["--config", "config.json", "--run-id", "artifacts", "--resume"])
    assert (tmp_path / "dataset.jsonl").read_bytes() == before
