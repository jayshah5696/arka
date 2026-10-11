import json
from pathlib import Path

from arka.cli import main
from arka.pipeline.output import OutputWriter


def test_identity_claim_cli_exports_denial_and_drops_claim(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    seeds = [
        {"instruction": "Who are you?", "response": "I am Gemini."},
        {
            "instruction": "Explain ChatGPT",
            "response": "I'm not ChatGPT. Here is an explanation.",
        },
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
            {"type": "identity_claim"},
        ],
        "output": {"format": "chatml", "path": "dataset.jsonl"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    main(["--config", "config.json", "--run-id", "identity"])
    rows = [
        json.loads(line)
        for line in (tmp_path / "dataset.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["messages"][-1]["content"] == seeds[1]["response"]
    directory = tmp_path / "runs/identity/stages/02d_identity_claim_filter"
    dropped = OutputWriter().read_parquet(directory / "dropped.parquet")
    assert dropped[0].payload.response == "I am Gemini."
    stats = json.loads((directory / "stats.json").read_text())
    assert stats["pattern_group_counts"]["first_person"] == 1
