import json

from arka.cli import main
from arka.pipeline.output import OutputWriter


def test_decontamination_exports_clean_records_and_resumes(tmp_path):
    question = "What year was the Eiffel Tower constructed in Paris?"
    (tmp_path / "eval.jsonl").write_text(json.dumps({"question": question}))
    (tmp_path / "seeds.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"instruction": question, "response": "1889"},
                {
                    "instruction": "Explain how sourdough rises.",
                    "response": "Yeast produces gas.",
                },
            ]
        )
    )
    config = {
        "version": "1",
        "llm": {
            "provider": "openai",
            "model": "test",
            "api_key": "fixture",
            "base_url": "http://localhost:1/v1",
        },
        "executor": {"mode": "threadpool", "max_workers": 1},
        "pipeline": [
            {"type": "seed_source", "path": "seeds.jsonl"},
            {"type": "decontaminate", "eval_paths": ["eval.jsonl"], "threshold": 0.9},
        ],
        "output": {"format": "chatml", "path": "dataset.jsonl"},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    command = ["--config", str(path), "--run-id", "decon"]
    main(command)
    before = (tmp_path / "dataset.jsonl").read_bytes()
    assert len(before.splitlines()) == 1
    assert (
        json.loads(before)["messages"][0]["content"] == "Explain how sourdough rises."
    )
    directory = tmp_path / "runs/decon/stages/02k_decontaminate"
    dropped = OutputWriter().read_parquet(directory / "dropped.parquet")
    assert dropped[0].scores.decon_match.benchmark == "eval.jsonl"
    assert json.loads((directory / "stats.json").read_text())[
        "benchmark_match_counts"
    ] == {"eval.jsonl": 1}
    main(command + ["--resume"])
    assert (tmp_path / "dataset.jsonl").read_bytes() == before
