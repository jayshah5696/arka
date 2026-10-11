import json
import os
from pathlib import Path

import pytest

from arka.cli import main
from arka.pipeline.output import OutputWriter


def test_real_fasttext_gate_preserves_matching_traces_and_resumes(tmp_path):
    pytest.importorskip("fasttext")
    model_path = os.environ.get("ARKA_TEST_LID_MODEL")
    if not model_path:
        pytest.skip(
            "Set ARKA_TEST_LID_MODEL to a local lid.176.ftz model for native integration"
        )
    german = "Das ist eine deutsche Erklärung über die Lösung dieser Aufgabe."
    english = "This is an English explanation about the solution to this problem."
    seeds = [
        {"instruction": german, "response": german, "reasoning": german},
        {"instruction": german, "response": german, "reasoning": english},
        {"instruction": german, "response": german},
    ]
    (tmp_path / "seeds.jsonl").write_text("\n".join(json.dumps(row) for row in seeds))
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
            {"type": "normalize_conversation"},
            {
                "type": "reasoning_language_gate",
                "lid_model_path": str(Path(model_path).resolve()),
            },
        ],
        "output": {"format": "chatml", "path": "dataset.jsonl"},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    command = ["--config", str(path), "--run-id", "reasoning"]
    main(command)
    before = (tmp_path / "dataset.jsonl").read_bytes()
    contents = [
        json.loads(line)["messages"][-1]["content"] for line in before.splitlines()
    ]
    assert contents == [f"<think>\n{german}\n</think>\n\n{german}", german, german]
    directory = tmp_path / "runs/reasoning/stages/02j_reasoning_language_gate"
    records = OutputWriter().read_parquet(directory / "data.parquet")
    assert [r.payload.reasoning for r in records] == [german, None, None]
    stats = json.loads((directory / "stats.json").read_text())
    assert stats["stripped_count"] == 1 and stats["dropped_count"] == 0
    main(command + ["--resume"])
    assert (tmp_path / "dataset.jsonl").read_bytes() == before
