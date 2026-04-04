from __future__ import annotations

from pathlib import Path

from arka.config.loader import ConfigLoader
from arka.pipeline.models import StageContext
from arka.pipeline.source_stages import SeedSourceStage


def build_config(data_source_path: str) -> dict:
    return {
        "version": "1",
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
        },
        "executor": {"mode": "threadpool", "max_workers": 2},
        "data_source": {"type": "seeds", "path": data_source_path},
        "generator": {
            "type": "prompt_based",
            "target_count": 2,
            "generation_multiplier": 1,
        },
        "filters": {"target_count": 2},
        "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
    }


def test_seed_source_stage_reads_jsonl_into_conversation_records(
    tmp_path: Path,
) -> None:
    seed_path = tmp_path / "seeds.jsonl"
    seed_path.write_text('{"instruction":" Say hello ","response":" Hello "}\n')
    config = ConfigLoader().load_dict(build_config("./seeds.jsonl"))
    ctx = StageContext(
        run_id="run-1",
        stage_name="01_source",
        work_dir=tmp_path / "work",
        config=config,
        executor_mode=config.executor.mode,
        max_workers=config.executor.max_workers,
    )

    records = SeedSourceStage(project_root=tmp_path).run([], ctx)

    assert len(records) == 1
    assert records[0].payload.instruction == "Say hello"
    assert records[0].payload.response == "Hello"
    assert records[0].source.type == "seed"


def test_seed_source_stage_reads_csv_into_conversation_records(tmp_path: Path) -> None:
    seed_path = tmp_path / "seeds.csv"
    seed_path.write_text("instruction,response\nQuestion,Answer\n")
    config = ConfigLoader().load_dict(build_config("./seeds.csv"))
    ctx = StageContext(
        run_id="run-1",
        stage_name="01_source",
        work_dir=tmp_path / "work",
        config=config,
        executor_mode=config.executor.mode,
        max_workers=config.executor.max_workers,
    )

    records = SeedSourceStage(project_root=tmp_path).run([], ctx)

    assert len(records) == 1
    assert records[0].payload.instruction == "Question"
    assert records[0].payload.response == "Answer"
