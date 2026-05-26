from __future__ import annotations

from pathlib import Path

import pytest

from arka.pipeline.models import StageContext
from arka.pipeline.runner import PipelineRunner
from arka.pipeline.stages import Stage
from arka.records.models import Record


class CrashingStage(Stage):
    name = "02_generate"
    stage_action = "generated"

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        raise ValueError("Something went terribly wrong inside the stage")


def test_runner_wraps_stage_exceptions_with_stage_name(tmp_path: Path) -> None:
    from arka.config.loader import ConfigLoader

    config_data = {
        "version": "1",
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "dummy",
            "base_url": "https://api.openai.com/v1",
        },
        "executor": {"mode": "threadpool", "max_workers": 2},
        "data_source": {"type": "seeds", "path": "./dummy.jsonl"},
        "generator": {
            "type": "prompt_based",
            "target_count": 1,
            "generation_multiplier": 1,
        },
        "filters": {"target_count": 1},
        "embeddings": {"provider": "huggingface", "model": "all-MiniLM-L6-v2"},
        "output": {"format": "jsonl", "path": "./output/dummy.jsonl"},
    }

    config = ConfigLoader().load_dict(config_data)
    runner = PipelineRunner(project_root=tmp_path)

    stages = [CrashingStage()]

    with pytest.raises(RuntimeError) as exc_info:
        runner.run(config=config, stages=stages, run_id="test-run", resume=False)

    err_msg = str(exc_info.value)
    assert "Stage '02_generate' failed" in err_msg
    assert "Something went terribly wrong inside the stage" in err_msg
