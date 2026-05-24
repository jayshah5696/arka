from __future__ import annotations

from arka.config.models import (
    CanaryFilterConfig,
    ResolvedConfig,
    SeedSourceConfig,
)
from arka.pipeline.models import StageContext
from arka.pipeline.stages import Stage
from arka.records.models import Record


class ExampleStage(Stage):
    name = "01_example"
    config = None

    def run(self, records: list[Record], ctx) -> list[Record]:
        return records


def test_stage_protocol_exposes_name() -> None:
    stage = ExampleStage()

    assert stage.name == "01_example"


def test_get_stage_config_resolution() -> None:
    stage = ExampleStage()

    # 1. Resolves explicit stage.config if present
    stage.config = SeedSourceConfig(path="explicit/path.jsonl")
    ctx_dummy = StageContext(
        run_id="run-1",
        stage_name="01_source",
        work_dir=None,
        config=None,
        executor_mode="threadpool",
        max_workers=1,
    )
    res = stage.get_stage_config(ctx_dummy, SeedSourceConfig)
    assert res.path == "explicit/path.jsonl"

    # Reset config for other tests
    stage.config = None

    # 2. Resolves from pipeline config type matching
    pipeline_cfg = [
        SeedSourceConfig(path="pipeline/path.jsonl"),
        CanaryFilterConfig(phrases=["canary"]),
    ]
    resolved_config = ResolvedConfig.model_validate(
        {
            "version": "1",
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "test-key",
                "base_url": "https://api.openai.com/v1",
            },
            "executor": {"mode": "threadpool", "max_workers": 1},
            "pipeline": pipeline_cfg,
            "output": {"format": "jsonl", "path": "./output.jsonl"},
        }
    )
    ctx_with_config = StageContext(
        run_id="run-1",
        stage_name="01_source",
        work_dir=None,
        config=resolved_config,
        executor_mode="threadpool",
        max_workers=1,
    )
    res_pipeline = stage.get_stage_config(ctx_with_config, CanaryFilterConfig)
    assert res_pipeline.phrases == ["canary"]

    # 3. Resolves legacy config
    # Build legacy-style data for ResolvedConfig
    legacy_resolved_config = ResolvedConfig.model_validate(
        {
            "version": "1",
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "test-key",
                "base_url": "https://api.openai.com/v1",
            },
            "executor": {"mode": "threadpool", "max_workers": 1},
            "data_source": {"type": "seeds", "path": "legacy/path.jsonl"},
            "generator": {
                "type": "prompt_based",
                "target_count": 2,
                "generation_multiplier": 1,
            },
            "filters": {
                "target_count": 2,
                "stages": [{"type": "canary", "phrases": ["legacy-canary"]}],
            },
            "output": {"format": "jsonl", "path": "./output.jsonl"},
        }
    )
    ctx_legacy = StageContext(
        run_id="run-1",
        stage_name="01_source",
        work_dir=None,
        config=legacy_resolved_config,
        executor_mode="threadpool",
        max_workers=1,
    )
    res_legacy = stage.get_stage_config(
        ctx_legacy, CanaryFilterConfig, legacy_field="canary"
    )
    assert res_legacy.phrases == ["legacy-canary"]

    # 4. Falls back to default_val if provided
    fallback_val = SeedSourceConfig(path="fallback/path.jsonl")
    res_fallback = stage.get_stage_config(
        ctx_dummy, SeedSourceConfig, default_val=fallback_val
    )
    assert res_fallback.path == "fallback/path.jsonl"

    # 5. Falls back to calling config_cls()
    res_default = stage.get_stage_config(ctx_dummy, CanaryFilterConfig)
    assert isinstance(res_default, CanaryFilterConfig)
    assert res_default.phrases == []
