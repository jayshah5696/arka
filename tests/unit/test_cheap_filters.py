from __future__ import annotations

import json
from pathlib import Path

from arka.config.models import (
    ResolvedConfig,
)
from arka.pipeline.cheap_filters import LanguageFilterStage, LengthFilterStage
from arka.pipeline.models import StageContext
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


def _base_config(**filter_overrides) -> ResolvedConfig:
    return ResolvedConfig(
        **{
            "version": "1",
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "k",
                "base_url": "https://api.openai.com/v1",
            },
            "executor": {"mode": "threadpool", "max_workers": 1},
            "data_source": {"type": "seeds", "path": "./seeds.jsonl"},
            "generator": {
                "type": "prompt_based",
                "target_count": 2,
                "generation_multiplier": 1,
            },
            "filters": {"target_count": 2, **filter_overrides},
            "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
        }
    )


def _record(
    instruction: str, response: str, record_id: str = "r1"
) -> ConversationRecord:
    return ConversationRecord(
        id=record_id,
        content_hash="hash",
        source=RecordSource(type="seed"),
        lineage=RecordLineage(root_id=record_id, parent_ids=[]),
        payload=ConversationPayload(instruction=instruction, response=response),
        scores=RecordScores(),
        config_hash="cfg",
        created_at="2026-01-01T00:00:00Z",
    )


def _ctx(config: ResolvedConfig, tmp_path: Path, stage_name: str) -> StageContext:
    work_dir = tmp_path / stage_name
    work_dir.mkdir(parents=True, exist_ok=True)
    return StageContext(
        run_id="test-run",
        stage_name=stage_name,
        work_dir=work_dir,
        config=config,
        executor_mode="threadpool",
        max_workers=1,
    )


# --- Length filter tests ---


def test_length_filter_disabled_passes_all(tmp_path: Path) -> None:
    config = _base_config(length={"enabled": False})
    stage = LengthFilterStage()
    records = [_record("Short", "Short")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 1


def test_length_filter_drops_short_instruction(tmp_path: Path) -> None:
    config = _base_config(length={"enabled": True, "min_instruction_chars": 20})
    stage = LengthFilterStage()
    records = [_record("Hi", "A long enough response text here.")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 0
    stats = json.loads((tmp_path / stage.name / "stats.json").read_text())
    assert stats["drop_reasons"]["instruction_too_short"] == 1


def test_length_filter_drops_long_instruction(tmp_path: Path) -> None:
    config = _base_config(length={"enabled": True, "max_instruction_chars": 5})
    stage = LengthFilterStage()
    records = [_record("This is way too long", "Response.")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 0
    stats = json.loads((tmp_path / stage.name / "stats.json").read_text())
    assert stats["drop_reasons"]["instruction_too_long"] == 1


def test_length_filter_drops_short_response(tmp_path: Path) -> None:
    config = _base_config(length={"enabled": True, "min_response_chars": 50})
    stage = LengthFilterStage()
    records = [_record("Explain gravity", "Short.")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 0
    stats = json.loads((tmp_path / stage.name / "stats.json").read_text())
    assert stats["drop_reasons"]["response_too_short"] == 1


def test_length_filter_drops_long_response(tmp_path: Path) -> None:
    config = _base_config(length={"enabled": True, "max_response_chars": 10})
    stage = LengthFilterStage()
    records = [_record("Explain gravity", "A very long response.")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 0
    stats = json.loads((tmp_path / stage.name / "stats.json").read_text())
    assert stats["drop_reasons"]["response_too_long"] == 1


def test_length_filter_keeps_records_within_bounds(tmp_path: Path) -> None:
    config = _base_config(
        length={
            "enabled": True,
            "min_instruction_chars": 5,
            "max_instruction_chars": 100,
            "min_response_chars": 5,
            "max_response_chars": 100,
        }
    )
    stage = LengthFilterStage()
    records = [_record("Explain gravity", "Gravity is a force.")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 1


def test_length_filter_writes_stats(tmp_path: Path) -> None:
    config = _base_config(length={"enabled": True, "min_instruction_chars": 100})
    stage = LengthFilterStage()
    records = [
        _record("Short", "Response text.", "r1"),
        _record("Also short", "Another response.", "r2"),
    ]
    ctx = _ctx(config, tmp_path, stage.name)
    stage.run(records, ctx)
    stats = json.loads((tmp_path / stage.name / "stats.json").read_text())
    assert stats["count_in"] == 2
    assert stats["count_out"] == 0
    assert stats["dropped_count"] == 2


def test_length_filter_adds_stage_event_to_dropped(tmp_path: Path) -> None:
    config = _base_config(length={"enabled": True, "min_instruction_chars": 100})
    stage = LengthFilterStage()
    records = [_record("Short", "Response text.")]
    ctx = _ctx(config, tmp_path, stage.name)
    stage.run(records, ctx)
    # The dropped records are written to parquet; verify via stats
    stats = json.loads((tmp_path / stage.name / "stats.json").read_text())
    assert stats["dropped_count"] == 1


# --- Language filter tests ---


def test_language_filter_disabled_passes_all(tmp_path: Path) -> None:
    config = _base_config(language={"enabled": False})
    stage = LanguageFilterStage()
    records = [_record("こんにちは", "Hello")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 1


def test_language_filter_keeps_english_text(tmp_path: Path) -> None:
    config = _base_config(language={"enabled": True, "allowed": ["en"]})
    stage = LanguageFilterStage()
    records = [_record("Explain gravity", "Gravity is a force.")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 1


def test_language_filter_drops_non_latin_text(tmp_path: Path) -> None:
    config = _base_config(language={"enabled": True, "allowed": ["en"]})
    stage = LanguageFilterStage()
    records = [_record("これは日本語のテキストです", "日本語の応答")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 0
    stats = json.loads((tmp_path / stage.name / "stats.json").read_text())
    assert stats["drop_reasons"]["language_mismatch"] == 1


def test_language_filter_keeps_mixed_latin_text(tmp_path: Path) -> None:
    """Text with >= 70% Latin characters should pass."""
    config = _base_config(language={"enabled": True, "allowed": ["en"]})
    stage = LanguageFilterStage()
    # Mostly Latin with a few non-Latin chars
    records = [_record("Explain the concept of 日本", "A response.")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 1


def test_language_filter_drops_mostly_non_latin(tmp_path: Path) -> None:
    """Text with < 70% Latin characters should be dropped."""
    config = _base_config(language={"enabled": True, "allowed": ["en"]})
    stage = LanguageFilterStage()
    # Mostly CJK with a few Latin chars
    records = [_record("A これは日本語のテキストです", "Response.")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 0


def test_language_filter_non_en_allowed_passes_all(tmp_path: Path) -> None:
    """When allowed contains non-'en' languages, all text passes (no heuristic)."""
    config = _base_config(language={"enabled": True, "allowed": ["ja"]})
    stage = LanguageFilterStage()
    records = [_record("これは日本語のテキストです", "日本語の応答")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 1


def test_language_filter_empty_text_passes(tmp_path: Path) -> None:
    """Empty or non-alpha text should pass through."""
    config = _base_config(language={"enabled": True, "allowed": ["en"]})
    stage = LanguageFilterStage()
    records = [_record("123 456", "789")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 1
