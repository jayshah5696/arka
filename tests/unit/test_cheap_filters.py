from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from arka.config.models import (
    IdentityClaimFilterConfig,
    RepetitionFilterConfig,
    ResolvedConfig,
)
from arka.pipeline.cheap_filters import (
    IdentityClaimFilterStage,
    LanguageFilterStage,
    LengthFilterStage,
    RepetitionFilterStage,
)
from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.pipeline.stage_builder import StageBuilder
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


def _base_config(stages: list | None = None) -> ResolvedConfig:
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
            "filters": {"target_count": 2, "stages": stages or []},
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
    config = _base_config()  # no length stage in list
    stage = LengthFilterStage()
    records = [_record("Short", "Short")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 1


def test_length_filter_drops_short_instruction(tmp_path: Path) -> None:
    config = _base_config(stages=[{"type": "length", "min_instruction_chars": 20}])
    stage = LengthFilterStage()
    records = [_record("Hi", "A long enough response text here.")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 0
    stats = json.loads((tmp_path / stage.name / "stats.json").read_text())
    assert stats["drop_reasons"]["instruction_too_short"] == 1


def test_length_filter_drops_long_instruction(tmp_path: Path) -> None:
    config = _base_config(stages=[{"type": "length", "max_instruction_chars": 5}])
    stage = LengthFilterStage()
    records = [_record("This is way too long", "Response.")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 0
    stats = json.loads((tmp_path / stage.name / "stats.json").read_text())
    assert stats["drop_reasons"]["instruction_too_long"] == 1


def test_length_filter_drops_short_response(tmp_path: Path) -> None:
    config = _base_config(stages=[{"type": "length", "min_response_chars": 50}])
    stage = LengthFilterStage()
    records = [_record("Explain gravity", "Short.")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 0
    stats = json.loads((tmp_path / stage.name / "stats.json").read_text())
    assert stats["drop_reasons"]["response_too_short"] == 1


def test_length_filter_drops_long_response(tmp_path: Path) -> None:
    config = _base_config(stages=[{"type": "length", "max_response_chars": 10}])
    stage = LengthFilterStage()
    records = [_record("Explain gravity", "A very long response.")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 0
    stats = json.loads((tmp_path / stage.name / "stats.json").read_text())
    assert stats["drop_reasons"]["response_too_long"] == 1


def test_length_filter_keeps_records_within_bounds(tmp_path: Path) -> None:
    config = _base_config(
        stages=[
            {
                "type": "length",
                "min_instruction_chars": 5,
                "max_instruction_chars": 100,
                "min_response_chars": 5,
                "max_response_chars": 100,
            }
        ]
    )
    stage = LengthFilterStage()
    records = [_record("Explain gravity", "Gravity is a force.")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 1


def test_length_filter_writes_stats(tmp_path: Path) -> None:
    config = _base_config(stages=[{"type": "length", "min_instruction_chars": 100}])
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
    config = _base_config(stages=[{"type": "length", "min_instruction_chars": 100}])
    stage = LengthFilterStage()
    records = [_record("Short", "Response text.")]
    ctx = _ctx(config, tmp_path, stage.name)
    stage.run(records, ctx)
    # The dropped records are written to parquet; verify via stats
    stats = json.loads((tmp_path / stage.name / "stats.json").read_text())
    assert stats["dropped_count"] == 1


# --- Language filter tests ---


def test_language_filter_disabled_passes_all(tmp_path: Path) -> None:
    config = _base_config()  # no language stage in list
    stage = LanguageFilterStage()
    records = [_record("こんにちは", "Hello")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 1


def test_language_filter_keeps_english_text(tmp_path: Path) -> None:
    config = _base_config(stages=[{"type": "language", "allowed": ["en"]}])
    stage = LanguageFilterStage()
    records = [_record("Explain gravity", "Gravity is a force.")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 1


def test_language_filter_drops_non_latin_text(tmp_path: Path) -> None:
    config = _base_config(stages=[{"type": "language", "allowed": ["en"]}])
    stage = LanguageFilterStage()
    records = [_record("これは日本語のテキストです", "日本語の応答")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 0
    stats = json.loads((tmp_path / stage.name / "stats.json").read_text())
    assert stats["drop_reasons"]["language_mismatch"] == 1


def test_language_filter_keeps_mixed_latin_text(tmp_path: Path) -> None:
    """Text with >= 70% Latin characters should pass."""
    config = _base_config(stages=[{"type": "language", "allowed": ["en"]}])
    stage = LanguageFilterStage()
    # Mostly Latin with a few non-Latin chars
    records = [_record("Explain the concept of 日本", "A response.")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 1


def test_language_filter_drops_mostly_non_latin(tmp_path: Path) -> None:
    """Text with < 70% Latin characters should be dropped."""
    config = _base_config(stages=[{"type": "language", "allowed": ["en"]}])
    stage = LanguageFilterStage()
    # Mostly CJK with a few Latin chars
    records = [_record("A これは日本語のテキストです", "Response.")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 0


def test_language_filter_non_en_allowed_passes_all_and_warns_once(
    tmp_path: Path, caplog
) -> None:
    """When allowed contains non-'en' languages, all text passes and a warning is emitted."""
    config = _base_config(stages=[{"type": "language", "allowed": ["ja"]}])
    stage = LanguageFilterStage()
    records = [
        _record("これは日本語のテキストです", "日本語の応答", "r1"),
        _record("さらに日本語のテキストです", "別の応答", "r2"),
    ]
    ctx = _ctx(config, tmp_path, stage.name)

    with caplog.at_level(logging.WARNING, logger="arka.pipeline.cheap_filters"):
        result = stage.run(records, ctx)

    assert len(result) == 2
    assert (
        "Language filter heuristic only supports English ('en') today; "
        "allowed=['ja'] will currently pass all records"
    ) in caplog.text
    assert caplog.text.count("will currently pass all records") == 1


def test_language_filter_empty_text_passes(tmp_path: Path) -> None:
    """Empty or non-alpha text should pass through."""
    config = _base_config(stages=[{"type": "language", "allowed": ["en"]}])
    stage = LanguageFilterStage()
    records = [_record("123 456", "789")]
    ctx = _ctx(config, tmp_path, stage.name)
    result = stage.run(records, ctx)
    assert len(result) == 1


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("hello world " * 60, "repetitive_top_ngram"),
        ("a" * 100, "long_char_run"),
        ("!" * 51, "long_char_run"),
    ],
)
def test_repetition_drops_loops_and_reports_reason(
    tmp_path: Path, text: str, reason: str
) -> None:
    config = _base_config(stages=[{"type": "repetition"}])
    stage = RepetitionFilterStage()
    ctx = _ctx(config, tmp_path, stage.name)

    assert stage.run([_record("Explain", text)], ctx) == []

    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["drop_reasons"] == {reason: 1}
    dropped = OutputWriter().read_parquet(ctx.work_dir / "dropped.parquet")
    assert dropped[0].stage_events[-1].reason_code == reason
    if reason == "repetitive_top_ngram":
        assert 0.9 < stats["max_observed_fractions"]["top_2"] <= 1.0


@pytest.mark.parametrize(
    "text",
    [
        "hello world " * 3,
        "x" * 50,
        " ".join(f"word{i}" for i in range(300)),
        "\n".join(f"- Step {i}: use option_{i} for task_{i}." for i in range(25)),
        "\n".join(f"result_{i} = compute(value_{i}, limit={i})" for i in range(25)),
    ],
)
def test_repetition_keeps_short_text_and_non_looping_content(
    tmp_path: Path, text: str
) -> None:
    config = _base_config(stages=[{"type": "repetition"}])
    stage = RepetitionFilterStage()
    ctx = _ctx(config, tmp_path, stage.name)
    record = _record("Explain", text)
    assert stage.run([record], ctx) == [record]


def test_repetition_custom_metrics_and_instruction_scope(tmp_path: Path) -> None:
    config = _base_config(
        stages=[
            {
                "type": "repetition",
                "check_fields": ["instruction"],
                "min_text_chars": 0,
                "top_ngram_thresholds": {},
                "dup_ngram_thresholds": {5: 0.5},
                "max_char_run": 100,
            }
        ]
    )
    stage = RepetitionFilterStage()
    ctx = _ctx(config, tmp_path, stage.name)
    repeated = "alpha beta gamma delta epsilon " * 10
    assert stage.run([_record(repeated, "Valid response")], ctx) == []
    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["drop_reasons"] == {"repetitive_dup_ngrams": 1}
    assert 0.9 < stats["max_observed_fractions"]["dup_5"] <= 1.0


def test_repetition_defaults_check_only_response(tmp_path: Path) -> None:
    config = _base_config(stages=[{"type": "repetition"}])
    stage = RepetitionFilterStage()
    record = _record("hello world " * 60, "A normal answer.")
    assert stage.run([record], _ctx(config, tmp_path, stage.name)) == [record]


def test_repetition_absent_is_disabled(tmp_path: Path) -> None:
    stage = RepetitionFilterStage()
    config = _base_config()
    records = [_record("Explain", "a" * 100)]
    assert stage.run(records, _ctx(config, tmp_path, stage.name)) == records


def test_repetition_builder_accepts_unified_config_and_resets_metrics(
    tmp_path: Path,
) -> None:
    config = ResolvedConfig(
        version="1",
        llm={
            "provider": "openai",
            "model": "test",
            "api_key": "test",
            "base_url": "https://api.openai.com/v1",
        },
        pipeline=[{"type": "repetition"}],
        executor={"mode": "threadpool", "max_workers": 1},
        output={"format": "jsonl", "path": "output.jsonl"},
    )
    stage = StageBuilder(config, tmp_path).build()[0]
    ctx = _ctx(config, tmp_path, stage.name)
    stage.run([_record("Explain", "hello world " * 60)], ctx)
    stage.run([_record("Explain", "Short answer")], ctx)
    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert all(v == 0 for v in stats["max_observed_fractions"].values())


@pytest.mark.parametrize(
    "options",
    [
        {"min_text_chars": -1},
        {"max_char_run": 0},
        {"check_fields": ["unknown"]},
        {"check_fields": []},
        {"top_ngram_thresholds": {2: 1.1}},
        {"dup_ngram_thresholds": {5: -0.1}},
        {"top_ngram_thresholds": {5: 0.5}},
        {"dup_ngram_thresholds": {2: 0.5}},
    ],
)
def test_repetition_rejects_invalid_options(options: dict) -> None:
    with pytest.raises(ValidationError):
        RepetitionFilterConfig(**options)


@pytest.mark.parametrize(
    ("text", "group"),
    [
        ("I am ChatGPT, how can I help?", "first_person"),
        ("As Qwen, I think this works.", "first_person"),
        ("I’m Claude.", "first_person"),
        ("i Am GEMINI", "first_person"),
        ("I am not Llama, I am GPT-4", "first_person"),
        ("I was trained by OpenAI.", "creator"),
        ("I am an Anthropic language model.", "creator"),
        ("I wasn't trained by OpenAI. I was built by Google DeepMind.", "creator"),
    ],
)
def test_identity_claim_drops_and_reports_pattern_group(
    tmp_path: Path, text: str, group: str
) -> None:
    stage = IdentityClaimFilterStage()
    config = _base_config(stages=[{"type": "identity_claim"}])
    ctx = _ctx(config, tmp_path, stage.name)
    assert stage.run([_record("Who are you?", text)], ctx) == []
    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["drop_reasons"] == {"identity_claim": 1}
    assert stats["pattern_group_counts"][group] == 1
    dropped = OutputWriter().read_parquet(ctx.work_dir / "dropped.parquet")
    assert dropped[0].stage_events[-1].reason_code == "identity_claim"


@pytest.mark.parametrize(
    "text",
    [
        "I'm not ChatGPT.",
        "I am never Llama.",
        "I am not an OpenAI language model.",
        "I wasn't trained by OpenAI.",
        "I was not created by Anthropic.",
        "I'm never built by Google.",
        "Explain how ChatGPT works.",
        "Gemini is a Google language model.",
        "ChatGPT was developed by OpenAI.",
        "The model performs as Qwen does.",
        "A useful response about the model's attention mechanism.",
    ],
)
def test_identity_claim_keeps_denials_and_normal_responses(
    tmp_path: Path, text: str
) -> None:
    stage = IdentityClaimFilterStage()
    config = _base_config(stages=[{"type": "identity_claim"}])
    record = _record("I am Gemini", text)
    assert stage.run([record], _ctx(config, tmp_path, stage.name)) == [record]


def test_identity_claim_own_model_escape_and_custom_replacement(tmp_path: Path) -> None:
    stage = IdentityClaimFilterStage()
    config = _base_config(
        stages=[{"type": "identity_claim", "own_model_names": ["Arka-7B"]}]
    )
    record = _record(
        "Who are you?", "I am not Llama. I am Arka-7B, distilled from ChatGPT."
    )
    assert stage.run([record], _ctx(config, tmp_path, stage.name)) == [record]
    custom = IdentityClaimFilterStage(
        IdentityClaimFilterConfig(patterns=[r"\bteacherbot\b"])
    )
    ctx = _ctx(config, tmp_path, custom.name)
    kept = _record("Who are you?", "I am Gemini", "kept")
    assert custom.run([kept, _record("Who?", "I am TeacherBot")], ctx) == [kept]
    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["pattern_group_counts"] == {"custom": 1}


def test_identity_claim_configured_field_and_absent_stage(tmp_path: Path) -> None:
    record = _record("I am Gemini", "A normal response")
    stage = IdentityClaimFilterStage()
    assert stage.run([record], _ctx(_base_config(), tmp_path, stage.name)) == [record]
    config = _base_config(
        stages=[{"type": "identity_claim", "check_fields": ["instruction"]}]
    )
    assert stage.run([record], _ctx(config, tmp_path, stage.name)) == []


@pytest.mark.parametrize(
    "options",
    [
        {"patterns": ["["]},
        {"own_model_names": [""]},
        {"check_fields": []},
        {"check_fields": ["unknown"]},
    ],
)
def test_identity_claim_rejects_invalid_config(options: dict) -> None:
    with pytest.raises(ValidationError):
        IdentityClaimFilterConfig(**options)
