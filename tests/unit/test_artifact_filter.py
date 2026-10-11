import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from arka.config.models import ArtifactFilterConfig, ResolvedConfig
from arka.pipeline.artifact_filter import ArtifactFilterStage
from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.records.identity import content_hash
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


def _record(response: str, instruction: str = "Explain") -> ConversationRecord:
    payload = ConversationPayload(instruction=instruction, response=response)
    return ConversationRecord(
        id="record",
        content_hash=content_hash(payload),
        source=RecordSource(type="generated"),
        lineage=RecordLineage(root_id="record", parent_ids=[]),
        payload=payload,
        scores=RecordScores(),
        config_hash="config",
        created_at="2026-10-10T00:00:00Z",
    )


def _ctx(tmp_path: Path, configured: bool = True) -> StageContext:
    config = ResolvedConfig(
        version="1",
        llm={
            "provider": "openai",
            "model": "test",
            "api_key": "test",
            "base_url": "https://api.openai.com/v1",
        },
        executor={"mode": "threadpool", "max_workers": 1},
        pipeline=[{"type": "artifact_filter"}] if configured else [],
        output={"format": "jsonl", "path": "output.jsonl"},
    )
    return StageContext(
        run_id="artifacts",
        stage_name="02i_artifact_filter",
        work_dir=tmp_path,
        config=config,
        executor_mode="threadpool",
        max_workers=1,
    )


@pytest.mark.parametrize(
    ("response", "group"),
    [
        ("Answer. As an AI language model, I cannot know.", "self_reference"),
        ("As of my last update, the number was 12.", "self_reference"),
        ("Hello [Your Name]", "placeholders"),
        ("See [link here].", "placeholders"),
        ("Source citeturn0search0", "tool_fingerprints"),
        ("Source :contentReference[oaicite:0]", "tool_fingerprints"),
    ],
)
def test_artifact_filter_drops_named_groups(
    tmp_path: Path, response: str, group: str
) -> None:
    stage = ArtifactFilterStage()
    ctx = _ctx(tmp_path)
    assert stage.run([_record(response)], ctx) == []
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["drop_reasons"] == {f"artifact_{group}": 1}
    assert stats["group_match_counts"][group] == 1
    assert stats["stripped_count"] == 0
    dropped = OutputWriter().read_parquet(tmp_path / "dropped.parquet")
    assert dropped[0].payload.response == response


def test_artifact_strip_preserves_body_and_updates_hash_without_mutating_input(
    tmp_path: Path,
) -> None:
    body = "Answer  with two spaces.\n\nKeep this formatting."
    original = _record(f"Certainly!\n{body}\nI hope this helps!")
    ctx = _ctx(tmp_path)
    result = ArtifactFilterStage().run([original], ctx)
    assert result[0].payload.response == body
    assert original.payload.response.startswith("Certainly!")
    assert original.stage_events == []
    assert result[0].id == original.id
    assert result[0].lineage == original.lineage
    assert result[0].content_hash == content_hash(result[0].payload)
    assert result[0].content_hash != original.content_hash
    assert result[0].stage_events[-1].action == "stripped"
    assert "before_length=" in result[0].stage_events[-1].details
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["stripped_count"] == 1
    assert stats["strip_match_count"] == 2
    assert stats["group_match_counts"] == {"openers": 1, "closers": 1}


@pytest.mark.parametrize(
    "response",
    [
        "As an AI researcher studying this topic, I use experiments.",
        "We can say, Certainly, the data shows a difference.",
        "Certainly, the data shows a difference.",
        'The phrase "As an AI language model" is a disclaimer.',
        "Use `[Your Name]` in the template.",
        "```text\n[Your Name]\nAs an AI language model\n```",
        "~~~text\n:contentReference[oaicite:0]\n~~~",
        "```text\n[Your Name]",  # An unclosed fence still contains literal code.
    ],
)
def test_artifact_filter_preserves_legitimate_text_and_literal_examples(
    tmp_path: Path, response: str
) -> None:
    record = _record(response)
    assert ArtifactFilterStage().run([record], _ctx(tmp_path)) == [record]


def test_artifact_filter_does_not_hide_real_matches_after_code(tmp_path: Path) -> None:
    record = _record("```text\n[Your Name]\n```\nActual [link here]")
    assert ArtifactFilterStage().run([record], _ctx(tmp_path)) == []


def test_artifact_strip_limit_catches_repeated_anchored_openers(tmp_path: Path) -> None:
    stage = ArtifactFilterStage()
    assert stage.run([_record("Certainly! " * 4 + "Answer.")], _ctx(tmp_path)) == []
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["drop_reasons"] == {"artifact_strip_limit": 1}


def test_artifact_empty_required_field_is_dropped(tmp_path: Path) -> None:
    assert ArtifactFilterStage().run([_record("Certainly!")], _ctx(tmp_path)) == []
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["drop_reasons"] == {"artifact_empty_content": 1}


def test_artifact_custom_actions_and_overlapping_matches(tmp_path: Path) -> None:
    config = ArtifactFilterConfig(
        groups={"custom": ["REMOVE", "REMOVE"]},
        group_action={"custom": "strip"},
        fields=["instruction"],
    )
    stage = ArtifactFilterStage(config)
    result = stage.run(
        [_record("REMOVE untouched", "left REMOVE right")], _ctx(tmp_path)
    )
    assert result[0].payload.instruction == "left  right"
    assert result[0].payload.response == "REMOVE untouched"
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["strip_match_count"] == 1
    assert stats["group_match_counts"] == {"custom": 1}


def test_artifact_opener_can_be_configured_to_drop(tmp_path: Path) -> None:
    stage = ArtifactFilterStage(ArtifactFilterConfig(group_action={"openers": "drop"}))
    assert stage.run([_record("Certainly! Answer")], _ctx(tmp_path)) == []


def test_artifact_stage_absent_is_disabled(tmp_path: Path) -> None:
    record = _record("Certainly! Answer [Your Name]")
    assert ArtifactFilterStage().run([record], _ctx(tmp_path, configured=False)) == [
        record
    ]
    assert not (tmp_path / "stats.json").exists()


@pytest.mark.parametrize(
    "options",
    [
        {"groups": {"custom": ["["]}},
        {"groups": {"custom": [""]}},
        {"groups": {"custom": ["^"]}},
        {"group_action": {"unknown": "strip"}},
        {"max_strips_per_record": -1},
        {"fields": []},
        {"fields": ["unknown"]},
    ],
)
def test_artifact_filter_rejects_invalid_config(options: dict) -> None:
    with pytest.raises(ValidationError):
        ArtifactFilterConfig(**options)


def test_artifact_zero_length_context_match_raises_instead_of_looping(
    tmp_path: Path,
) -> None:
    stage = ArtifactFilterStage(
        ArtifactFilterConfig(
            groups={"custom": ["(?=Answer)"]}, group_action={"custom": "strip"}
        )
    )
    with pytest.raises(ValueError, match="consume text"):
        stage.run([_record("Answer")], _ctx(tmp_path))
