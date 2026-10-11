import json

import pytest

from arka.config.models import ReasoningLanguageGateConfig, ResolvedConfig
from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.pipeline.reasoning_language import (
    LanguagePrediction,
    ReasoningLanguageGateStage,
)
from arka.pipeline.transforms import NormalizeConversationStage
from arka.records.identity import content_hash
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


class Detector:
    def __init__(self):
        self.calls = []

    def predict(self, text):
        self.calls.append(text)
        if text == "code":
            return LanguagePrediction("en", 0.1)
        return LanguagePrediction("de" if text.startswith("Deutsch") else "en", 0.99)


def record(
    reasoning="Deutsch reasoning",
    instruction="Deutsch question",
    response="Deutsch answer",
):
    payload = ConversationPayload(
        instruction=instruction, response=response, reasoning=reasoning
    )
    return ConversationRecord(
        id="r",
        content_hash=content_hash(payload),
        source=RecordSource(type="seed"),
        lineage=RecordLineage(root_id="r", parent_ids=[]),
        payload=payload,
        scores=RecordScores(),
        config_hash="c",
        created_at="2026-10-10T00:00:00Z",
    )


def context(tmp_path, **options):
    config = ResolvedConfig(
        version="1",
        llm={
            "provider": "openai",
            "model": "test",
            "api_key": "test",
            "base_url": "http://localhost:1/v1",
        },
        executor={"mode": "threadpool", "max_workers": 1},
        pipeline=[{"type": "reasoning_language_gate", **options}],
        output={"format": "chatml", "path": "data.jsonl"},
    )
    return StageContext(
        run_id="test",
        stage_name="02j_reasoning_language_gate",
        work_dir=tmp_path,
        config=config,
        executor_mode="threadpool",
        max_workers=1,
    )


def test_strip_trace_rehashes_and_preserves_answer_and_lineage(tmp_path):
    original = record(reasoning="English reasoning")
    ctx = context(tmp_path)
    kept = ReasoningLanguageGateStage(detector=Detector()).run([original], ctx)
    assert kept[0].payload.reasoning is None
    assert kept[0].payload.response == original.payload.response
    assert kept[0].id == original.id and kept[0].lineage == original.lineage
    assert (
        kept[0].content_hash == content_hash(kept[0].payload) != original.content_hash
    )
    assert kept[0].stage_events[-1].action == "reasoning_stripped"
    details = json.loads(kept[0].stage_events[-1].details)
    assert details["reference"]["language"] == "de"
    assert details["mismatches"]["reasoning"]["language"] == "en"
    assert original.payload.reasoning == "English reasoning"
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["stripped_count"] == 1
    assert stats["field_mismatch_counts"] == {"reasoning": 1}


@pytest.mark.parametrize("reasoning", ["Deutsch reasoning", "code", None])
def test_matching_low_confidence_and_absent_traces_pass(tmp_path, reasoning):
    original = record(reasoning=reasoning)
    detector = Detector()
    assert ReasoningLanguageGateStage(detector=detector).run(
        [original], context(tmp_path)
    ) == [original]
    assert not original.stage_events
    if reasoning is None:
        assert not detector.calls


def test_low_confidence_reference_cannot_establish_mismatch(tmp_path):
    original = record(instruction="code", reasoning="Deutsch reasoning")
    assert ReasoningLanguageGateStage(detector=Detector()).run(
        [original], context(tmp_path)
    ) == [original]


def test_response_mismatch_drop_and_stats(tmp_path):
    original = record(response="English answer")
    ctx = context(tmp_path, on_mismatch="drop_record")
    assert ReasoningLanguageGateStage(detector=Detector()).run([original], ctx) == []
    dropped = OutputWriter().read_parquet(tmp_path / "dropped.parquet")
    assert dropped[0].stage_events[-1].reason_code == "reasoning_language_mismatch"
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["field_mismatch_counts"] == {"response": 1}
    assert stats["field_mismatch_rates"]["response"] == 1


def test_missing_model_warns_and_passes(tmp_path, caplog):
    original = record(reasoning="English reasoning")
    assert ReasoningLanguageGateStage().run([original], context(tmp_path)) == [original]
    assert "lid_model_path" in caplog.text
    assert json.loads((tmp_path / "stats.json").read_text())["enabled"] is False


def test_configured_missing_model_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReasoningLanguageGateStage(project_root=tmp_path).run(
            [record()], context(tmp_path, lid_model_path="missing.ftz")
        )


@pytest.mark.parametrize(
    "options",
    [
        {"min_confidence": -0.1},
        {"min_confidence": 1.1},
        {"checked_fields": []},
        {"reference_field": "unknown"},
        {"on_mismatch": "llm_heal"},
    ],
)
def test_invalid_gate_config(options):
    with pytest.raises(ValueError):
        ReasoningLanguageGateConfig(**options)


@pytest.mark.parametrize("reasoning", [None, "A valid trace."])
def test_reasoning_survives_parquet_and_formats(tmp_path, reasoning):
    original = record(reasoning=reasoning)
    writer = OutputWriter()
    path = tmp_path / "records.parquet"
    writer.write_parquet([original], path)
    restored = writer.read_parquet(path)[0]
    assert restored.payload.reasoning == reasoning
    for fmt in ("jsonl", "chatml", "alpaca"):
        output = tmp_path / f"{fmt}.jsonl"
        writer.write_jsonl([restored], output, output_format=fmt)
        row = json.loads(output.read_text())
        if fmt == "jsonl":
            assert row.get("reasoning") == reasoning
        else:
            answer = (
                row["messages"][-1]["content"] if fmt == "chatml" else row["output"]
            )
            expected = (
                f"<think>\n{reasoning}\n</think>\n\nDeutsch answer"
                if reasoning
                else "Deutsch answer"
            )
            assert answer == expected


def test_normalize_preserves_trace(tmp_path):
    original = record(reasoning=" A valid trace. ")
    result = NormalizeConversationStage().run([original], context(tmp_path))
    assert result[0].payload.reasoning == "A valid trace."
    assert result[0].content_hash == content_hash(result[0].payload)
