import json

import pytest

from arka.config.models import DecontaminateConfig, ResolvedConfig
from arka.pipeline.decontamination import BenchmarkIndex, DecontaminateStage
from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


def record(instruction, response="A useful answer."):
    return ConversationRecord(
        id="r",
        content_hash="h",
        source=RecordSource(type="seed"),
        lineage=RecordLineage(root_id="r", parent_ids=[]),
        payload=ConversationPayload(instruction=instruction, response=response),
        scores=RecordScores(),
        config_hash="c",
        created_at="2026-10-10T00:00:00Z",
    )


def context(tmp_path, options):
    config = ResolvedConfig(
        version="1",
        llm={
            "provider": "openai",
            "model": "test",
            "api_key": "test",
            "base_url": "http://localhost:1/v1",
        },
        executor={"mode": "threadpool", "max_workers": 1},
        pipeline=[options],
        output={"format": "chatml", "path": "data.jsonl"},
    )
    directory = tmp_path / "stage"
    directory.mkdir(exist_ok=True)
    return StageContext(
        run_id="test",
        stage_name="02k_decontaminate",
        work_dir=directory,
        config=config,
        executor_mode="threadpool",
        max_workers=1,
    )


def eval_file(path, questions):
    path.write_text("\n".join(json.dumps({"question": q}) for q in questions))
    return str(path)


def test_drop_and_flag_store_typed_match_and_drop_columns(tmp_path):
    question = "What year was the Eiffel Tower constructed in Paris?"
    path = eval_file(tmp_path / "facts.jsonl", [question])
    options = {"type": "decontaminate", "eval_paths": [path], "threshold": 0.9}
    ctx = context(tmp_path, options)
    original = record("WHAT year was the Eiffel Tower constructed in Paris?!")
    stage = DecontaminateStage(project_root=tmp_path)
    assert stage.run([original, record("Explain sourdough starter.")], ctx) == [
        record("Explain sourdough starter.")
    ]
    dropped = OutputWriter().read_parquet(ctx.work_dir / "dropped.parquet")
    match = dropped[0].scores.decon_match
    assert match.benchmark == "facts.jsonl" and match.score == 1
    assert dropped[0].stage_events[-1].reason_code == "benchmark_overlap"
    assert "facts.jsonl" in dropped[0].stage_events[-1].details
    import polars as pl

    row = pl.read_parquet(ctx.work_dir / "dropped.parquet").row(0, named=True)
    assert row["decon_benchmark"] == "facts.jsonl" and row["decon_score"] == 1
    flagged = DecontaminateStage(
        DecontaminateConfig(**{**options, "action": "flag"}), tmp_path
    ).run([original], ctx)
    assert flagged[0].payload == original.payload
    assert flagged[0].scores.decon_match == match
    assert original.scores.decon_match is None


def test_per_benchmark_idf_suppresses_common_instructions(tmp_path):
    common = "Answer with a single letter"
    benchmark = eval_file(
        tmp_path / "exam.jsonl",
        [f"{common} unique{i} distinct{i} question{i}" for i in range(90)],
    )
    background = eval_file(
        tmp_path / "other.jsonl",
        [f"Other problem alpha{i} beta{i} gamma{i} delta{i}" for i in range(90)],
    )
    local = BenchmarkIndex([benchmark, background], "question", 5, "per_benchmark")
    global_index = BenchmarkIndex([benchmark, background], "question", 5, "global")
    for text in [common + "!" * i for i in range(90)]:
        assert local.match(text, "instruction", 8) is None
        assert global_index.match(text, "instruction", 8).score >= 0.03


def test_window_does_not_combine_distant_chance_matches(tmp_path):
    path = eval_file(tmp_path / "eval.jsonl", ["a b c d e f g h i j"])
    index = BenchmarkIndex([path], "question", 2, "per_benchmark")
    short = index.match("a b c d e", "instruction", 1)
    long = index.match("a b c " + "unrelated " * 50 + "c d e", "instruction", 1)
    assert short.score > 0.4
    assert long.score < 0.4
    embedded = index.match(
        "unrelated " * 100 + "a b c d e f g h i j" + " unrelated" * 100,
        "instruction",
        1,
    )
    assert embedded.score == 1


@pytest.mark.parametrize("contaminated_prefix", [False, True])
def test_calibration_reports_curve_and_preserves_records(tmp_path, contaminated_prefix):
    path = eval_file(
        tmp_path / "eval.jsonl",
        ["Calculate the total number of red apples in the large basket."],
    )
    ctx = context(
        tmp_path, {"type": "decontaminate", "eval_paths": [path], "calibrate": True}
    )
    originals = [
        record(
            "Calculate the total number of red apples in the large basket."
            if contaminated_prefix
            else "Explain how bread rises."
        )
    ]
    assert DecontaminateStage(project_root=tmp_path).run(originals, ctx) == originals
    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    curve = stats["calibration_curve"]
    assert len(curve) > 1
    assert all(row["clean_rows_dropped"] == int(contaminated_prefix) for row in curve)
    assert curve[-1]["recall"] < 1  # A dictionary substitution is not a perfect match.
    assert any(row["recall"] == 1 for row in curve)
    assert stats["calibration_plant_count"] == 3
    assert stats["dropped_count"] == 0


@pytest.mark.parametrize(
    "options",
    [
        {"eval_paths": []},
        {"eval_paths": ["x"]},
        {"eval_paths": ["x"], "threshold": 0},
        {"eval_paths": ["x"], "threshold": 1.1},
        {"eval_paths": ["x"], "threshold": 0.5, "n": 0},
        {"eval_paths": ["x"], "threshold": 0.5, "fields": []},
    ],
)
def test_invalid_decontamination_config(options):
    with pytest.raises(ValueError):
        DecontaminateConfig(**options)


@pytest.mark.parametrize(
    "content", ["{}", '{"question": null}', '{"question": "tiny"}', "not json"]
)
def test_invalid_eval_data_fails_with_file_context(tmp_path, content):
    path = tmp_path / "bad.jsonl"
    path.write_text(content)
    with pytest.raises(ValueError, match="bad.jsonl"):
        BenchmarkIndex([str(path)], "question", 5, "per_benchmark")
