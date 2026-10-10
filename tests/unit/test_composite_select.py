"""Tests for the CompositeSelectStage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from arka.config.loader import ConfigLoader
from arka.config.models import CompositeSelectConfig
from arka.pipeline.models import StageContext
from arka.pipeline.scoring_stages import CompositeSelectStage
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)


def _record(
    record_id: str,
    *,
    quality: float | None = None,
    reward_model: float | None = None,
    ifd: float | None = None,
    taxonomy: dict[str, list[str]] | None = None,
) -> ConversationRecord:
    return ConversationRecord(
        id=record_id,
        content_hash=f"hash-{record_id}",
        source=RecordSource(type="generated"),
        lineage=RecordLineage(root_id=f"root-{record_id}", parent_ids=[]),
        payload=ConversationPayload(
            instruction=f"Q-{record_id}", response=f"A-{record_id}"
        ),
        scores=RecordScores(
            quality=quality,
            reward_model=reward_model,
            ifd=ifd,
            quality_per_dim={"taxonomy_nodes": taxonomy} if taxonomy else {},
        ),
        config_hash="cfg-1",
        created_at="2026-04-14T00:00:00Z",
    )


def _ctx(tmp_path: Path, **select_overrides) -> StageContext:
    config = ConfigLoader().load_dict(
        {
            "version": "1",
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "test-key",
                "base_url": "https://api.openai.com/v1",
            },
            "executor": {"mode": "threadpool", "max_workers": 1},
            "data_source": {"type": "seeds", "path": "./seeds.jsonl"},
            "generator": {
                "type": "prompt_based",
                "target_count": 2,
                "generation_multiplier": 1,
            },
            "filters": {
                "target_count": 10,
                "stages": [{"type": "select", **select_overrides}],
            },
            "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
        }
    )
    work_dir = tmp_path / "runs" / "run-1" / "stages" / "05_composite_select"
    work_dir.mkdir(parents=True, exist_ok=True)
    return StageContext(
        run_id="run-1",
        stage_name="05_composite_select",
        work_dir=work_dir,
        config=config,
        executor_mode=config.executor.mode,
        max_workers=config.executor.max_workers,
    )


def test_composite_select_keeps_top_n_by_weighted_score(tmp_path: Path) -> None:
    ctx = _ctx(
        tmp_path,
        target_count=2,
        weights={"quality": 0.5, "reward_model": 0.5},
        strategy="top_n",
    )
    stage = CompositeSelectStage()

    records = stage.run(
        [
            _record("1", quality=0.9, reward_model=0.8),  # composite = 0.85
            _record("2", quality=0.3, reward_model=0.2),  # composite = 0.25
            _record("3", quality=0.7, reward_model=0.6),  # composite = 0.65
        ],
        ctx,
    )

    assert len(records) == 2
    assert [r.id for r in records] == ["1", "3"]

    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["count_in"] == 3
    assert stats["count_out"] == 2
    assert stats["dropped_count"] == 1


def test_composite_select_handles_missing_scores_as_zero(tmp_path: Path) -> None:
    ctx = _ctx(
        tmp_path,
        target_count=1,
        weights={"quality": 0.5, "ifd": 0.5},
    )
    stage = CompositeSelectStage()

    records = stage.run(
        [
            _record("1", quality=0.8, ifd=None),  # composite = 0.4
            _record("2", quality=0.3, ifd=0.9),  # composite = 0.6
        ],
        ctx,
    )

    assert len(records) == 1
    assert records[0].id == "2"


def test_composite_select_passes_all_when_disabled(tmp_path: Path) -> None:
    config = ConfigLoader().load_dict(
        {
            "version": "1",
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "test-key",
                "base_url": "https://api.openai.com/v1",
            },
            "executor": {"mode": "threadpool", "max_workers": 1},
            "data_source": {"type": "seeds", "path": "./seeds.jsonl"},
            "generator": {
                "type": "prompt_based",
                "target_count": 2,
                "generation_multiplier": 1,
            },
            "filters": {"target_count": 10},
            "output": {"format": "jsonl", "path": "./output/dataset.jsonl"},
        }
    )
    work_dir = tmp_path / "runs" / "run-1" / "stages" / "05_composite_select"
    work_dir.mkdir(parents=True, exist_ok=True)
    ctx = StageContext(
        run_id="run-1",
        stage_name="05_composite_select",
        work_dir=work_dir,
        config=config,
        executor_mode=config.executor.mode,
        max_workers=config.executor.max_workers,
    )
    stage = CompositeSelectStage()

    all_records = [_record("1", quality=0.5), _record("2", quality=0.3)]
    result = stage.run(all_records, ctx)

    assert len(result) == 2


def test_coverage_reserves_rare_path_and_fills_by_score(tmp_path: Path) -> None:
    ctx = _ctx(
        tmp_path,
        target_count=2,
        weights={"quality": 1.0},
        coverage=[{"factor": "topic", "path": ["science"], "min_count": 1}],
    )
    records = [
        _record("high", quality=1.0),
        _record("medium", quality=0.8, taxonomy={"topic": ["arts"]}),
        _record("rare", quality=0.1, taxonomy={"topic": ["science", "physics"]}),
    ]
    selected = CompositeSelectStage().run(records, ctx)
    assert [r.id for r in selected] == ["high", "rare"]
    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["coverage_status"] == "satisfied"
    assert stats["unmet_constraints"] == []
    assert stats["score_difference_from_top_n"] == pytest.approx(-0.7)
    assert stats["taxonomy_coverage"][0] == {
        "factor": "topic",
        "path": ["science"],
        "min_count": 1,
        "input_count": 1,
        "selected_count": 1,
    }


def test_coverage_search_recovers_from_greedy_multifactor_choice(
    tmp_path: Path,
) -> None:
    # Greedy takes AB, leaving C and D for two slots. AC + BD fits two slots.
    ctx = _ctx(
        tmp_path,
        target_count=2,
        weights={"quality": 1},
        coverage=[{"factor": f, "path": ["yes"]} for f in "ABCD"],
    )
    records = [
        _record("AB", quality=1, taxonomy={f: ["yes"] for f in "AB"}),
        _record("AC", quality=0.8, taxonomy={f: ["yes"] for f in "AC"}),
        _record("BD", quality=0.7, taxonomy={f: ["yes"] for f in "BD"}),
        _record("C", quality=0.5, taxonomy={"C": ["yes"]}),
        _record("D", quality=0.4, taxonomy={"D": ["yes"]}),
    ]
    selected = CompositeSelectStage().run(records, ctx)
    assert [r.id for r in selected] == ["AC", "BD"]
    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert all(entry["selected_count"] == 1 for entry in stats["taxonomy_coverage"])


@pytest.mark.parametrize("policy", ["fail_fast", "best_effort"])
def test_coverage_reports_infeasible_minimums(tmp_path: Path, policy: str) -> None:
    ctx = _ctx(
        tmp_path,
        target_count=1,
        weights={"quality": 1},
        coverage=[
            {"factor": "topic", "path": ["science"]},
            {"factor": "topic", "path": ["arts"]},
        ],
        coverage_policy=policy,
    )
    records = [
        _record("science", quality=1, taxonomy={"topic": ["science"]}),
        _record("arts", quality=0.5, taxonomy={"topic": ["arts"]}),
    ]
    stage = CompositeSelectStage()
    if policy == "fail_fast":
        with pytest.raises(ValueError, match="infeasible"):
            stage.run(records, ctx)
    else:
        assert [r.id for r in stage.run(records, ctx)] == ["science"]
    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["coverage_status"] == "infeasible"
    assert stats["unmet_constraints"]
    assert stats["count_out"] == (0 if policy == "fail_fast" else 1)


def test_coverage_reports_absent_nodes_even_when_all_records_fit(
    tmp_path: Path,
) -> None:
    ctx = _ctx(
        tmp_path,
        target_count=10,
        weights={},
        coverage=[{"factor": "topic", "path": ["missing"], "min_count": 2}],
        coverage_policy="best_effort",
    )
    records = [_record("unassigned")]
    assert CompositeSelectStage().run(records, ctx) == records
    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["unmet_constraints"][0]["shortfall"] == 2
    assert stats["taxonomy_coverage"][0]["input_count"] == 0


def test_coverage_honors_depth_counts_and_stable_ties(tmp_path: Path) -> None:
    ctx = _ctx(
        tmp_path,
        target_count=2,
        weights={"quality": 1},
        coverage=[{"factor": "topic", "path": ["science", "physics"], "min_count": 2}],
    )
    records = [
        _record("wrong-depth", quality=1, taxonomy={"topic": ["physics"]}),
        _record("first", quality=0.5, taxonomy={"topic": ["science", "physics"]}),
        _record(
            "second", quality=0.5, taxonomy={"topic": ["science", "physics", "optics"]}
        ),
        _record("third", quality=0.5, taxonomy={"topic": ["science", "physics"]}),
    ]
    assert [r.id for r in CompositeSelectStage().run(records, ctx)] == [
        "first",
        "second",
    ]


def test_coverage_search_limit_is_reported_as_unresolved(tmp_path: Path) -> None:
    ctx = _ctx(
        tmp_path,
        target_count=2,
        weights={"quality": 1},
        coverage=[{"factor": f, "path": ["yes"]} for f in "ABCD"],
        coverage_policy="best_effort",
        coverage_search_limit=1,
    )
    records = [
        _record("AB", quality=1, taxonomy={f: ["yes"] for f in "AB"}),
        _record("AC", quality=0.8, taxonomy={f: ["yes"] for f in "AC"}),
        _record("BD", quality=0.7, taxonomy={f: ["yes"] for f in "BD"}),
        _record("C", quality=0.5, taxonomy={"C": ["yes"]}),
        _record("D", quality=0.4, taxonomy={"D": ["yes"]}),
    ]
    CompositeSelectStage().run(records, ctx)
    stats = json.loads((ctx.work_dir / "stats.json").read_text())
    assert stats["coverage_status"] == "search_limit"
    assert stats["unmet_constraints"]


@pytest.mark.parametrize(
    "options",
    [
        {"coverage": [{"factor": "topic", "path": []}]},
        {"coverage": [{"factor": "topic", "path": ["science"], "min_count": 0}]},
        {"coverage_policy": "ignore"},
        {"coverage_search_limit": 0},
    ],
)
def test_coverage_rejects_invalid_config(options: dict) -> None:
    with pytest.raises(ValidationError):
        CompositeSelectConfig(**options)
