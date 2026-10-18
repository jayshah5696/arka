"""Architectural invariant tests enforcing pipeline decoupling, seams, and guardrails."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, get_args, get_origin

from arka.config.models import PipelineStageConfig
from arka.pipeline.decision_stages import get_decision_spec
from arka.pipeline.registry import get_stage_spec
from arka.pipeline.stages import Stage


def test_all_stage_configs_have_registered_stage_spec() -> None:
    """Every concrete stage config in PipelineStageConfig must have a registered StageSpec."""
    union = get_args(PipelineStageConfig)[0]
    members = get_args(union)
    assert len(members) > 0

    for m in members:
        cls = get_args(m)[0] if get_origin(m) is Annotated else m
        stage_type = cls.model_fields["type"].default
        spec = get_stage_spec(stage_type)
        assert spec is not None, (
            f"Missing StageSpec registration for stage type: {stage_type!r}"
        )
        stage_cls = spec.get_stage_class()
        assert issubclass(stage_cls, Stage), (
            f"StageSpec for {stage_type!r} resolves to {stage_cls!r}, which is not a Stage subclass"
        )


def test_all_decision_supporting_configs_have_registered_decision_spec() -> None:
    """Every stage config supporting backend='decision' must have a registered DecisionStageSpec."""
    union = get_args(PipelineStageConfig)[0]
    members = get_args(union)

    decision_supporting_stages: list[str] = []
    for m in members:
        cls = get_args(m)[0] if get_origin(m) is Annotated else m
        stage_type = cls.model_fields["type"].default
        if "backend" in cls.model_fields:
            ann = cls.model_fields["backend"].annotation
            args = get_args(ann)
            if "decision" in args:
                decision_supporting_stages.append(stage_type)
                spec = get_decision_spec(stage_type)
                assert spec is not None, (
                    f"Stage config {cls.__name__} ({stage_type!r}) supports backend='decision' "
                    f"but lacks a registered DecisionStageSpec in decision_stages.py"
                )

    assert len(decision_supporting_stages) >= 5, (
        f"Expected at least 5 decision-supporting stages, found: {decision_supporting_stages}"
    )


def test_no_stage_file_writes_stats_json_directly() -> None:
    """No pipeline stage file (except artifacts.py and runner.py) may write stats.json directly."""
    repo_root = Path(__file__).resolve().parents[2]
    pipeline_dir = repo_root / "src" / "arka" / "pipeline"
    excluded = {"artifacts.py", "runner.py"}

    violations: list[str] = []
    for p in pipeline_dir.glob("*.py"):
        if p.name in excluded:
            continue
        text = p.read_text(encoding="utf-8")
        if '"stats.json"' in text and ".write_text(" in text:
            violations.append(p.name)

    assert not violations, (
        f"Found direct stats.json writes in stage files (must use StageArtifacts): {violations}"
    )


def test_no_stage_file_instantiates_thread_pool_executor() -> None:
    """No pipeline stage file (except models.py and runner.py) may instantiate ThreadPoolExecutor."""
    repo_root = Path(__file__).resolve().parents[2]
    pipeline_dir = repo_root / "src" / "arka" / "pipeline"
    excluded = {"models.py", "runner.py"}

    violations: list[str] = []
    for p in pipeline_dir.glob("*.py"):
        if p.name in excluded:
            continue
        text = p.read_text(encoding="utf-8")
        if "ThreadPoolExecutor(" in text:
            violations.append(p.name)

    assert not violations, (
        f"Found direct ThreadPoolExecutor instantiation in stage files (must use ctx.map_parallel): {violations}"
    )


def test_agents_md_guardrails() -> None:
    """AGENTS.md must exist, stay under 50 lines, and define the Architecture Seams."""
    repo_root = Path(__file__).resolve().parents[2]
    agents_md = repo_root / "AGENTS.md"
    assert agents_md.exists(), "AGENTS.md must exist in repo root"

    lines = agents_md.read_text(encoding="utf-8").splitlines()
    assert len(lines) < 50, (
        f"AGENTS.md has {len(lines)} lines; must stay strictly under 50 lines"
    )

    content = "\n".join(lines)
    assert "## Architecture Seams" in content
    assert "Single Stage Artifact Seam" in content
    assert "Bound Stage Config" in content
    assert "Context-Owned Clients & Concurrency" in content
    assert "Registry-Driven Dispatch" in content
    assert "Boundary vs. Internal Types" in content
    assert "Iterative Verified Slices" in content
