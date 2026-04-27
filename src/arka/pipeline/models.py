from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from arka.config.models import ResolvedConfig

if TYPE_CHECKING:
    from arka.pipeline.checkpoint import CheckpointManager


@dataclass(frozen=True)
class StageContext:
    run_id: str
    stage_name: str
    work_dir: Path
    config: ResolvedConfig
    executor_mode: str
    max_workers: int
    checkpoint_manager: CheckpointManager | None = None


@dataclass(frozen=True)
class RunResult:
    run_id: str
    final_count: int
    dataset_path: Path | None
    manifest_path: Path
    output_path: Path


class StageErrorInfo(BaseModel):
    """A typed error captured when a Stage raises during a Run.

    Promoted to Pydantic alongside ``StageStat`` per ADR-0001's deferred
    cleanup item: report aggregation has grown enough that boundary-style
    typing buys more than a frozen dataclass does.
    """

    model_config = ConfigDict(frozen=True)

    type: str
    message: str


class StageStat(BaseModel):
    """Per-Stage row aggregated by the runner across one Run.

    Combines the typed StageReport that the Stage wrote to ``stats.json``
    with the runner's view of resume/error/status. Exposed in the Manifest's
    ``stage_stats`` and in the run report's ``stage_yields``.
    """

    model_config = ConfigDict(frozen=True)

    stage: str
    count_in: int
    count_out: int
    status: str
    resumed: bool
    dropped_count: int = 0
    drop_reasons: dict[str, int] = Field(default_factory=dict)
    quality_distribution: dict[str, float] | None = None
    error: StageErrorInfo | None = None
    cost_usd: float | None = None
