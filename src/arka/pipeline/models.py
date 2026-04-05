from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

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


@dataclass(frozen=True)
class StageErrorInfo:
    type: str
    message: str


@dataclass(frozen=True)
class StageStat:
    stage: str
    count_in: int
    count_out: int
    status: str
    resumed: bool
    dropped_count: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)
    quality_distribution: dict[str, float] | None = None
    error: StageErrorInfo | None = None
    cost_usd: float | None = None
