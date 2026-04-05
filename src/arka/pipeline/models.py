from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from arka.config.models import ResolvedConfig


@dataclass(frozen=True)
class StageContext:
    run_id: str
    stage_name: str
    work_dir: Path
    config: ResolvedConfig
    executor_mode: str
    max_workers: int


@dataclass(frozen=True)
class RunResult:
    run_id: str
    final_count: int
    dataset_path: Path
    manifest_path: Path
    output_path: Path


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
