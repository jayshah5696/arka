from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root_dir: Path
    run_id: str
    run_dir: Path
    stages_dir: Path
    report_dir: Path
    manifest_path: Path
    run_report_path: Path
    resolved_config_path: Path
    sqlite_path: Path

    @classmethod
    def bootstrap(cls, root_dir: Path, run_id: str) -> RunPaths:
        runs_dir = root_dir / "runs"
        run_dir = runs_dir / run_id
        stages_dir = run_dir / "stages"
        report_dir = run_dir / "report"
        manifest_path = run_dir / "manifest.json"
        run_report_path = report_dir / "run_report.json"
        resolved_config_path = run_dir / "config.resolved.yaml"
        sqlite_path = root_dir / "state.db"

        stages_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            root_dir=root_dir,
            run_id=run_id,
            run_dir=run_dir,
            stages_dir=stages_dir,
            report_dir=report_dir,
            manifest_path=manifest_path,
            run_report_path=run_report_path,
            resolved_config_path=resolved_config_path,
            sqlite_path=sqlite_path,
        )

    def stage_dir(self, stage_name: str) -> Path:
        return self.stages_dir / stage_name

    def stage_data_path(self, stage_name: str) -> Path:
        return self.stage_dir(stage_name) / "data.parquet"

    def stage_dropped_path(self, stage_name: str) -> Path:
        return self.stage_dir(stage_name) / "dropped.parquet"

    def stage_stats_path(self, stage_name: str) -> Path:
        return self.stage_dir(stage_name) / "stats.json"
