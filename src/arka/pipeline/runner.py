from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from arka.config.loader import ConfigLoader
from arka.config.models import ResolvedConfig
from arka.core.paths import RunPaths
from arka.pipeline.checkpoint import CheckpointManager
from arka.pipeline.models import RunResult, StageContext, StageStat
from arka.pipeline.output import OutputWriter
from arka.pipeline.stages import Stage
from arka.records.models import Record, StageEvent


class PipelineRunner:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.config_loader = ConfigLoader()
        self.output_writer = OutputWriter()

    def run(
        self,
        config: dict[str, Any] | ResolvedConfig,
        stages: list[Stage],
        run_id: str,
        resume: bool = False,
    ) -> RunResult:
        resolved_config = self._resolve_config(config)
        run_paths = RunPaths.bootstrap(root_dir=self.project_root, run_id=run_id)
        checkpoint_manager = CheckpointManager(run_paths.sqlite_path)
        config_hash = self._config_hash(resolved_config)

        checkpoint_manager.register_run(
            run_id=run_id,
            config_hash=config_hash,
            status="running",
        )
        self._write_resolved_config(resolved_config, run_paths.resolved_config_path)

        records: list[Record] = []
        stage_stats: list[StageStat] = []
        for stage in stages:
            stage_path = run_paths.stage_data_path(stage.name)
            if resume and stage_path.exists():
                records = self.output_writer.read_parquet(stage_path)
                stage_stats.append(
                    StageStat(
                        stage=stage.name,
                        count_in=len(records),
                        count_out=len(records),
                        status="resumed",
                        resumed=True,
                    )
                )
                continue

            stage_dir = run_paths.stage_dir(stage.name)
            stage_dir.mkdir(parents=True, exist_ok=True)
            context = StageContext(
                run_id=run_id,
                stage_name=stage.name,
                work_dir=stage_dir,
                config=resolved_config,
                executor_mode=resolved_config.executor.mode,
                max_workers=resolved_config.executor.max_workers,
            )
            count_in = len(records)
            stage_output = list(stage.run(records, context))
            records = self._append_stage_events(
                records=stage_output,
                stage_name=stage.name,
            )
            self.output_writer.write_parquet(records=records, path=stage_path)
            checkpoint_manager.save_stage(
                run_id=run_id,
                stage_name=stage.name,
                artifact_path=stage_path,
                count_in=count_in,
                count_out=len(records),
                status="completed",
            )
            stage_stats.append(
                StageStat(
                    stage=stage.name,
                    count_in=count_in,
                    count_out=len(records),
                    status="completed",
                    resumed=False,
                )
            )

        output_path = self.project_root / resolved_config.output.path
        dataset_path = self.output_writer.write_jsonl(records=records, path=output_path)
        manifest = {
            "run_id": run_id,
            "config_hash": config_hash,
            "timestamp": datetime.now(UTC).isoformat(),
            "stage_names": [stage.name for stage in stages],
            "stage_stats": [asdict(stage_stat) for stage_stat in stage_stats],
            "final_count": len(records),
            "dataset_path": str(dataset_path),
        }
        run_report = {
            "run_id": run_id,
            "config_hash": config_hash,
            "timestamp": manifest["timestamp"],
            "stage_yields": manifest["stage_stats"],
            "final_count": len(records),
            "dataset_path": str(dataset_path),
        }
        run_paths.manifest_path.write_text(json.dumps(manifest, indent=2))
        run_paths.run_report_path.write_text(json.dumps(run_report, indent=2))
        checkpoint_manager.update_run_status(run_id=run_id, status="completed")

        return RunResult(
            run_id=run_id,
            final_count=len(records),
            dataset_path=dataset_path,
            manifest_path=run_paths.manifest_path,
            output_path=output_path,
        )

    def _resolve_config(
        self, config: dict[str, Any] | ResolvedConfig
    ) -> ResolvedConfig:
        if isinstance(config, ResolvedConfig):
            return config
        return self.config_loader.load_dict(config)

    def _config_hash(self, config: ResolvedConfig) -> str:
        payload = json.dumps(config.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _write_resolved_config(self, config: ResolvedConfig, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))

    def _append_stage_events(
        self, records: list[Record], stage_name: str
    ) -> list[Record]:
        updated_records: list[Record] = []
        for record in records:
            next_seq = len(record.stage_events) + 1
            stage_events = [
                *record.stage_events,
                StageEvent(
                    stage=stage_name,
                    action="transformed",
                    seq=next_seq,
                ),
            ]
            updated_records.append(
                record.model_copy(update={"stage_events": stage_events})
            )
        return updated_records
