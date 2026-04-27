from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from arka.config.loader import ConfigLoader
from arka.config.models import ResolvedConfig
from arka.core.paths import RunPaths
from arka.pipeline.artifacts import StageArtifacts, get_extra
from arka.pipeline.checkpoint import CheckpointManager
from arka.pipeline.models import RunResult, StageContext, StageErrorInfo, StageStat
from arka.pipeline.output import OutputWriter
from arka.pipeline.reporter import RunReporter
from arka.pipeline.stages import Stage
from arka.records.models import Record, StageEvent

logger = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.config_loader = ConfigLoader()
        self.output_writer = OutputWriter()
        self.reporter = RunReporter(
            project_root=project_root, output_writer=self.output_writer
        )

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
        timestamp = datetime.now(UTC).isoformat()

        checkpoint_manager.register_run(
            run_id=run_id,
            config_hash=config_hash,
            status="running",
        )
        self._write_resolved_config(resolved_config, run_paths.resolved_config_path)

        records: list[Record] = []
        stage_stats: list[StageStat] = []
        failed_stage_name: str | None = None
        failed_error: StageErrorInfo | None = None

        try:
            for i, stage in enumerate(stages, 1):
                stage_path = run_paths.stage_data_path(stage.name)
                stage_checkpoint = checkpoint_manager.load_stage(run_id, stage.name)
                if self._should_resume_stage(
                    resume=resume,
                    stage_path=stage_path,
                    stage_checkpoint=stage_checkpoint,
                ):
                    records = self.output_writer.read_parquet(stage_path)
                    stage_stats.append(
                        self._build_stage_stat(
                            stage_name=stage.name,
                            count_in=int(stage_checkpoint["count_out"]),
                            count_out=len(records),
                            status="resumed",
                            resumed=True,
                            stats_path=run_paths.stage_stats_path(stage.name),
                        )
                    )
                    # DX: Provide per-stage progress indication for skipped stages
                    print(
                        f"Skipping stage {i}/{len(stages)}: {stage.name} (resumed from checkpoint)..."
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
                    checkpoint_manager=checkpoint_manager,
                )
                count_in = len(records)
                # DX: Provide per-stage progress indication during long runs
                print(
                    f"Running stage {i}/{len(stages)}: {stage.name} ({count_in} records in)..."
                )
                try:
                    stage_output = list(stage.run(records, context))
                except Exception as exc:
                    failed_stage_name = stage.name
                    failed_error = StageErrorInfo(
                        type=exc.__class__.__name__,
                        message=str(exc),
                    )
                    stage_stats.append(
                        self._build_stage_stat(
                            stage_name=stage.name,
                            count_in=count_in,
                            count_out=len(records),
                            status="failed",
                            resumed=False,
                            stats_path=run_paths.stage_stats_path(stage.name),
                            error=failed_error,
                        )
                    )
                    checkpoint_manager.save_stage(
                        run_id=run_id,
                        stage_name=stage.name,
                        artifact_path=stage_path,
                        count_in=count_in,
                        count_out=len(records),
                        status="failed",
                    )
                    raise

                records = self._append_stage_events(
                    records=stage_output,
                    stage_name=stage.name,
                    action=stage.stage_action,
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
                    self._build_stage_stat(
                        stage_name=stage.name,
                        count_in=count_in,
                        count_out=len(records),
                        status="completed",
                        resumed=False,
                        stats_path=run_paths.stage_stats_path(stage.name),
                    )
                )
        except Exception:
            failed_error_payload = RunReporter.serialize_error(
                failed_stage_name, failed_error
            )
            manifest = self.reporter.build_manifest(
                run_id=run_id,
                config_hash=config_hash,
                timestamp=timestamp,
                stages=stages,
                stage_stats=stage_stats,
                final_count=len(records),
                dataset_path=None,
                status="failed",
                error=failed_error_payload,
            )
            run_report = self.reporter.build_run_report(
                manifest=manifest,
                stage_stats=stage_stats,
                dataset_path=None,
                status="failed",
                error=failed_error_payload,
                report_dir=run_paths.report_dir,
                records=records,
                config=resolved_config,
                checkpoint_manager=checkpoint_manager,
            )
            run_paths.manifest_path.write_text(json.dumps(manifest, indent=2))
            run_paths.run_report_path.write_text(json.dumps(run_report, indent=2))
            checkpoint_manager.update_run_status(run_id=run_id, status="failed")
            raise

        output_path = self.project_root / resolved_config.output.path
        dataset_path = self.output_writer.write_jsonl(
            records=records,
            path=output_path,
            output_format=resolved_config.output.format,
        )
        manifest = self.reporter.build_manifest(
            run_id=run_id,
            config_hash=config_hash,
            timestamp=timestamp,
            stages=stages,
            stage_stats=stage_stats,
            final_count=len(records),
            dataset_path=dataset_path,
            status="completed",
            error=None,
        )
        run_report = self.reporter.build_run_report(
            manifest=manifest,
            stage_stats=stage_stats,
            dataset_path=dataset_path,
            status="completed",
            error=None,
            report_dir=run_paths.report_dir,
            records=records,
            config=resolved_config,
            checkpoint_manager=checkpoint_manager,
        )
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
        from arka.records.identity import config_hash

        return config_hash(config)

    def _write_resolved_config(self, config: ResolvedConfig, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))

    def _append_stage_events(
        self, records: list[Record], stage_name: str, action: str = "transformed"
    ) -> list[Record]:
        updated_records: list[Record] = []
        for record in records:
            next_seq = len(record.stage_events) + 1
            stage_events = [
                *record.stage_events,
                StageEvent(
                    stage=stage_name,
                    action=action,
                    seq=next_seq,
                ),
            ]
            updated_records.append(
                record.model_copy(update={"stage_events": stage_events})
            )
        return updated_records

    def _build_stage_stat(
        self,
        stage_name: str,
        count_in: int,
        count_out: int,
        status: str,
        resumed: bool,
        stats_path: Path,
        error: StageErrorInfo | None = None,
    ) -> StageStat:
        report = StageArtifacts.load_report(stats_path)
        return StageStat(
            stage=stage_name,
            count_in=count_in,
            count_out=count_out,
            status=status,
            resumed=resumed,
            dropped_count=report.dropped_count if report else 0,
            drop_reasons=dict(report.drop_reasons) if report else {},
            quality_distribution=self._normalize_quality_distribution(
                get_extra(report, "quality_distribution")
            ),
            error=error,
            cost_usd=self._normalize_cost_usd(get_extra(report, "cost_usd")),
        )

    def _should_resume_stage(
        self,
        *,
        resume: bool,
        stage_path: Path,
        stage_checkpoint: dict[str, str | int] | None,
    ) -> bool:
        if not resume or not stage_path.exists() or stage_checkpoint is None:
            return False
        if str(stage_checkpoint["artifact_path"]) != str(stage_path):
            logger.warning(
                "Skipping resume for %s because checkpoint artifact path %s does not match %s",
                stage_checkpoint["stage_name"],
                stage_checkpoint["artifact_path"],
                stage_path,
            )
            return False
        return stage_checkpoint["status"] == "completed"

    def _normalize_quality_distribution(self, payload: Any) -> dict[str, float] | None:
        if not isinstance(payload, dict):
            return None
        return {
            str(key): float(value)
            for key, value in payload.items()
            if isinstance(value, int | float)
        }

    def _normalize_cost_usd(self, payload: Any) -> float | None:
        if payload is None:
            return None
        if not isinstance(payload, int | float):
            return None
        return float(payload)
