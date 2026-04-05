from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from arka.config.loader import ConfigLoader
from arka.config.models import ResolvedConfig
from arka.core.paths import RunPaths
from arka.pipeline.checkpoint import CheckpointManager
from arka.pipeline.models import RunResult, StageContext, StageErrorInfo, StageStat
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
            for stage in stages:
                stage_path = run_paths.stage_data_path(stage.name)
                if resume and stage_path.exists():
                    records = self.output_writer.read_parquet(stage_path)
                    stage_stats.append(
                        self._build_stage_stat(
                            stage_name=stage.name,
                            count_in=len(records),
                            count_out=len(records),
                            status="resumed",
                            resumed=True,
                            stats_path=run_paths.stage_stats_path(stage.name),
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
            manifest = self._build_manifest(
                run_id=run_id,
                config_hash=config_hash,
                timestamp=timestamp,
                stages=stages,
                stage_stats=stage_stats,
                final_count=len(records),
                dataset_path=None,
                status="failed",
                error=self._serialize_error(failed_stage_name, failed_error),
            )
            run_report = self._build_run_report(
                manifest=manifest,
                stage_stats=stage_stats,
                dataset_path=None,
                status="failed",
                error=self._serialize_error(failed_stage_name, failed_error),
            )
            run_paths.manifest_path.write_text(json.dumps(manifest, indent=2))
            run_paths.run_report_path.write_text(json.dumps(run_report, indent=2))
            checkpoint_manager.update_run_status(run_id=run_id, status="failed")
            raise

        output_path = self.project_root / resolved_config.output.path
        dataset_path = self.output_writer.write_jsonl(records=records, path=output_path)
        manifest = self._build_manifest(
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
        run_report = self._build_run_report(
            manifest=manifest,
            stage_stats=stage_stats,
            dataset_path=dataset_path,
            status="completed",
            error=None,
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
        payload = json.dumps(config.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

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
        stats_payload = self._load_stage_stats(stats_path)
        return StageStat(
            stage=stage_name,
            count_in=count_in,
            count_out=count_out,
            status=status,
            resumed=resumed,
            dropped_count=int(stats_payload.get("dropped_count", 0)),
            drop_reasons={
                str(key): int(value)
                for key, value in dict(stats_payload.get("drop_reasons", {})).items()
            },
            quality_distribution=self._normalize_quality_distribution(
                stats_payload.get("quality_distribution")
            ),
            error=error,
        )

    def _load_stage_stats(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def _serialize_stage_stat(self, stage_stat: StageStat) -> dict[str, Any]:
        payload = {
            "stage": stage_stat.stage,
            "count_in": stage_stat.count_in,
            "count_out": stage_stat.count_out,
            "status": stage_stat.status,
            "resumed": stage_stat.resumed,
            "dropped_count": stage_stat.dropped_count,
        }
        if stage_stat.drop_reasons:
            payload["drop_reasons"] = stage_stat.drop_reasons
        if stage_stat.quality_distribution is not None:
            payload["quality_distribution"] = stage_stat.quality_distribution
        if stage_stat.error is not None:
            payload["error"] = {
                "type": stage_stat.error.type,
                "message": stage_stat.error.message,
            }
        return payload

    def _aggregate_drop_reasons(self, stage_stats: list[StageStat]) -> dict[str, int]:
        totals: Counter[str] = Counter()
        for stage_stat in stage_stats:
            totals.update(stage_stat.drop_reasons)
        return dict(totals)

    def _report_quality_distribution(
        self, stage_stats: list[StageStat]
    ) -> dict[str, float] | None:
        for stage_stat in reversed(stage_stats):
            if stage_stat.quality_distribution is not None:
                return stage_stat.quality_distribution
        return None

    def _normalize_quality_distribution(self, payload: Any) -> dict[str, float] | None:
        if not isinstance(payload, dict):
            return None
        return {
            str(key): float(value)
            for key, value in payload.items()
            if isinstance(value, int | float)
        }

    def _build_manifest(
        self,
        *,
        run_id: str,
        config_hash: str,
        timestamp: str,
        stages: list[Stage],
        stage_stats: list[StageStat],
        final_count: int,
        dataset_path: Path | None,
        status: str,
        error: dict[str, str] | None,
    ) -> dict[str, Any]:
        manifest = {
            "run_id": run_id,
            "config_hash": config_hash,
            "timestamp": timestamp,
            "stage_names": [stage.name for stage in stages],
            "stage_stats": [
                self._serialize_stage_stat(stage_stat) for stage_stat in stage_stats
            ],
            "final_count": final_count,
            "dataset_path": str(dataset_path) if dataset_path is not None else None,
            "status": status,
        }
        if error is not None:
            manifest["error"] = error
        return manifest

    def _build_run_report(
        self,
        *,
        manifest: dict[str, Any],
        stage_stats: list[StageStat],
        dataset_path: Path | None,
        status: str,
        error: dict[str, str] | None,
    ) -> dict[str, Any]:
        run_report = {
            "run_id": manifest["run_id"],
            "config_hash": manifest["config_hash"],
            "timestamp": manifest["timestamp"],
            "stage_yields": manifest["stage_stats"],
            "final_count": manifest["final_count"],
            "dataset_path": str(dataset_path) if dataset_path is not None else None,
            "drop_reasons": self._aggregate_drop_reasons(stage_stats),
            "quality_distribution": self._report_quality_distribution(stage_stats),
            "status": status,
        }
        if error is not None:
            run_report["error"] = error
        return run_report

    def _serialize_error(
        self, stage_name: str | None, error: StageErrorInfo | None
    ) -> dict[str, str] | None:
        if stage_name is None or error is None:
            return None
        return {
            "stage": stage_name,
            "type": error.type,
            "message": error.message,
        }
