from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from arka.config.loader import ConfigLoader
from arka.config.models import ResolvedConfig
from arka.core.paths import RunPaths
from arka.decisions.factory import DecisionClientFactory
from arka.pipeline.artifacts import StageArtifacts, get_extra
from arka.pipeline.checkpoint import CheckpointManager
from arka.pipeline.models import RunResult, StageContext, StageErrorInfo, StageStat
from arka.pipeline.output import OutputWriter
from arka.pipeline.reporter import RunReporter
from arka.pipeline.stages import Stage
from arka.records.models import Record, StageEvent

logger = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(
        self,
        project_root: Path,
        *,
        decision_client_factory: DecisionClientFactory | None = None,
    ) -> None:
        self.project_root = project_root
        self.decision_client_factory = decision_client_factory
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

        # PERF: Reusing a single ThreadPoolExecutor across all pipeline stages to avoid the overhead of repeatedly creating and destroying thread pools per stage/batch.
        try:
            with ThreadPoolExecutor(
                max_workers=resolved_config.executor.max_workers
            ) as shared_executor:
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
                        executor=shared_executor,
                        checkpoint_manager=checkpoint_manager,
                        decision_client_factory=self.decision_client_factory,
                    )
                    count_in = len(records)
                    # DX: Provide per-stage progress indication during long runs
                    print(
                        f"\033[36mRunning stage {i}/{len(stages)}: {stage.name} ({count_in} records in)...\033[0m"
                    )
                    stage_start_time = time.time()
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
                                count_out=0,
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
                            count_out=0,
                            status="failed",
                        )
                        # DX: Wrap stage failures with the stage name to improve error visibility in CLI
                        record_str = "record was" if count_in == 1 else "records were"
                        raise RuntimeError(
                            f"Stage '{stage.name}' failed - {exc}. {count_in} {record_str} lost."
                        ) from exc

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
                    stage_duration = time.time() - stage_start_time
                    # DX: Provide clear completion feedback and timing for each stage so developers can monitor performance during long runs
                    print(
                        f"\033[32mCompleted stage {stage.name} in {stage_duration:.1f}s ({len(records)} records out)\033[0m"
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
            checkpoint_manager.close()
            raise

        decision_report = self._decision_report(records, stages, run_paths)
        if resolved_config.output.on_unresolved_escalation == "drop":
            unresolved = [
                r.dropped_by("output", "unresolved_escalation")
                for r in records
                if r.is_escalated()
            ]
            if unresolved:
                self.output_writer.write_dropped_parquet(
                    records=unresolved,
                    path=run_paths.run_dir / "unresolved_escalations.parquet",
                )
            records = [r for r in records if not r.is_escalated()]
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
        manifest.update(decision_report)
        run_report.update(decision_report)
        run_paths.manifest_path.write_text(json.dumps(manifest, indent=2))
        run_paths.run_report_path.write_text(json.dumps(run_report, indent=2))
        checkpoint_manager.update_run_status(run_id=run_id, status="completed")
        checkpoint_manager.close()

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

    def _decision_report(
        self, records: list[Record], stages: list[Stage], run_paths: RunPaths
    ) -> dict[str, Any]:
        import polars as pl

        usage_by_stage = {}
        escalation_rows = []
        seen = set()
        for stage in stages:
            report = StageArtifacts.load_report(run_paths.stage_stats_path(stage.name))
            usage = get_extra(report, "decision_usage")
            if usage is not None:
                usage_by_stage[stage.name] = usage
            # Include escalations on records subsequently dropped as well as survivors.
            path = run_paths.stage_data_path(stage.name)
            if path.exists():
                stage_records = self.output_writer.read_parquet(path)
                for record in stage_records:
                    for event in record.stage_events:
                        key = (record.id, event.stage, event.seq)
                        if event.action != "escalated" or key in seen:
                            continue
                        seen.add(key)
                        details = {}
                        if event.details:
                            try:
                                details = json.loads(event.details)
                            except (ValueError, TypeError):
                                pass
                        answers = details if isinstance(details, dict) else {}
                        answer_items = [
                            (name, value)
                            for name, value in answers.items()
                            if isinstance(value, dict) and "type" in value
                        ]
                        question, answer = min(
                            answer_items,
                            key=lambda item: item[1].get(
                                "confidence",
                                abs(2 * item[1].get("probability", 0.5) - 1),
                            ),
                            default=(None, {}),
                        )
                        probability = answer.get("probability")
                        confidence = answer.get("confidence")
                        if answer.get("type") == "yes_no":
                            confidence = abs(2 * probability - 1)
                        escalation_rows.append(
                            {
                                "record_id": record.id,
                                "stage": event.stage,
                                "reason_code": event.reason_code,
                                "details": event.details,
                                "question": question,
                                "probability": probability,
                                "confidence": confidence,
                                "resolved": False,
                            }
                        )
        if not usage_by_stage and not escalation_rows:
            return {}
        schema = {
            "record_id": pl.String,
            "stage": pl.String,
            "reason_code": pl.String,
            "details": pl.String,
            "question": pl.String,
            "probability": pl.Float64,
            "confidence": pl.Float64,
            "resolved": pl.Boolean,
        }
        pl.DataFrame(escalation_rows, schema=schema).write_parquet(
            run_paths.run_dir / "escalations.parquet"
        )
        costs = [
            u["cost_usd"]
            for u in usage_by_stage.values()
            if u.get("cost_usd") is not None
        ]
        return {
            "decision_usage": {
                "requests": sum(u.get("requests", 0) for u in usage_by_stage.values()),
                "input_tokens": sum(
                    u.get("input_tokens", 0) for u in usage_by_stage.values()
                ),
                "cache_hits": sum(
                    u.get("cache_hits", 0) for u in usage_by_stage.values()
                ),
                "cost_usd": sum(costs) if costs else None,
                "by_stage": usage_by_stage,
            },
            "escalation_counts": {"unresolved": len(escalation_rows), "resolved": 0},
        }

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
