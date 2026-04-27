"""RunReporter: build the per-Run Manifest and run report Artifacts.

Lifted out of ``PipelineRunner`` (~250 lines) so the runner's remaining
responsibility stays small: orchestrate Stages under Checkpoint semantics.

This module owns:

  - The Manifest schema (``stage_names``, ``stage_stats``, ``status``,
    ``final_count``, ``dataset_path``, ``error``).
  - The run report schema (samples, canaries, diversity score, drop-reason
    aggregation, cost roll-up, per-stage StageStat serialisation).
  - The samples and canaries sub-Artifacts (``report/samples.jsonl`` and
    ``report/canaries.json``).

A future caller that only wants a Manifest -- e.g. an ``arka manifest
<run_id>`` command, or a dashboard reading completed runs -- can use
:class:`RunReporter` without booting the full PipelineRunner.

Schema field names are unchanged; downstream consumers (tests, the run
report viewer) read the JSON directly.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from arka.config.models import ResolvedConfig
from arka.embeddings import Embedder
from arka.labeling.rubric import RubricLoader
from arka.pipeline.models import StageErrorInfo, StageStat
from arka.pipeline.output import OutputWriter
from arka.pipeline.stages import Stage
from arka.records.models import Record

if TYPE_CHECKING:
    from arka.pipeline.checkpoint import CheckpointManager


class RunReporter:
    """Builder for the Manifest + run report at the end of a Run.

    Construct once per Run with the project root and the OutputWriter the
    runner already uses; call :meth:`build_manifest` and
    :meth:`build_run_report` (or :meth:`write_all` for the typical case).
    """

    def __init__(
        self, project_root: Path, output_writer: OutputWriter | None = None
    ) -> None:
        self._project_root = project_root
        self._output_writer = output_writer or OutputWriter()

    # --- Top-level builders ---

    def build_manifest(
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
        manifest: dict[str, Any] = {
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

    def build_run_report(
        self,
        *,
        manifest: dict[str, Any],
        stage_stats: list[StageStat],
        dataset_path: Path | None,
        status: str,
        error: dict[str, str] | None,
        report_dir: Path,
        records: list[Record],
        config: ResolvedConfig,
        checkpoint_manager: CheckpointManager | None = None,
    ) -> dict[str, Any]:
        report_dir.mkdir(parents=True, exist_ok=True)
        samples_path = self._write_samples(
            records, report_dir / "samples.jsonl", config
        )
        canaries_path = report_dir / "canaries.json"
        canaries = self._build_canaries(config=config, report_path=canaries_path)
        diversity_score = Embedder(config).compute_diversity_score(
            records=records,
            checkpoint_manager=checkpoint_manager,
        )

        stage_costs = [
            stage_stat.cost_usd
            for stage_stat in stage_stats
            if stage_stat.cost_usd is not None
        ]
        total_cost = round(sum(stage_costs), 6) if stage_costs else None
        run_report: dict[str, Any] = {
            "run_id": manifest["run_id"],
            "config_hash": manifest["config_hash"],
            "timestamp": manifest["timestamp"],
            "stage_yields": manifest["stage_stats"],
            "final_count": manifest["final_count"],
            "dataset_path": str(dataset_path) if dataset_path is not None else None,
            "samples_path": str(samples_path),
            "canaries_path": str(canaries_path),
            "drop_reasons": self._aggregate_drop_reasons(stage_stats),
            "quality_distribution": self._report_quality_distribution(stage_stats),
            "diversity_score": diversity_score,
            "canaries": canaries,
            "cost_usd": total_cost,
            "status": status,
        }
        if error is not None:
            run_report["error"] = error
        return run_report

    # --- StageStat / StageErrorInfo serialisation ---

    def _serialize_stage_stat(self, stage_stat: StageStat) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
        if stage_stat.cost_usd is not None:
            payload["cost_usd"] = stage_stat.cost_usd
        return payload

    @staticmethod
    def serialize_error(
        stage_name: str | None, error: StageErrorInfo | None
    ) -> dict[str, str] | None:
        if stage_name is None or error is None:
            return None
        return {
            "stage": stage_name,
            "type": error.type,
            "message": error.message,
        }

    # --- Aggregations across StageStat list ---

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

    # --- Samples + canaries sub-artifacts ---

    def _write_samples(
        self,
        records: list[Record],
        path: Path,
        config: ResolvedConfig,
    ) -> Path:
        rng = random.Random(0)
        samples = list(records)
        if len(samples) > 20:
            samples = rng.sample(samples, 20)
        return self._output_writer.write_jsonl(
            records=samples,
            path=path,
            output_format=config.output.format,
        )

    def _build_canaries(
        self,
        *,
        config: ResolvedConfig,
        report_path: Path,
    ) -> dict[str, Any]:
        if report_path.exists():
            return json.loads(report_path.read_text())

        filter_cfg = config.filters.get_stage_config("labeling_engine")
        rubric_path_value = (
            filter_cfg.rubric_path if filter_cfg is not None else None
        ) or config.labeling_engine.rubric_path
        empty_payload: dict[str, Any] = {
            "known_good": [],
            "known_bad": [],
            "status": None,
        }
        if filter_cfg is None or rubric_path_value is None:
            report_path.write_text(json.dumps(empty_payload, indent=2))
            return empty_payload

        rubric_path = Path(rubric_path_value)
        if not rubric_path.is_absolute():
            rubric_path = self._project_root / rubric_path
        rubric = RubricLoader().load(rubric_path)
        if len(rubric.few_shot) < 2:
            report_path.write_text(json.dumps(empty_payload, indent=2))
            return empty_payload

        good = next(
            (
                example
                for example in rubric.few_shot
                if example.expected_verdict == "pass"
            ),
            None,
        )
        bad = next(
            (
                example
                for example in rubric.few_shot
                if example.expected_verdict == "fail"
            ),
            None,
        )
        if good is None or bad is None:
            report_path.write_text(json.dumps(empty_payload, indent=2))
            return empty_payload
        good_score = self._weighted_score(good.scores, rubric.overall_weights)
        bad_score = self._weighted_score(bad.scores, rubric.overall_weights)
        payload = {
            "known_good": [
                {
                    "id": f"few_shot_{rubric.few_shot.index(good)}",
                    "expected": "high",
                    "actual_score": good_score,
                }
            ],
            "known_bad": [
                {
                    "id": f"few_shot_{rubric.few_shot.index(bad)}",
                    "expected": "low",
                    "actual_score": bad_score,
                }
            ],
            "status": "pass" if bad_score < good_score else "warn",
        }
        report_path.write_text(json.dumps(payload, indent=2))
        return payload

    @staticmethod
    def _weighted_score(scores: dict[str, int], weights: dict[str, float]) -> float:
        total = sum(scores[name] * weights[name] for name in weights)
        return round(float(total), 4)
