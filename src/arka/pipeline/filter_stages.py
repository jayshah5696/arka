from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from arka.labeling.engine import LabelingEngine
from arka.labeling.rubric import RubricLoader
from arka.llm.client import LLMClient, LLMClientError
from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.pipeline.stages import Stage
from arka.records.models import (
    ConversationRecord,
    Record,
    RecordScores,
    StageEvent,
)


class LabelingQualityFilterStage(Stage):
    name = "03_label_quality"

    def __init__(self, project_root: Path, llm_client: Any | None = None) -> None:
        self.project_root = project_root
        self._llm_client = llm_client
        self._output_writer = OutputWriter()

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        filter_config = ctx.config.filters.labeling_engine
        if not filter_config.enabled or filter_config.rubric_path is None:
            return records
        rubric_path = self.project_root / filter_config.rubric_path
        rubric = RubricLoader().load(rubric_path)
        llm_client = self._llm_client or LLMClient(config=ctx.config.llm)
        engine = LabelingEngine(llm_client=llm_client)

        conversation_records: list[ConversationRecord] = [
            record for record in records if isinstance(record, ConversationRecord)
        ]
        if not conversation_records:
            return records

        pairs = [
            (record.payload.instruction, record.payload.response)
            for record in conversation_records
        ]
        min_overall = filter_config.min_overall_score or 0.0
        kept_records: list[Record] = []
        dropped_records: list[Record] = []
        scored_overall: list[float] = []
        drop_reasons: dict[str, int] = {}

        try:
            results = engine.label_batch(
                pairs=pairs,
                rubric=rubric,
                max_workers=ctx.max_workers,
            )
        except LLMClientError as exc:
            reason_code = self._reason_code_for_label_error(exc)
            kept_records = [
                record
                for record in records
                if not isinstance(record, ConversationRecord)
            ]
            for record in conversation_records:
                dropped_records.append(
                    self._drop_record(
                        record=record,
                        reason_code=reason_code,
                        details=exc.message,
                    )
                )
            drop_reasons[reason_code] = len(conversation_records)
            self._write_stage_artifacts(
                ctx=ctx,
                dropped_records=dropped_records,
                scored_count=0,
                kept_count=len(kept_records),
                dropped_count=len(dropped_records),
                drop_reasons=drop_reasons,
                scored_overall=scored_overall,
            )
            return kept_records

        result_by_id = {
            record.id: result
            for record, result in zip(conversation_records, results, strict=True)
        }

        for record in records:
            if not isinstance(record, ConversationRecord):
                kept_records.append(record)
                continue
            result = result_by_id[record.id]
            scored_overall.append(result.overall)
            updated_record = record.model_copy(
                update={
                    "scores": RecordScores(
                        quality=result.overall,
                        quality_per_dim=result.scores,
                        rubric_hash=result.rubric_hash,
                        rubric_version=result.rubric_version,
                        judge_model=result.judge_model,
                        judge_prompt_hash=result.judge_prompt_hash,
                    )
                }
            )
            if result.overall >= min_overall:
                kept_records.append(updated_record)
                continue

            reason_code = "low_quality_score"
            dropped_records.append(
                self._drop_record(
                    record=updated_record,
                    reason_code=reason_code,
                    details=(
                        f"overall_score={result.overall} < "
                        f"min_overall_score={min_overall}"
                    ),
                )
            )
            drop_reasons[reason_code] = drop_reasons.get(reason_code, 0) + 1

        self._write_stage_artifacts(
            ctx=ctx,
            dropped_records=dropped_records,
            scored_count=len(conversation_records),
            kept_count=len(kept_records),
            dropped_count=len(dropped_records),
            drop_reasons=drop_reasons,
            scored_overall=scored_overall,
        )
        return kept_records

    def _write_stage_artifacts(
        self,
        ctx: StageContext,
        dropped_records: list[Record],
        scored_count: int,
        kept_count: int,
        dropped_count: int,
        drop_reasons: dict[str, int],
        scored_overall: list[float],
    ) -> None:
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        dropped_path = ctx.work_dir / "dropped.parquet"
        self._write_dropped_records(dropped_records=dropped_records, path=dropped_path)
        stats = {
            "stage": self.name,
            "count_in": scored_count,
            "count_out": kept_count,
            "scored_count": scored_count,
            "dropped_count": dropped_count,
            "drop_reasons": drop_reasons,
            "quality_distribution": self._quality_distribution(scored_overall),
        }
        (ctx.work_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    def _write_dropped_records(self, dropped_records: list[Record], path: Path) -> None:
        self._output_writer.write_dropped_parquet(records=dropped_records, path=path)

    def _drop_record(
        self,
        record: Record,
        reason_code: str,
        details: str,
    ) -> Record:
        return record.model_copy(
            update={
                "stage_events": [
                    *record.stage_events,
                    StageEvent(
                        stage=self.name,
                        action="dropped",
                        reason_code=reason_code,
                        details=details,
                        seq=len(record.stage_events) + 1,
                    ),
                ]
            }
        )

    def _reason_code_for_label_error(self, error: LLMClientError) -> str:
        if error.code == "auth_error":
            return "label_auth_failure"
        if error.code == "retryable_api_error":
            return "label_retryable_api_failure"
        if error.code == "invalid_structured_response":
            return "invalid_structured_response"
        return "label_parse_failure"

    def _quality_distribution(self, scores: list[float]) -> dict[str, float] | None:
        if not scores:
            return None
        mean_score = statistics.fmean(scores)
        std_score = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        return {
            "mean": round(mean_score, 4),
            "std": round(std_score, 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        }
