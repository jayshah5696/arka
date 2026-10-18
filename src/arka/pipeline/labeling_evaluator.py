"""Shared labeling evaluation helper for filtering and scoring stages."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any

from arka.config.models import LabelingFilterConfig
from arka.labeling.engine import LabelingEngine
from arka.labeling.rubric import RubricLoader
from arka.llm.client import LLMClientError
from arka.pipeline.artifacts import StageArtifacts, StageReport
from arka.records.models import ConversationRecord, Record

if TYPE_CHECKING:
    from arka.pipeline.models import StageContext
    from arka.pipeline.stages import Stage

__all__ = [
    "evaluate_labeling",
    "quality_distribution",
    "reason_code_for_label_error",
]


def quality_distribution(scores: list[float]) -> dict[str, float] | None:
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


def reason_code_for_label_error(error: LLMClientError) -> str:
    if error.code == "auth_error":
        return "label_auth_failure"
    if error.code == "retryable_api_error":
        return "label_retryable_api_failure"
    if error.code == "invalid_structured_response":
        return "invalid_structured_response"
    return "label_parse_failure"


def evaluate_labeling(
    stage: Stage,
    records: list[Record],
    ctx: StageContext,
    *,
    config: LabelingFilterConfig | None = None,
    project_root: Path,
    llm_client: Any | None = None,
    drop_below_threshold: bool = True,
    missing_file_exc_cls: type[Exception] = ValueError,
) -> list[Record]:
    """Execute rubric-based LLM labeling for either filtering or scoring.

    When drop_below_threshold is True (filtering), records falling below
    min_overall_score are dropped and recorded to dropped.parquet.
    When drop_below_threshold is False (scoring), all scored records are preserved.
    """
    if config is None or config.rubric_path is None:
        return records

    filter_config = config
    if filter_config.backend == "decision":
        from arka.pipeline.decision_stages import run_decision_stage

        decision_cfg = (
            filter_config
            if drop_below_threshold
            else filter_config.model_copy(
                update={"min_overall_score": None, "on_uncertain": "keep"}
            )
        )
        return run_decision_stage(stage, records, ctx, decision_cfg)

    rubric_path = project_root / filter_config.rubric_path
    if not rubric_path.exists():
        raise missing_file_exc_cls(
            f"filters.labeling_engine.rubric_path points to a missing file: {rubric_path}"
        )
    rubric = RubricLoader().load(rubric_path)
    client = llm_client or ctx.llm_client()
    engine = LabelingEngine(llm_client=client)

    conversation_records, passthrough = stage.partition_conversations(records)
    if not conversation_records:
        if not drop_below_threshold:
            StageArtifacts(ctx).write(
                report=StageReport(
                    stage=stage.name,
                    count_in=0,
                    count_out=0,
                    scored_count=0,
                    quality_distribution=None,
                )
            )
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
            executor=getattr(ctx, "executor", None),
        )
    except LLMClientError as exc:
        if not drop_below_threshold:
            raise
        reason_code = reason_code_for_label_error(exc)
        kept_records = list(passthrough)
        for record in conversation_records:
            dropped_records.append(
                record.dropped_by(
                    stage.name,
                    reason_code,
                    exc.message,
                )
            )
        drop_reasons[reason_code] = len(conversation_records)
        StageArtifacts(ctx).write(
            report=StageReport(
                stage=stage.name,
                count_in=len(conversation_records),
                count_out=len(kept_records),
                dropped_count=len(dropped_records),
                drop_reasons=drop_reasons,
                scored_count=0,
                quality_distribution=None,
            ),
            dropped=dropped_records,
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
                "scores": record.scores.model_copy(
                    update={
                        "quality": result.overall,
                        "quality_per_dim": {
                            **record.scores.quality_per_dim,
                            **result.scores,
                        },
                        "rubric_hash": result.rubric_hash,
                        "rubric_version": result.rubric_version,
                        "judge_model": result.judge_model,
                        "judge_prompt_hash": result.judge_prompt_hash,
                    }
                )
            }
        )
        if not drop_below_threshold or result.overall >= min_overall:
            kept_records.append(updated_record)
            continue

        reason_code = "low_quality_score"
        dropped_records.append(
            updated_record.dropped_by(
                stage.name,
                reason_code,
                f"overall_score={result.overall} < min_overall_score={min_overall}",
            )
        )
        drop_reasons[reason_code] = drop_reasons.get(reason_code, 0) + 1

    if drop_below_threshold:
        StageArtifacts(ctx).write(
            report=StageReport(
                stage=stage.name,
                count_in=len(conversation_records),
                count_out=len(kept_records),
                dropped_count=len(dropped_records),
                drop_reasons=drop_reasons,
                scored_count=len(conversation_records),
                quality_distribution=quality_distribution(scored_overall),
            ),
            dropped=dropped_records,
        )
    else:
        StageArtifacts(ctx).write(
            report=StageReport(
                stage=stage.name,
                count_in=len(conversation_records),
                count_out=len(kept_records),
                scored_count=len(conversation_records),
                quality_distribution=quality_distribution(scored_overall),
            ),
        )

    return kept_records
