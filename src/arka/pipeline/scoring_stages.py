"""Score-only stages that annotate records without filtering."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from arka.config.models import resolve_llm_override
from arka.labeling.engine import LabelingEngine
from arka.labeling.rubric import RubricLoader
from arka.llm.client import LLMClient
from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.pipeline.stages import Stage
from arka.records.models import ConversationRecord, Record, RecordScores, StageEvent


class LabelingScoreStage(Stage):
    """Score records with the LabelingEngine rubric judge without dropping any.

    Unlike ``LabelingQualityFilterStage``, this stage annotates every
    scoreable record and returns ALL of them — no threshold filtering.
    Downstream filter or select stages can consume the scores.
    """

    name = "02s_label_score"
    stage_action = "scored"

    def __init__(self, project_root: Path, llm_client: Any | None = None) -> None:
        self.project_root = project_root
        self._llm_client = llm_client

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        filter_config = ctx.config.filters.labeling_engine
        if not filter_config.enabled or filter_config.rubric_path is None:
            return records

        rubric_path = self.project_root / filter_config.rubric_path
        if not rubric_path.exists():
            raise ValueError(
                f"filters.labeling_engine.rubric_path points to a missing file: "
                f"{rubric_path}"
            )
        rubric = RubricLoader().load(rubric_path)
        llm_client = self._llm_client or LLMClient(config=ctx.config.llm)
        engine = LabelingEngine(llm_client=llm_client)

        conversation_records: list[ConversationRecord] = [
            record for record in records if isinstance(record, ConversationRecord)
        ]
        if not conversation_records:
            self._write_artifacts(ctx=ctx, scored_count=0, scored_overall=[])
            return records

        pairs = [
            (record.payload.instruction, record.payload.response)
            for record in conversation_records
        ]
        results = engine.label_batch(
            pairs=pairs,
            rubric=rubric,
            max_workers=ctx.max_workers,
        )

        result_by_id = {
            record.id: result
            for record, result in zip(conversation_records, results, strict=True)
        }

        scored_records: list[Record] = []
        scored_overall: list[float] = []

        for record in records:
            if not isinstance(record, ConversationRecord):
                scored_records.append(record)
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
            scored_records.append(updated_record)

        self._write_artifacts(
            ctx=ctx,
            scored_count=len(conversation_records),
            scored_overall=scored_overall,
        )
        return scored_records

    def _write_artifacts(
        self,
        *,
        ctx: StageContext,
        scored_count: int,
        scored_overall: list[float],
    ) -> None:
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        stats = {
            "stage": self.name,
            "count_in": scored_count,
            "count_out": scored_count,
            "scored_count": scored_count,
            "quality_distribution": self._quality_distribution(scored_overall),
        }
        (ctx.work_dir / "stats.json").write_text(json.dumps(stats, indent=2))

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


class RewardModelScoringStage(Stage):
    """Score records with a reward model endpoint.

    Calls an LLM/reward model that returns a scalar score for each record.
    Stores the score in ``RecordScores.reward_model``.
    Optionally filters by ``min_score``.
    """

    name = "02r_reward_score"
    stage_action = "scored"

    def __init__(self, llm_client: Any | None = None) -> None:
        self._llm_client = llm_client
        self._output_writer = OutputWriter()

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        reward_config = ctx.config.filters.reward_model
        if not reward_config.enabled:
            return records

        effective_llm_config = resolve_llm_override(
            ctx.config.llm, reward_config.llm_override
        )
        llm_client = self._llm_client or LLMClient(config=effective_llm_config)

        kept_records: list[Record] = []
        dropped_records: list[Record] = []
        drop_reasons: dict[str, int] = {}
        scores: list[float] = []

        for record in records:
            if not isinstance(record, ConversationRecord):
                kept_records.append(record)
                continue

            messages = [
                {"role": "user", "content": record.payload.instruction},
                {"role": "assistant", "content": record.payload.response},
            ]
            output = llm_client.complete(messages=messages)
            try:
                score = float(output.text.strip())
            except (ValueError, AttributeError) as exc:
                raise ValueError(
                    f"Reward model returned non-numeric response: {output.text!r}"
                ) from exc

            scores.append(score)
            updated_record = record.model_copy(
                update={
                    "scores": record.scores.model_copy(
                        update={"reward_model": score}
                    )
                }
            )

            if (
                reward_config.min_score is not None
                and score < reward_config.min_score
            ):
                reason_code = "low_reward_score"
                dropped_records.append(
                    self._drop_record(
                        record=updated_record,
                        reason_code=reason_code,
                        details=(
                            f"reward_model={score} < "
                            f"min_score={reward_config.min_score}"
                        ),
                    )
                )
                drop_reasons[reason_code] = drop_reasons.get(reason_code, 0) + 1
            else:
                kept_records.append(updated_record)

        self._write_artifacts(
            ctx=ctx,
            dropped_records=dropped_records,
            count_in=len(scores),
            count_out=len(kept_records),
            drop_reasons=drop_reasons,
            scores=scores,
        )
        return kept_records

    def _drop_record(self, record: Record, reason_code: str, details: str) -> Record:
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

    def _write_artifacts(
        self,
        *,
        ctx: StageContext,
        dropped_records: list[Record],
        count_in: int,
        count_out: int,
        drop_reasons: dict[str, int],
        scores: list[float],
    ) -> None:
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        self._output_writer.write_dropped_parquet(
            records=dropped_records,
            path=ctx.work_dir / "dropped.parquet",
        )
        stats = {
            "stage": self.name,
            "count_in": count_in,
            "count_out": count_out,
            "dropped_count": len(dropped_records),
            "drop_reasons": drop_reasons,
            "reward_distribution": self._score_distribution(scores),
        }
        (ctx.work_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    def _score_distribution(self, scores: list[float]) -> dict[str, float] | None:
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


class PairDeltaFilterStage(Stage):
    """Filter records by score improvement over their parent.

    Compares a configurable score field between a child record and its
    parent (resolved via ``lineage.parent_ids``). Records with insufficient
    delta are dropped. Records without a matching parent pass through.
    """

    name = "04_pair_delta"
    stage_action = "filtered"

    def __init__(self) -> None:
        self._output_writer = OutputWriter()

    def run(
        self,
        records: list[Record],
        ctx: StageContext,
        *,
        parent_records: list[Record] | None = None,
    ) -> list[Record]:
        pair_config = ctx.config.filters.pair_delta
        if not pair_config.enabled:
            return records

        parent_by_id: dict[str, Record] = {}
        if parent_records:
            parent_by_id = {record.id: record for record in parent_records}

        kept_records: list[Record] = []
        dropped_records: list[Record] = []
        drop_reasons: dict[str, int] = {}

        for record in records:
            parent = self._find_parent(record, parent_by_id)
            if parent is None:
                kept_records.append(record)
                continue

            child_score = self._get_score(record, pair_config.score_field)
            parent_score = self._get_score(parent, pair_config.score_field)

            if child_score is None or parent_score is None:
                kept_records.append(record)
                continue

            # Check length ratio
            if pair_config.length_ratio_max is not None:
                child_len = self._text_length(record)
                parent_len = self._text_length(parent)
                if parent_len > 0:
                    ratio = child_len / parent_len
                    if ratio > pair_config.length_ratio_max:
                        reason = "length_ratio_exceeded"
                        dropped_records.append(
                            self._drop_record(
                                record, reason,
                                f"ratio={ratio:.2f} > max={pair_config.length_ratio_max}",
                            )
                        )
                        drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                        continue

            # Check delta
            delta = child_score - parent_score
            if delta < pair_config.min_delta:
                reason = "insufficient_delta"
                dropped_records.append(
                    self._drop_record(
                        record, reason,
                        f"delta={delta:.4f} < min_delta={pair_config.min_delta}",
                    )
                )
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
            else:
                kept_records.append(record)

        self._write_artifacts(
            ctx=ctx,
            dropped_records=dropped_records,
            count_in=len(records),
            count_out=len(kept_records),
            drop_reasons=drop_reasons,
        )
        return kept_records

    def _find_parent(
        self, record: Record, parent_by_id: dict[str, Record]
    ) -> Record | None:
        for parent_id in record.lineage.parent_ids:
            if parent_id in parent_by_id:
                return parent_by_id[parent_id]
        return None

    def _get_score(self, record: Record, field: str) -> float | None:
        return getattr(record.scores, field, None)

    def _text_length(self, record: Record) -> int:
        if isinstance(record, ConversationRecord):
            return len(record.payload.response)
        return 0

    def _drop_record(self, record: Record, reason_code: str, details: str) -> Record:
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

    def _write_artifacts(
        self,
        *,
        ctx: StageContext,
        dropped_records: list[Record],
        count_in: int,
        count_out: int,
        drop_reasons: dict[str, int],
    ) -> None:
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        self._output_writer.write_dropped_parquet(
            records=dropped_records,
            path=ctx.work_dir / "dropped.parquet",
        )
        stats = {
            "stage": self.name,
            "count_in": count_in,
            "count_out": count_out,
            "dropped_count": len(dropped_records),
            "drop_reasons": drop_reasons,
        }
        (ctx.work_dir / "stats.json").write_text(json.dumps(stats, indent=2))


class CompositeSelectStage(Stage):
    """Select top-N records by weighted composite score.

    Combines multiple score fields (quality, reward_model, ifd, etc.)
    with configurable weights and keeps the top ``target_count`` records.
    """

    name = "05_composite_select"
    stage_action = "selected"

    def __init__(self) -> None:
        self._output_writer = OutputWriter()

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        select_config = ctx.config.filters.select
        if not select_config.enabled:
            return records

        weights = select_config.weights
        if not weights:
            return records

        target_count = select_config.target_count or len(records)
        if target_count >= len(records):
            return records

        scored_pairs: list[tuple[float, Record]] = []
        for record in records:
            composite = self._composite_score(record, weights)
            scored_pairs.append((composite, record))

        scored_pairs.sort(key=lambda pair: pair[0], reverse=True)
        kept = [record for _, record in scored_pairs[:target_count]]

        dropped_with_events = [
            self._drop_record(
                record, "composite_select",
                f"rank={i + target_count + 1} exceeds target_count={target_count}",
            )
            for i, (_, record) in enumerate(
                scored_pairs[target_count:]
            )
        ]

        self._write_artifacts(
            ctx=ctx,
            dropped_records=dropped_with_events,
            count_in=len(records),
            count_out=len(kept),
        )
        return kept

    def _composite_score(
        self, record: Record, weights: dict[str, float]
    ) -> float:
        total = 0.0
        for field, weight in weights.items():
            value = getattr(record.scores, field, None)
            if value is None:
                value = 0.0
            total += float(value) * weight
        return total

    def _drop_record(self, record: Record, reason_code: str, details: str) -> Record:
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

    def _write_artifacts(
        self,
        *,
        ctx: StageContext,
        dropped_records: list[Record],
        count_in: int,
        count_out: int,
    ) -> None:
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        self._output_writer.write_dropped_parquet(
            records=dropped_records,
            path=ctx.work_dir / "dropped.parquet",
        )
        stats = {
            "stage": self.name,
            "count_in": count_in,
            "count_out": count_out,
            "dropped_count": len(dropped_records),
            "drop_reasons": {"composite_select": len(dropped_records)}
            if dropped_records
            else {},
        }
        (ctx.work_dir / "stats.json").write_text(json.dumps(stats, indent=2))
