from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from arka.labeling.engine import LabelingEngine
from arka.labeling.rubric import RubricLoader
from arka.llm.client import (
    LLMClient,
    LLMClientError,
    provider_supports_sequence_scoring,
)
from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.pipeline.stages import Stage
from arka.records.models import (
    ConversationRecord,
    Record,
    RecordScores,
)


class CanaryFilterStage(Stage):
    """Drop records whose text contains any configured canary phrase."""

    name = "02g_canary_filter"
    stage_action = "filtered"

    def __init__(self) -> None:
        self._output_writer = OutputWriter()

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        filter_config = ctx.config.filters.get_stage_config("canary")
        if filter_config is None or not filter_config.phrases:
            return records

        kept: list[Record] = []
        dropped: list[Record] = []
        drop_reasons: dict[str, int] = {}

        for record in records:
            if not isinstance(record, ConversationRecord):
                kept.append(record)
                continue

            text = f"{record.payload.instruction}\n{record.payload.response}"
            matched = next((p for p in filter_config.phrases if p in text), None)

            if matched is None:
                kept.append(record)
            else:
                reason = "canary_leak"
                dropped.append(
                    record.dropped_by(
                        self.name, reason, f"Matched canary phrase: {matched}"
                    )
                )
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

        _write_filter_artifacts(
            self._output_writer,
            ctx,
            self.name,
            len(records),
            len(kept),
            dropped,
            drop_reasons,
        )
        return kept


class SemanticSimilarityFilterStage(Stage):
    """Drop generated records too similar to their seed records."""

    name = "02h_semantic_similarity_filter"
    stage_action = "filtered"

    def __init__(self) -> None:
        self._output_writer = OutputWriter()

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        filter_config = ctx.config.filters.get_stage_config("semantic_similarity")
        if filter_config is None:
            return records

        generated: list[ConversationRecord] = []
        seeds: list[ConversationRecord] = []
        other: list[Record] = []

        for record in records:
            if not isinstance(record, ConversationRecord):
                other.append(record)
                continue
            if record.source.type == "seed":
                seeds.append(record)
            else:
                generated.append(record)

        if not seeds or not generated:
            return records

        from arka.pipeline.runner import PipelineRunner

        runner = PipelineRunner(project_root=ctx.work_dir)
        gen_texts = [
            f"{r.payload.instruction}\n{r.payload.response}" for r in generated
        ]
        seed_texts = [f"{r.payload.instruction}\n{r.payload.response}" for r in seeds]

        gen_emb = runner._embed_texts(
            config=ctx.config,
            texts=gen_texts,
            checkpoint_manager=ctx.checkpoint_manager,
        )
        seed_emb = runner._embed_texts(
            config=ctx.config,
            texts=seed_texts,
            checkpoint_manager=ctx.checkpoint_manager,
        )

        if gen_emb is None or seed_emb is None:
            return records

        # Cosine similarity matrix
        gen_norm = gen_emb / np.maximum(
            np.linalg.norm(gen_emb, axis=1, keepdims=True), 1e-9
        )
        seed_norm = seed_emb / np.maximum(
            np.linalg.norm(seed_emb, axis=1, keepdims=True), 1e-9
        )
        sim_matrix = gen_norm @ seed_norm.T

        kept: list[Record] = list(seeds) + list(other)
        dropped: list[Record] = []
        drop_reasons: dict[str, int] = {}

        for i, record in enumerate(generated):
            max_sim = float(np.max(sim_matrix[i]))
            if max_sim > filter_config.threshold:
                reason = "high_semantic_similarity"
                dropped.append(
                    record.dropped_by(
                        self.name,
                        reason,
                        f"Max cosine similarity {max_sim:.4f} > {filter_config.threshold}",
                    )
                )
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
            else:
                kept.append(record)

        _write_filter_artifacts(
            self._output_writer,
            ctx,
            self.name,
            len(records),
            len(kept),
            dropped,
            drop_reasons,
        )
        return kept


def _write_filter_artifacts(
    writer: OutputWriter,
    ctx: StageContext,
    stage_name: str,
    count_in: int,
    count_out: int,
    dropped: list[Record],
    drop_reasons: dict[str, int],
) -> None:
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    if dropped:
        writer.write_dropped_parquet(
            records=dropped, path=ctx.work_dir / "dropped.parquet"
        )
    stats = {
        "stage": stage_name,
        "count_in": count_in,
        "count_out": count_out,
        "dropped_count": len(dropped),
        "drop_reasons": drop_reasons,
    }
    (ctx.work_dir / "stats.json").write_text(json.dumps(stats, indent=2))


class LabelingQualityFilterStage(Stage):
    name = "03_label_quality"
    stage_action = "filtered"

    def __init__(self, project_root: Path, llm_client: Any | None = None) -> None:
        self.project_root = project_root
        self._llm_client = llm_client
        self._output_writer = OutputWriter()

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        filter_config = ctx.config.filters.get_stage_config("labeling_engine")
        if filter_config is None or filter_config.rubric_path is None:
            return records
        rubric_path = self.project_root / filter_config.rubric_path
        try:
            rubric = RubricLoader().load(rubric_path)
        except FileNotFoundError as exc:
            raise ValueError(
                "filters.labeling_engine.rubric_path points to a missing file: "
                f"{rubric_path}"
            ) from exc
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
                    record.dropped_by(
                        self.name,
                        reason_code,
                        exc.message,
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
                updated_record.dropped_by(
                    self.name,
                    reason_code,
                    f"overall_score={result.overall} < min_overall_score={min_overall}",
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


def validate_ifd_capability(ctx: StageContext) -> None:
    filter_config = ctx.config.filters.get_stage_config("ifd")
    if filter_config is None:
        return
    if not provider_supports_sequence_scoring(ctx.config.llm):
        raise ValueError(
            "IFD requires provider/model response-scoring capability; "
            f"unsupported for provider={ctx.config.llm.provider!r} "
            f"model={ctx.config.llm.model!r}"
        )
