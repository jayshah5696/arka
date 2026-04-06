from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from arka.labeling.engine import LabelingEngine
import numpy as np

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
    StageEvent,
)


class SemanticSimilarityFilterStage(Stage):
    name = "02h_semantic_similarity_filter"
    stage_action = "filtered"

    def __init__(self) -> None:
        self._hf_embedding_model = None
        self._openai_client = None

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        filter_config = ctx.config.filters.semantic_similarity
        if not filter_config.enabled:
            return records

        generated_records: list[ConversationRecord] = []
        seed_records: list[ConversationRecord] = []
        other_records: list[Record] = []

        for record in records:
            if not isinstance(record, ConversationRecord):
                other_records.append(record)
                continue
            if record.source.type == "seed":
                seed_records.append(record)
            elif record.source.type == "generated":
                generated_records.append(record)
            else:
                other_records.append(record)

        if not seed_records or not generated_records:
            return records

        generated_texts = [f"{r.payload.instruction}\n{r.payload.response}" for r in generated_records]
        seed_texts = [f"{r.payload.instruction}\n{r.payload.response}" for r in seed_records]

        try:
            generated_embeddings = self._embed_texts(config=ctx.config, texts=generated_texts)
            seed_embeddings = self._embed_texts(config=ctx.config, texts=seed_texts)
        except Exception as exc:
            raise RuntimeError(f"Semantic similarity embedding failed: {exc}") from exc

        if generated_embeddings is None or seed_embeddings is None:
            raise RuntimeError("Semantic similarity embedding failed: returned None")

        kept_records: list[Record] = list(seed_records) + list(other_records)
        dropped_records: list[Record] = []
        drop_reasons: dict[str, int] = {}

        generated_norms = np.linalg.norm(generated_embeddings, axis=1, keepdims=True)
        seed_norms = np.linalg.norm(seed_embeddings, axis=1, keepdims=True)

        generated_embeddings_normalized = generated_embeddings / np.where(generated_norms == 0, 1, generated_norms)
        seed_embeddings_normalized = seed_embeddings / np.where(seed_norms == 0, 1, seed_norms)

        similarities = np.dot(generated_embeddings_normalized, seed_embeddings_normalized.T)

        for i, record in enumerate(generated_records):
            max_sim = np.max(similarities[i])
            if max_sim > filter_config.threshold:
                reason_code = "high_semantic_similarity"
                dropped_records.append(
                    self._drop_record(
                        record=record,
                        reason_code=reason_code,
                        details=f"Max cosine similarity with seeds: {max_sim:.4f} > {filter_config.threshold}",
                    )
                )
                drop_reasons[reason_code] = drop_reasons.get(reason_code, 0) + 1
            else:
                kept_records.append(record)

        self._write_stage_artifacts(
            ctx=ctx,
            dropped_records=dropped_records,
            scored_count=len(generated_records),
            kept_count=len(kept_records),
            dropped_count=len(dropped_records),
            drop_reasons=drop_reasons,
        )
        return kept_records

    def _embed_texts(self, config: Any, texts: list[str]) -> np.ndarray | None:
        if config.embeddings.provider == "huggingface":
            model_name = config.embeddings.model
            if "/" not in model_name:
                model_name = f"sentence-transformers/{model_name}"

            if self._hf_embedding_model is None:
                from fastembed import TextEmbedding
                self._hf_embedding_model = TextEmbedding(model_name=model_name)

            vectors = list(self._hf_embedding_model.embed(texts))
            if not vectors:
                return None
            return np.array(vectors, dtype=float)
        else:
            if self._openai_client is None:
                from arka.config.models import LLMConfig
                from arka.llm.openai_client import build_openai_client

                embedding_cfg = config.embeddings
                api_key = embedding_cfg.api_key or config.llm.api_key
                base_url = embedding_cfg.base_url or config.llm.base_url
                timeout_seconds = (
                    embedding_cfg.timeout_seconds
                    if embedding_cfg.timeout_seconds is not None
                    else config.llm.timeout_seconds
                )
                max_retries = (
                    embedding_cfg.max_retries
                    if embedding_cfg.max_retries is not None
                    else config.llm.max_retries
                )
                openai_compatible = (
                    embedding_cfg.openai_compatible or config.llm.openai_compatible
                )
                llm_config = LLMConfig(
                    provider="openai",
                    model=embedding_cfg.model,
                    api_key=api_key,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                    openai_compatible=openai_compatible,
                )
                self._openai_client = build_openai_client(llm_config)

            response = self._openai_client.embeddings.create(
                model=config.embeddings.model,
                input=texts,
            )
            vectors = [item.embedding for item in response.data]
            if not vectors:
                return None
            return np.array(vectors, dtype=float)

    def _write_stage_artifacts(
        self,
        ctx: StageContext,
        dropped_records: list[Record],
        scored_count: int,
        kept_count: int,
        dropped_count: int,
        drop_reasons: dict[str, int],
    ) -> None:
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        if dropped_records:
            writer = OutputWriter()
            writer.write_dropped_parquet(
                records=dropped_records, path=ctx.work_dir / "dropped.parquet"
            )
        stats = {
            "stage": self.name,
            "count_in": scored_count,
            "count_out": kept_count,
            "dropped_count": dropped_count,
            "drop_reasons": drop_reasons,
        }
        (ctx.work_dir / "stats.json").write_text(json.dumps(stats, indent=2))

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


class CanaryFilterStage(Stage):
    name = "02g_canary_filter"
    stage_action = "filtered"

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        filter_config = ctx.config.filters.canary
        if not filter_config.enabled or not filter_config.phrases:
            return records

        kept_records: list[Record] = []
        dropped_records: list[Record] = []
        drop_reasons: dict[str, int] = {}

        for record in records:
            if not isinstance(record, ConversationRecord):
                kept_records.append(record)
                continue

            text_to_check = f"{record.payload.instruction}\n{record.payload.response}"
            matched_phrase = next(
                (phrase for phrase in filter_config.phrases if phrase in text_to_check),
                None,
            )

            if matched_phrase is None:
                kept_records.append(record)
            else:
                reason_code = "canary_leak"
                dropped_records.append(
                    self._drop_record(
                        record=record,
                        reason_code=reason_code,
                        details=f"Matched canary phrase: {matched_phrase}",
                    )
                )
                drop_reasons[reason_code] = drop_reasons.get(reason_code, 0) + 1

        self._write_stage_artifacts(
            ctx=ctx,
            dropped_records=dropped_records,
            scored_count=len(records),
            kept_count=len(kept_records),
            dropped_count=len(dropped_records),
            drop_reasons=drop_reasons,
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
    ) -> None:
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        if dropped_records:
            writer = OutputWriter()
            writer.write_dropped_parquet(
                records=dropped_records, path=ctx.work_dir / "dropped.parquet"
            )
        stats = {
            "stage": self.name,
            "count_in": scored_count,
            "count_out": kept_count,
            "dropped_count": dropped_count,
            "drop_reasons": drop_reasons,
        }
        (ctx.work_dir / "stats.json").write_text(json.dumps(stats, indent=2))

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


class LabelingQualityFilterStage(Stage):
    name = "03_label_quality"
    stage_action = "filtered"

    def __init__(self, project_root: Path, llm_client: Any | None = None) -> None:
        self.project_root = project_root
        self._llm_client = llm_client
        self._output_writer = OutputWriter()

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        filter_config = ctx.config.filters.labeling_engine
        if not filter_config.enabled or filter_config.rubric_path is None:
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


def validate_ifd_capability(ctx: StageContext) -> None:
    filter_config = ctx.config.filters.ifd
    if not filter_config.enabled:
        return
    if not provider_supports_sequence_scoring(ctx.config.llm):
        raise ValueError(
            "IFD requires provider/model response-scoring capability; "
            f"unsupported for provider={ctx.config.llm.provider!r} "
            f"model={ctx.config.llm.model!r}"
        )
