from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from arka.config.models import (
    CanaryFilterConfig,
    IFDFilterConfig,
    LabelingFilterConfig,
    SemanticSimilarityFilterConfig,
)
from arka.embeddings import Embedder
from arka.llm.client import (
    LLMClientError,
    provider_supports_sequence_scoring,
)
from arka.pipeline.artifacts import StageArtifacts, StageReport
from arka.pipeline.labeling_evaluator import (
    evaluate_labeling,
    quality_distribution,
    reason_code_for_label_error,
)
from arka.pipeline.models import StageContext
from arka.pipeline.stages import BaseFilterStage, Stage
from arka.records.models import (
    ConversationRecord,
    Record,
)

__all__ = [
    "CanaryFilterStage",
    "LabelingQualityFilterStage",
    "SemanticSimilarityFilterStage",
    "validate_ifd_capability",
]


class CanaryFilterStage(BaseFilterStage):
    """Drop records whose text contains any configured canary phrase."""

    name = "02g_canary_filter"
    config_type = "canary"
    config_class = CanaryFilterConfig

    def _is_active(self, config: CanaryFilterConfig) -> bool:
        return bool(config.phrases)

    def _check_record(
        self, record: ConversationRecord, config: CanaryFilterConfig
    ) -> tuple[str, str] | None:
        text = f"{record.payload.instruction}\n{record.payload.response}"
        matched = next((p for p in config.phrases if p in text), None)
        if matched is None:
            return None
        return ("canary_leak", f"Matched canary phrase: {matched}")


class SemanticSimilarityFilterStage(Stage):
    """Drop generated records too similar to their seed records."""

    name = "02h_semantic_similarity_filter"
    stage_action = "filtered"

    def __init__(self, config: SemanticSimilarityFilterConfig | None = None) -> None:
        self.config = config

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        self.config = self.get_stage_config(
            ctx, SemanticSimilarityFilterConfig, "semantic_similarity"
        )
        if self.config is None:
            return records
        filter_config = self.config

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

        embedder = Embedder(ctx.config)
        gen_texts = [
            f"{r.payload.instruction}\n{r.payload.response}" for r in generated
        ]
        seed_texts = [f"{r.payload.instruction}\n{r.payload.response}" for r in seeds]

        gen_emb = embedder.embed(gen_texts, checkpoint_manager=ctx.checkpoint_manager)
        seed_emb = embedder.embed(seed_texts, checkpoint_manager=ctx.checkpoint_manager)

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
        decision_results = []

        for i, record in enumerate(generated):
            max_sim = float(np.max(sim_matrix[i]))
            decision_cfg = filter_config.decision_band
            reason = None
            if decision_cfg.enabled:
                if max_sim > decision_cfg.upper:
                    reason = "high_semantic_similarity"
                elif max_sim >= decision_cfg.lower:
                    from arka.decisions.models import YesNoQuestion
                    from arka.pipeline.decision_stages import (
                        attach_answers,
                        record_state,
                    )

                    nearest = seeds[int(np.argmax(sim_matrix[i]))]
                    questions = [
                        YesNoQuestion(
                            name="semantic_similarity.copies_seed",
                            instructions="Does the generated example reproduce the seed's specific facts, names, numbers or task and answer, rather than only sharing its topic?",
                        ),
                        YesNoQuestion(
                            name="semantic_similarity.leaks_specifics",
                            instructions="Does the generated example reveal a specific name, figure, identifier or quote present in the seed?",
                        ),
                    ]
                    with ctx.decision_client(
                        override=decision_cfg.decision_override
                    ) as client:
                        result = client.decide(
                            json.dumps(
                                {
                                    "generated": record_state(record),
                                    "seed": record_state(nearest),
                                }
                            ),
                            questions,
                        )
                    decision_results.append(result)
                    record = attach_answers(record, result.answers)
                    if (
                        max(a.probability for a in result.answers.values())
                        >= decision_cfg.reject_at
                    ):
                        reason = "seed_copy_confirmed"
            elif max_sim > filter_config.threshold:
                reason = "high_semantic_similarity"
            if reason is not None:
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

        from arka.pipeline.decision_stages import decision_usage

        StageArtifacts(ctx).write(
            report=StageReport(
                stage=self.name,
                count_in=len(records),
                count_out=len(kept),
                dropped_count=len(dropped),
                drop_reasons=drop_reasons,
                decision_usage=decision_usage(decision_results),
            ),
            dropped=dropped if dropped else None,
        )
        return kept


def _write_filter_artifacts(
    ctx: StageContext,
    stage_name: str,
    count_in: int,
    count_out: int,
    dropped: list[Record],
    drop_reasons: dict[str, int],
) -> None:
    StageArtifacts(ctx).write(
        report=StageReport(
            stage=stage_name,
            count_in=count_in,
            count_out=count_out,
            dropped_count=len(dropped),
            drop_reasons=drop_reasons,
        ),
        # Preserve historical behaviour: skip dropped.parquet when empty.
        dropped=dropped if dropped else None,
    )


class LabelingQualityFilterStage(Stage):
    name = "03_label_quality"
    stage_action = "filtered"

    def __init__(
        self,
        config: LabelingFilterConfig | None = None,
        *,
        project_root: Path,
        llm_client: Any | None = None,
    ) -> None:
        self.project_root = project_root
        self._llm_client = llm_client
        self.config = config

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        self.config = self.get_stage_config(
            ctx, LabelingFilterConfig, "labeling_engine"
        )
        return evaluate_labeling(
            stage=self,
            records=records,
            ctx=ctx,
            config=self.config,
            project_root=self.project_root,
            llm_client=self._llm_client,
            drop_below_threshold=True,
            missing_file_exc_cls=ValueError,
        )

    def _reason_code_for_label_error(self, error: LLMClientError) -> str:
        return reason_code_for_label_error(error)

    def _quality_distribution(self, scores: list[float]) -> dict[str, float] | None:
        return quality_distribution(scores)


def validate_ifd_capability(config: IFDFilterConfig | Any, ctx: Any = None) -> None:
    """Validate that sequence scoring capability is present for IFD."""
    llm = (
        getattr(ctx, "config", None)
        and getattr(ctx.config, "llm", None)
        or getattr(ctx, "llm", None)
        or ctx
    )
    if llm is not None and not provider_supports_sequence_scoring(llm):
        raise ValueError(
            "IFD requires provider/model response-scoring capability; "
            f"unsupported for provider={llm.provider!r} "
            f"model={llm.model!r}"
        )
