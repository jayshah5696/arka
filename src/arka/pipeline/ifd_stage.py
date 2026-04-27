from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from arka.llm.client import LLMClient, LLMClientError
from arka.llm.models import SequenceScore
from arka.pipeline.artifacts import StageArtifacts, StageReport
from arka.pipeline.stages import Stage
from arka.records.models import ConversationRecord, Record


class IFDFilterStage(Stage):
    name = "02e_ifd_filter"
    stage_action = "filtered"

    def __init__(self, *, project_root: Path, llm_client: Any | None = None) -> None:
        self.project_root = project_root
        self._llm_client = llm_client

    def run(self, records: list[Record], ctx) -> list[Record]:
        filter_config = ctx.config.filters.get_stage_config("ifd")
        if filter_config is None:
            return records

        llm_client = self._llm_client or LLMClient(config=ctx.config.llm)
        if not llm_client.supports_sequence_scoring():
            raise ValueError(
                "IFD requires provider/model response-scoring capability; "
                f"unsupported for provider={ctx.config.llm.provider!r} "
                f"model={ctx.config.llm.model!r}"
            )

        kept_records: list[Record] = []
        dropped_records: list[Record] = []
        scores: list[float] = []
        drop_reasons: dict[str, int] = {}

        for record in records:
            if not isinstance(record, ConversationRecord):
                kept_records.append(record)
                continue
            conditioned_messages = [
                {"role": "user", "content": record.payload.instruction}
            ]
            unconditioned_messages = [{"role": "user", "content": ""}]
            try:
                conditioned = llm_client.score_response(
                    messages=conditioned_messages,
                    target_text=record.payload.response,
                )
                unconditioned = llm_client.score_response(
                    messages=unconditioned_messages,
                    target_text=record.payload.response,
                )
            except LLMClientError as exc:
                raise ValueError(str(exc)) from exc

            ifd_score = compute_ifd(conditioned, unconditioned)
            scores.append(ifd_score)
            updated_record = record.model_copy(
                update={"scores": record.scores.model_copy(update={"ifd": ifd_score})}
            )
            if ifd_score >= filter_config.min_score:
                kept_records.append(updated_record)
                continue

            reason_code = "low_ifd"
            dropped_records.append(
                updated_record.dropped_by(
                    self.name,
                    reason_code,
                    f"ifd={ifd_score} < min_score={filter_config.min_score}",
                )
            )
            drop_reasons[reason_code] = drop_reasons.get(reason_code, 0) + 1

        self._write_artifacts(
            ctx=ctx,
            dropped_records=dropped_records,
            scored_count=len(scores),
            kept_count=len(kept_records),
            drop_reasons=drop_reasons,
            scores=scores,
        )
        return kept_records

    def _write_artifacts(
        self,
        *,
        ctx,
        dropped_records: list[Record],
        scored_count: int,
        kept_count: int,
        drop_reasons: dict[str, int],
        scores: list[float],
    ) -> None:
        StageArtifacts(ctx).write(
            report=StageReport(
                stage=self.name,
                count_in=scored_count,
                count_out=kept_count,
                dropped_count=len(dropped_records),
                drop_reasons=drop_reasons,
                scored_count=scored_count,
                ifd_distribution=ifd_distribution(scores),
            ),
            dropped=dropped_records,
        )


def compute_ifd(conditioned: SequenceScore, unconditioned: SequenceScore) -> float:
    return round(conditioned.mean_logprob - unconditioned.mean_logprob, 4)


def ifd_distribution(scores: list[float]) -> dict[str, float] | None:
    if not scores:
        return None
    std_score = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    return {
        "mean": round(statistics.fmean(scores), 4),
        "std": round(std_score, 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
    }
