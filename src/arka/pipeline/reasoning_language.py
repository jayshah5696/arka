"""Compare confidently identified languages on records carrying reasoning."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from arka.config.models import ReasoningLanguageGateConfig
from arka.pipeline.artifacts import StageArtifacts, StageReport
from arka.pipeline.decision_stages import decision_usage
from arka.pipeline.models import StageContext
from arka.pipeline.stages import Stage
from arka.records.identity import content_hash
from arka.records.models import ConversationRecord, Record

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LanguagePrediction:
    language: str
    confidence: float


class LanguageDetector(Protocol):
    def predict(self, text: str) -> LanguagePrediction: ...


class FastTextDetector:
    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Language identification model not found: {path}")
        try:
            import fasttext
        except ImportError as exc:
            raise RuntimeError(
                "Install the language extra with `uv sync --extra language`"
            ) from exc
        self.model = fasttext.load_model(str(path))

    def predict(self, text: str) -> LanguagePrediction:
        # The public batch API handles NumPy 2 without fastText's scalar copy=False path.
        labels, probabilities = self.model.predict([" ".join(text.split())], k=1)
        if not labels[0]:
            raise ValueError("Language identification model returned no label")
        return LanguagePrediction(
            labels[0][0].removeprefix("__label__"), float(probabilities[0][0])
        )


class ReasoningLanguageGateStage(Stage):
    name = "02j_reasoning_language_gate"
    stage_action = "filtered"

    def __init__(
        self,
        config: ReasoningLanguageGateConfig | None = None,
        project_root: Path | None = None,
        detector: LanguageDetector | None = None,
    ) -> None:
        self.config = config
        self.project_root = project_root or Path.cwd()
        self.detector = detector

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        self.config = self.get_stage_config(
            ctx, ReasoningLanguageGateConfig, "reasoning_language_gate"
        )
        if self.config is None:
            return records
        cfg = self.config
        detector = self.detector
        if detector is None:
            if cfg.lid_model_path is None:
                logger.warning(
                    "reasoning_language_gate has no lid_model_path; passing records without language checks"
                )
            else:
                detector = FastTextDetector(self.project_root / cfg.lid_model_path)
        kept: list[Record] = []
        dropped: list[Record] = []
        mismatches: Counter[str] = Counter()
        checked: Counter[str] = Counter()
        low_confidence: Counter[str] = Counter()
        stripped = 0
        unjudged_reference = 0
        decision_results = []
        for record in records:
            if (
                (detector is None and cfg.low_confidence_backend == "skip")
                or not isinstance(record, ConversationRecord)
                or not record.payload.reasoning
            ):
                kept.append(record)
                continue
            reference_text = getattr(record.payload, cfg.reference_field) or ""
            if not reference_text.strip():
                unjudged_reference += 1
                kept.append(record)
                continue
            reference = (
                detector.predict(reference_text)
                if detector
                else LanguagePrediction("unknown", 0)
            )
            reference_uncertain = reference.confidence < cfg.min_confidence
            if reference_uncertain:
                unjudged_reference += 1
                if cfg.low_confidence_backend == "skip":
                    kept.append(record)
                    continue
            found: dict[str, LanguagePrediction] = {}
            skipped = []
            for field in dict.fromkeys(cfg.checked_fields):
                text = getattr(record.payload, field) or ""
                if not text.strip():
                    continue
                prediction = (
                    detector.predict(text)
                    if detector
                    else LanguagePrediction("unknown", 0)
                )
                if reference_uncertain or prediction.confidence < cfg.min_confidence:
                    low_confidence[field] += 1
                    if cfg.low_confidence_backend == "decision":
                        skipped.append(field)
                    continue
                checked[field] += 1
                if prediction.language != reference.language:
                    mismatches[field] += 1
                    found[field] = prediction
            if skipped:
                from arka.decisions.models import YesNoQuestion
                from arka.pipeline.decision_stages import attach_answers, record_state

                questions = [
                    YesNoQuestion(
                        name=f"reasoning_language_gate.same_language_{field}",
                        instructions=f"Is the natural-language prose in {field} in the same language as {cfg.reference_field}? Ignore code, math and quoted terms.",
                    )
                    for field in skipped
                ]
                with ctx.decision_client(override=cfg.decision_override) as client:
                    result = client.decide(record_state(record), questions)
                decision_results.append(result)
                record = attach_answers(record, result.answers)
                for field in skipped:
                    checked[field] += 1
                    probability = result.answers[
                        f"reasoning_language_gate.same_language_{field}"
                    ].probability
                    if probability < 1 - cfg.mismatch_at:
                        found[field] = LanguagePrediction(
                            "decision_mismatch", 1 - probability
                        )
                        mismatches[field] += 1
            if not found:
                kept.append(record)
                continue
            details = json.dumps(
                {
                    "reference_field": cfg.reference_field,
                    "reference": asdict(reference),
                    "mismatches": {
                        field: asdict(prediction) for field, prediction in found.items()
                    },
                }
            )
            if cfg.on_mismatch == "drop_record":
                dropped.append(
                    record.dropped_by(self.name, "reasoning_language_mismatch", details)
                )
                continue
            payload = record.payload.model_copy(update={"reasoning": None})
            record = record.model_copy(
                update={"payload": payload, "content_hash": content_hash(payload)}
            )
            kept.append(
                record.with_event(
                    stage=self.name, action="reasoning_stripped", details=details
                )
            )
            stripped += 1
        StageArtifacts(ctx).write(
            report=StageReport(
                stage=self.name,
                count_in=len(records),
                count_out=len(kept),
                dropped_count=len(dropped),
                drop_reasons={"reasoning_language_mismatch": len(dropped)}
                if dropped
                else {},
                enabled=detector is not None,
                stripped_count=stripped,
                field_mismatch_counts=dict(mismatches),
                field_checked_counts=dict(checked),
                field_mismatch_rates={
                    field: mismatches[field] / count for field, count in checked.items()
                },
                field_low_confidence_counts=dict(low_confidence),
                unjudged_reference_count=unjudged_reference,
                decision_usage=decision_usage(decision_results),
            ),
            dropped=dropped,
        )
        return kept
