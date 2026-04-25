"""Zero-LLM-cost filter stages that run before the expensive labeling path."""

from __future__ import annotations

import json
import logging
import re
import statistics

from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.pipeline.stages import Stage
from arka.records.models import ConversationRecord, Record, StageEvent

logger = logging.getLogger(__name__)


def write_filter_artifacts(
    *,
    stage_name: str,
    ctx: StageContext,
    dropped: list[Record],
    count_in: int,
    count_out: int,
    drop_reasons: dict[str, int],
) -> None:
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    writer = OutputWriter()
    writer.write_dropped_parquet(records=dropped, path=ctx.work_dir / "dropped.parquet")
    stats = {
        "stage": stage_name,
        "count_in": count_in,
        "count_out": count_out,
        "dropped_count": len(dropped),
        "drop_reasons": drop_reasons,
    }
    (ctx.work_dir / "stats.json").write_text(json.dumps(stats, indent=2))


class LengthFilterStage(Stage):
    """Drop records whose instruction or response length is outside bounds."""

    name = "02a_length_filter"
    stage_action = "filtered"

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        cfg = ctx.config.filters.get_stage_config("length")
        if cfg is None:
            return records

        kept: list[Record] = []
        dropped: list[Record] = []
        drop_reasons: dict[str, int] = {}

        for record in records:
            if not isinstance(record, ConversationRecord):
                kept.append(record)
                continue

            reason = self._check(record, cfg)
            if reason is None:
                kept.append(record)
            else:
                dropped.append(self._drop_record(record, reason_code=reason))
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

        write_filter_artifacts(
            stage_name=self.name,
            ctx=ctx,
            dropped=dropped,
            count_in=len(records),
            count_out=len(kept),
            drop_reasons=drop_reasons,
        )
        return kept

    def _check(self, record: ConversationRecord, cfg) -> str | None:
        inst_len = len(record.payload.instruction)
        resp_len = len(record.payload.response)
        if inst_len < cfg.min_instruction_chars:
            return "instruction_too_short"
        if inst_len > cfg.max_instruction_chars:
            return "instruction_too_long"
        if resp_len < cfg.min_response_chars:
            return "response_too_short"
        if resp_len > cfg.max_response_chars:
            return "response_too_long"
        return None

    def _drop_record(self, record: Record, reason_code: str) -> Record:
        return record.model_copy(
            update={
                "stage_events": [
                    *record.stage_events,
                    StageEvent(
                        stage=self.name,
                        action="dropped",
                        reason_code=reason_code,
                        seq=len(record.stage_events) + 1,
                    ),
                ]
            }
        )


class LanguageFilterStage(Stage):
    """Drop records whose instruction is not in the allowed language set.

    Uses a simple heuristic based on character-set analysis. This avoids adding
    an external dependency (like ``langdetect`` or ``fasttext``) while still
    catching the most common mismatches. When ``allowed`` contains only
    ``"en"``, records whose instruction is predominantly non-Latin script are
    dropped.
    """

    name = "02b_language_filter"
    stage_action = "filtered"

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        cfg = ctx.config.filters.get_stage_config("language")
        if cfg is None:
            return records

        self._warn_if_no_heuristic_available(cfg.allowed)

        kept: list[Record] = []
        dropped: list[Record] = []
        drop_reasons: dict[str, int] = {}

        for record in records:
            if not isinstance(record, ConversationRecord):
                kept.append(record)
                continue

            if self._is_allowed(record.payload.instruction, cfg.allowed):
                kept.append(record)
            else:
                reason = "language_mismatch"
                dropped.append(self._drop_record(record, reason_code=reason))
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

        write_filter_artifacts(
            stage_name=self.name,
            ctx=ctx,
            dropped=dropped,
            count_in=len(records),
            count_out=len(kept),
            drop_reasons=drop_reasons,
        )
        return kept

    def _is_allowed(self, text: str, allowed: list[str]) -> bool:
        if "en" in allowed:
            return self._is_predominantly_latin(text)
        # For non-English allowed sets, accept everything (no heuristic yet).
        return True

    def _warn_if_no_heuristic_available(self, allowed: list[str]) -> None:
        if "en" in allowed:
            return
        logger.warning(
            "Language filter heuristic only supports English ('en') today; "
            "allowed=%s will currently pass all records",
            allowed,
        )

    def _is_predominantly_latin(self, text: str) -> bool:
        """Return True if >= 70% of alphabetic chars are Basic Latin / Latin-1."""
        alpha_chars = [ch for ch in text if ch.isalpha()]
        if not alpha_chars:
            return True  # Empty or non-alpha text is allowed through.
        latin_count = sum(1 for ch in alpha_chars if ord(ch) < 0x0250)
        return latin_count / len(alpha_chars) >= 0.7

    def _drop_record(self, record: Record, reason_code: str) -> Record:
        return record.model_copy(
            update={
                "stage_events": [
                    *record.stage_events,
                    StageEvent(
                        stage=self.name,
                        action="dropped",
                        reason_code=reason_code,
                        seq=len(record.stage_events) + 1,
                    ),
                ]
            }
        )


_SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")


def _sentence_lengths(text: str) -> list[int]:
    """Split text on sentence-ending punctuation and return word counts."""
    parts = _SENTENCE_SPLIT_PATTERN.split(text)
    return [len(part.split()) for part in parts if part.strip()]


def _coefficient_of_variation(values: list[int]) -> float:
    """Return the coefficient of variation (stdev / mean)."""
    if len(values) < 2:
        return 1.0  # single sentence passes by convention
    mean = statistics.fmean(values)
    if mean == 0:
        return 0.0
    std = statistics.pstdev(values)
    return std / mean


class SentenceVarianceFilterStage(Stage):
    """Drop records whose response has too-uniform sentence lengths."""

    name = "02f_sentence_variance"
    stage_action = "filtered"

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        cfg = ctx.config.filters.get_stage_config("sentence_variance")
        if cfg is None:
            return records

        kept: list[Record] = []
        dropped: list[Record] = []
        drop_reasons: dict[str, int] = {}

        for record in records:
            if not isinstance(record, ConversationRecord):
                kept.append(record)
                continue

            lengths = _sentence_lengths(record.payload.response)
            cv = _coefficient_of_variation(lengths)

            if cv >= cfg.min_cv:
                kept.append(record)
            else:
                reason = "low_sentence_variance"
                dropped.append(self._drop_record(record, reason_code=reason))
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

        write_filter_artifacts(
            stage_name=self.name,
            ctx=ctx,
            dropped=dropped,
            count_in=len(records),
            count_out=len(kept),
            drop_reasons=drop_reasons,
        )
        return kept

    def _drop_record(self, record: Record, reason_code: str) -> Record:
        return record.model_copy(
            update={
                "stage_events": [
                    *record.stage_events,
                    StageEvent(
                        stage=self.name,
                        action="dropped",
                        reason_code=reason_code,
                        seq=len(record.stage_events) + 1,
                    ),
                ]
            }
        )
