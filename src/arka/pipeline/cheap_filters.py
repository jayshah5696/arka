"""Zero-LLM-cost filter stages that run before the expensive labeling path."""

from __future__ import annotations

import json

from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.pipeline.stages import Stage
from arka.records.models import ConversationRecord, Record, StageEvent


class LengthFilterStage(Stage):
    """Drop records whose instruction or response length is outside bounds."""

    name = "02a_length_filter"
    stage_action = "filtered"

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        cfg = ctx.config.filters.length
        if not cfg.enabled:
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

        self._write_artifacts(ctx, dropped, len(records), len(kept), drop_reasons)
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

    def _write_artifacts(
        self,
        ctx: StageContext,
        dropped: list[Record],
        count_in: int,
        count_out: int,
        drop_reasons: dict[str, int],
    ) -> None:
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        writer = OutputWriter()
        writer.write_dropped_parquet(
            records=dropped, path=ctx.work_dir / "dropped.parquet"
        )
        stats = {
            "stage": self.name,
            "count_in": count_in,
            "count_out": count_out,
            "dropped_count": len(dropped),
            "drop_reasons": drop_reasons,
        }
        (ctx.work_dir / "stats.json").write_text(json.dumps(stats, indent=2))


class LanguageFilterStage(Stage):
    """Drop records whose instruction is not in the allowed language set.

    Uses a simple heuristic based on character-set analysis.  This avoids
    adding an external dependency (like ``langdetect`` or ``fasttext``)
    while still catching the most common mismatches.  When ``allowed``
    contains only ``"en"``, records whose instruction is predominantly
    non-Latin script are dropped.
    """

    name = "02b_language_filter"
    stage_action = "filtered"

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        cfg = ctx.config.filters.language
        if not cfg.enabled:
            return records

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

        self._write_artifacts(ctx, dropped, len(records), len(kept), drop_reasons)
        return kept

    def _is_allowed(self, text: str, allowed: list[str]) -> bool:
        if "en" in allowed:
            return self._is_predominantly_latin(text)
        # For non-English allowed sets, accept everything (no heuristic yet).
        return True

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

    def _write_artifacts(
        self,
        ctx: StageContext,
        dropped: list[Record],
        count_in: int,
        count_out: int,
        drop_reasons: dict[str, int],
    ) -> None:
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        writer = OutputWriter()
        writer.write_dropped_parquet(
            records=dropped, path=ctx.work_dir / "dropped.parquet"
        )
        stats = {
            "stage": self.name,
            "count_in": count_in,
            "count_out": count_out,
            "dropped_count": len(dropped),
            "drop_reasons": drop_reasons,
        }
        (ctx.work_dir / "stats.json").write_text(json.dumps(stats, indent=2))
