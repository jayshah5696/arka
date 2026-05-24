"""Zero-LLM-cost filter stages that run before the expensive labeling path."""

from __future__ import annotations

import logging
import re
import statistics
from typing import TYPE_CHECKING

from arka.config.models import (
    LanguageFilterConfig,
    LengthFilterConfig,
    SentenceVarianceFilterConfig,
)
from arka.pipeline.stages import BaseFilterStage

if TYPE_CHECKING:
    from arka.records.models import ConversationRecord

logger = logging.getLogger(__name__)


class LengthFilterStage(BaseFilterStage):
    """Drop records whose instruction or response length is outside bounds."""

    name = "02a_length_filter"
    config_type = "length"
    config_class = LengthFilterConfig

    def _check_record(
        self, record: ConversationRecord, config: LengthFilterConfig
    ) -> tuple[str] | None:
        inst_len = len(record.payload.instruction)
        resp_len = len(record.payload.response)
        if inst_len < config.min_instruction_chars:
            return ("instruction_too_short",)
        if inst_len > config.max_instruction_chars:
            return ("instruction_too_long",)
        if resp_len < config.min_response_chars:
            return ("response_too_short",)
        if resp_len > config.max_response_chars:
            return ("response_too_long",)
        return None


class LanguageFilterStage(BaseFilterStage):
    """Drop records whose instruction is not in the allowed language set.

    Uses a simple heuristic based on character-set analysis. This avoids adding
    an external dependency (like ``langdetect`` or ``fasttext``) while still
    catching the most common mismatches. When ``allowed`` contains only
    ``"en"``, records whose instruction is predominantly non-Latin script are
    dropped.
    """

    name = "02b_language_filter"
    config_type = "language"
    config_class = LanguageFilterConfig

    def _is_active(self, config: LanguageFilterConfig) -> bool:
        self._warn_if_no_heuristic_available(config.allowed)
        return True

    def _check_record(
        self, record: ConversationRecord, config: LanguageFilterConfig
    ) -> tuple[str] | None:
        if self._is_allowed(record.payload.instruction, config.allowed):
            return None
        return ("language_mismatch",)

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


class SentenceVarianceFilterStage(BaseFilterStage):
    """Drop records whose response has too-uniform sentence lengths."""

    name = "02f_sentence_variance"
    config_type = "sentence_variance"
    config_class = SentenceVarianceFilterConfig

    def _check_record(
        self, record: ConversationRecord, config: SentenceVarianceFilterConfig
    ) -> tuple[str] | None:
        lengths = _sentence_lengths(record.payload.response)
        cv = _coefficient_of_variation(lengths)

        if cv >= config.min_cv:
            return None
        return ("low_sentence_variance",)
