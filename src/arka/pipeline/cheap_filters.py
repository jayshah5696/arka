"""Zero-LLM-cost filter stages that run before the expensive labeling path."""

from __future__ import annotations

import logging
import re
import statistics
from itertools import groupby
from typing import TYPE_CHECKING

from arka.config.models import (
    IdentityClaimFilterConfig,
    LanguageFilterConfig,
    LengthFilterConfig,
    RepetitionFilterConfig,
    SentenceVarianceFilterConfig,
)
from arka.pipeline.artifacts import StageArtifacts, StageReport
from arka.pipeline.stages import BaseFilterStage

if TYPE_CHECKING:
    from arka.pipeline.models import StageContext
    from arka.records.models import ConversationRecord, Record

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


def _covered_fraction(spans: list[tuple[int, int]], text_length: int) -> float:
    """Count characters in matching spans once, including internal whitespace."""
    covered = 0
    end = 0
    for start, stop in sorted(spans):
        covered += max(0, stop - max(start, end))
        end = max(end, stop)
    return covered / text_length


class RepetitionFilterStage(BaseFilterStage):
    """Drop clear character runs or repeated word sequences without LLM calls."""

    name = "02c_repetition_filter"
    config_type = "repetition"
    config_class = RepetitionFilterConfig

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        self.max_observed_fractions: dict[str, float] = {}
        return super().run(records, ctx)

    def _check_record(
        self, record: ConversationRecord, config: RepetitionFilterConfig
    ) -> tuple[str, str] | None:
        for field in config.check_fields:
            text = getattr(record.payload, field) or ""
            # Character runs are reliable even below the n-gram length gate.
            if any(
                sum(1 for _ in run) > config.max_char_run for _, run in groupby(text)
            ):
                return ("long_char_run", f"field={field}")
            if len(text) < config.min_text_chars or not text:
                continue

            words = list(re.finditer(r"\S+", text))
            failure: tuple[str, str] | None = None
            for family, thresholds in (
                ("top", config.top_ngram_thresholds),
                ("dup", config.dup_ngram_thresholds),
            ):
                for n, threshold in thresholds.items():
                    occurrences: dict[tuple[str, ...], list[tuple[int, int]]] = {}
                    for i in range(len(words) - n + 1):
                        key = tuple(word.group() for word in words[i : i + n])
                        occurrences.setdefault(key, []).append(
                            (words[i].start(), words[i + n - 1].end())
                        )
                    repeated = [
                        spans for spans in occurrences.values() if len(spans) > 1
                    ]
                    if family == "top" and repeated:
                        max_count = max(len(spans) for spans in repeated)
                        fraction = max(
                            _covered_fraction(spans, len(text))
                            for spans in repeated
                            if len(spans) == max_count
                        )
                    else:
                        fraction = _covered_fraction(
                            [span for spans in repeated for span in spans], len(text)
                        )
                    metric = f"{family}_{n}"
                    self.max_observed_fractions[metric] = max(
                        self.max_observed_fractions.get(metric, 0.0), fraction
                    )
                    if fraction > threshold and failure is None:
                        reason = (
                            "repetitive_top_ngram"
                            if family == "top"
                            else "repetitive_dup_ngrams"
                        )
                        failure = (
                            reason,
                            f"field={field}, metric={metric}, fraction={fraction:.6f}",
                        )
            if failure:
                return failure
        return None

    def _write_artifacts(
        self,
        ctx: StageContext,
        count_in: int,
        count_out: int,
        dropped: list[Record],
        drop_reasons: dict[str, int],
    ) -> None:
        StageArtifacts(ctx).write(
            report=StageReport(
                stage=self.name,
                count_in=count_in,
                count_out=count_out,
                dropped_count=len(dropped),
                drop_reasons=drop_reasons,
                max_observed_fractions=self.max_observed_fractions,
            ),
            dropped=dropped if dropped else None,
        )


_MODEL_NAMES = (
    r"(?:gpt(?:-\d+)?|chatgpt|claude|gemini|llama|qwen|deepseek|mistral|"
    r"mixtral|grok|palm|bard|bing|copilot)\b"
)
_LAB_NAMES = (
    r"(?:openai|anthropic|google(?:\s+deepmind)?|meta|alibaba|mistral\s+ai|"
    r"deepseek|xai|microsoft)\b"
)
_CREATOR_CLAIM = rf"(?:created|developed|trained|built|made)\s+by\s+{_LAB_NAMES}"
_LAB_MODEL_CLAIM = rf"(?:an?\s+)?{_LAB_NAMES}\s+(?:ai\s+)?(?:language\s+)?model\b"
_IDENTITY_PATTERNS = {
    "first_person": [
        rf"\bi(?:'m|\s+am)\s+(?:an?\s+)?{_MODEL_NAMES}",
        rf"\bas\s+(?:an?\s+)?{_MODEL_NAMES}\s*,?\s*i\b",
    ],
    "creator": [
        rf"\bi(?:'m|\s+(?:am|was|have\s+been))\s+{_CREATOR_CLAIM}",
        rf"\bi(?:'m|\s+am)\s+{_LAB_MODEL_CLAIM}",
    ],
}
_IDENTITY_DENIAL = re.compile(
    r"\bi(?:'m\s+(?:not|never)|\s+(?:am|was)\s+(?:not|never)|"
    r"\s+(?:wasn't|amn't|ain't))\s+"
    rf"(?:(?:an?\s+)?{_MODEL_NAMES}|{_CREATOR_CLAIM}|{_LAB_MODEL_CLAIM})",
    re.IGNORECASE,
)


class IdentityClaimFilterStage(BaseFilterStage):
    """Drop foreign identity claims while preserving known-model denials."""

    name = "02d_identity_claim_filter"
    config_type = "identity_claim"
    config_class = IdentityClaimFilterConfig

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        self.pattern_group_counts: dict[str, int] = {}
        return super().run(records, ctx)

    def _is_active(self, config: IdentityClaimFilterConfig) -> bool:
        groups = {"custom": config.patterns} if config.patterns else _IDENTITY_PATTERNS
        self.pattern_groups = {
            group: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for group, patterns in groups.items()
        }
        return True

    def _check_record(
        self, record: ConversationRecord, config: IdentityClaimFilterConfig
    ) -> tuple[str, str] | None:
        texts = [getattr(record.payload, field) or "" for field in config.check_fields]
        if any(
            name.casefold() in text.casefold()
            for name in config.own_model_names
            for text in texts
        ):
            return None
        matches: set[str] = set()
        for text in texts:
            checked = _IDENTITY_DENIAL.sub("", text.replace("’", "'"))
            for group, patterns in self.pattern_groups.items():
                if any(pattern.search(checked) for pattern in patterns):
                    matches.add(group)
        if not matches:
            return None
        for group in matches:
            self.pattern_group_counts[group] = (
                self.pattern_group_counts.get(group, 0) + 1
            )
        return ("identity_claim", f"pattern_groups={','.join(sorted(matches))}")

    def _write_artifacts(
        self,
        ctx: StageContext,
        count_in: int,
        count_out: int,
        dropped: list[Record],
        drop_reasons: dict[str, int],
    ) -> None:
        StageArtifacts(ctx).write(
            report=StageReport(
                stage=self.name,
                count_in=count_in,
                count_out=count_out,
                dropped_count=len(dropped),
                drop_reasons=drop_reasons,
                pattern_group_counts=self.pattern_group_counts,
            ),
            dropped=dropped if dropped else None,
        )
