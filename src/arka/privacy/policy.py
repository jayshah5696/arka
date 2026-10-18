from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol


class PrivacyError(RuntimeError):
    """A privacy boundary failed. Messages must not contain input text."""


@dataclass(frozen=True)
class Span:
    """Character offsets into the original input, with an exclusive end."""

    start: int
    end: int
    entity: str


class Detector(Protocol):
    def detect(self, text: str) -> list[Span]: ...


@dataclass(frozen=True)
class ProtectionResult:
    value: Any
    matched_bytes: int
    total_bytes: int
    categories: dict[str, int]
    drop_reason: str | None


class PIIPolicy:
    """Apply one detector and policy to original text values before use.

    Detectors return offsets, never replacement text. Ratios use the union
    of original matched UTF-8 bytes. Dictionary keys and non-text values do
    not contribute to the ratio and are not changed.
    """

    def __init__(
        self,
        detector: Detector,
        *,
        placeholder_template: str = "[{entity}]",
        max_redacted_ratio: float | None = None,
        blocked_entities: set[str] | None = None,
    ) -> None:
        self.detector = detector
        self.placeholder_template = placeholder_template
        self.max_redacted_ratio = max_redacted_ratio
        self.blocked_entities = blocked_entities or set()
        self.fingerprint = ""
        prefix, _, suffix = placeholder_template.partition("{entity}")
        self._placeholder = (
            re.compile(re.escape(prefix) + r"[a-z_]+" + re.escape(suffix))
            if prefix and suffix
            else None
        )

    def protect(self, value: Any) -> ProtectionResult:
        categories: Counter[str] = Counter()
        matched_bytes = 0
        total_bytes = 0

        def visit(item: Any) -> Any:
            nonlocal matched_bytes, total_bytes
            if isinstance(item, str):
                protected, matched, found = self._text(item)
                matched_bytes += matched
                total_bytes += len(item.encode("utf-8"))
                categories.update(found)
                return protected
            if isinstance(item, dict):
                return {key: visit(child) for key, child in item.items()}
            if isinstance(item, list):
                return [visit(child) for child in item]
            return item

        protected = visit(value)
        reason = None
        if self.blocked_entities.intersection(categories):
            reason = "pii_blocked_entity"
        elif (
            self.max_redacted_ratio is not None
            and total_bytes
            and matched_bytes / total_bytes > self.max_redacted_ratio
        ):
            reason = "pii_ratio"
        return ProtectionResult(
            protected, matched_bytes, total_bytes, dict(categories), reason
        )

    def protect_response(self, text: str) -> ProtectionResult:
        """Decode JSON values before scanning, or scan malformed output as text."""
        try:
            value = json.loads(text)
        except (ValueError, TypeError):
            return self.protect(text)
        result = self.protect(value)
        return ProtectionResult(
            json.dumps(result.value, ensure_ascii=False),
            result.matched_bytes,
            result.total_bytes,
            result.categories,
            result.drop_reason,
        )

    def _text(self, text: str) -> tuple[str, int, Counter[str]]:
        try:
            spans = self.detector.detect(text) if text else []
            if any(
                not isinstance(span, Span)
                or type(span.start) is not int
                or type(span.end) is not int
                or not 0 <= span.start < span.end <= len(text)
                or not span.entity
                for span in spans
            ):
                raise PrivacyError("PII detector returned invalid spans")
            spans = sorted(set(spans), key=lambda s: (s.start, -s.end, s.entity))
            placeholders = (
                list(self._placeholder.finditer(text)) if self._placeholder else []
            )
            spans = [
                span
                for span in spans
                if not any(
                    match.start() <= span.start and span.end <= match.end()
                    for match in placeholders
                )
            ]
        except Exception:
            raise PrivacyError("PII detection failed") from None

        categories: Counter[str] = Counter(span.entity for span in spans)
        # Connected overlaps become one replacement. The first, longest span
        # at a shared start determines the placeholder category.
        merged: list[Span] = []
        for span in spans:
            if merged and span.start < merged[-1].end:
                previous = merged[-1]
                merged[-1] = Span(
                    previous.start, max(previous.end, span.end), previous.entity
                )
            else:
                merged.append(span)

        pieces: list[str] = []
        matched_bytes = 0
        cursor = 0
        for span in merged:
            pieces.append(text[cursor : span.start])
            pieces.append(self.placeholder_template.format(entity=span.entity))
            matched_bytes += len(text[span.start : span.end].encode("utf-8"))
            cursor = span.end
        pieces.append(text[cursor:])
        return "".join(pieces), matched_bytes, categories
