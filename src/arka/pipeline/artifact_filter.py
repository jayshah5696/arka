"""Strip localized generation residue and drop unreliable text before dedup."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from arka.config.models import ArtifactFilterConfig
from arka.pipeline.artifacts import StageArtifacts, StageReport
from arka.pipeline.models import StageContext
from arka.pipeline.stages import Stage
from arka.records.identity import content_hash
from arka.records.models import ConversationRecord, Record

# Treat fenced code, inline code, and double-quoted examples as literal text.
_LITERAL_TEXT = re.compile(
    r'(?s)```.*?(?:```|\Z)|~~~.*?(?:~~~|\Z)|`[^`\n]*`|"(?:\\.|[^"\\])*"'
)


@dataclass(frozen=True)
class ArtifactMatch:
    field: str
    group: str
    start: int
    end: int


@dataclass(frozen=True)
class ArtifactCheck:
    record: ConversationRecord
    reason: str | None
    group_counts: Counter[str]
    strip_count: int


class ArtifactFilterStage(Stage):
    name = "02i_artifact_filter"
    stage_action = "filtered"

    def __init__(self, config: ArtifactFilterConfig | None = None) -> None:
        self.config = config

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        self.config = self.get_stage_config(
            ctx, ArtifactFilterConfig, "artifact_filter"
        )
        if self.config is None:
            return records
        patterns = {
            group: [re.compile(pattern, re.IGNORECASE) for pattern in group_patterns]
            for group, group_patterns in self.config.groups.items()
        }
        kept: list[Record] = []
        dropped: list[Record] = []
        reasons: Counter[str] = Counter()
        group_counts: Counter[str] = Counter()
        stripped_count = 0
        strip_match_count = 0
        for record in records:
            if not isinstance(record, ConversationRecord):
                kept.append(record)
                continue
            checked = self._check_record(record, self.config, patterns)
            group_counts.update(checked.group_counts)
            stripped_count += checked.strip_count > 0
            strip_match_count += checked.strip_count
            if checked.reason:
                reasons[checked.reason] += 1
                dropped.append(checked.record.dropped_by(self.name, checked.reason))
            else:
                kept.append(checked.record)
        StageArtifacts(ctx).write(
            report=StageReport(
                stage=self.name,
                count_in=len(records),
                count_out=len(kept),
                dropped_count=len(dropped),
                drop_reasons=dict(reasons),
                group_match_counts=dict(group_counts),
                stripped_count=stripped_count,
                strip_match_count=strip_match_count,
            ),
            dropped=dropped if dropped else None,
        )
        return kept

    def _matches(
        self,
        record: ConversationRecord,
        config: ArtifactFilterConfig,
        patterns: dict[str, list[re.Pattern[str]]],
    ) -> list[ArtifactMatch]:
        matches: list[ArtifactMatch] = []
        seen: set[ArtifactMatch] = set()
        for field in config.fields:
            text = getattr(record.payload, field) or ""
            protected = [match.span() for match in _LITERAL_TEXT.finditer(text)]
            for group, group_patterns in patterns.items():
                for pattern in group_patterns:
                    for match in pattern.finditer(text):
                        if match.start() == match.end():
                            raise ValueError(
                                f"Artifact regex in {group} must consume text"
                            )
                        if any(
                            match.start() < end and start < match.end()
                            for start, end in protected
                        ):
                            continue
                        found = ArtifactMatch(field, group, match.start(), match.end())
                        if found not in seen:
                            seen.add(found)
                            matches.append(found)
        return matches

    def _check_record(
        self,
        record: ConversationRecord,
        config: ArtifactFilterConfig,
        patterns: dict[str, list[re.Pattern[str]]],
    ) -> ArtifactCheck:
        counts: Counter[str] = Counter()
        strip_count = 0
        while matches := self._matches(record, config, patterns):
            counts.update(match.group for match in matches)
            dropped = next(
                (
                    match
                    for match in matches
                    if config.action_for(match.group) == "drop"
                ),
                None,
            )
            if dropped:
                return ArtifactCheck(
                    record, f"artifact_{dropped.group}", counts, strip_count
                )

            fields = dict.fromkeys(match.field for match in matches)
            changes: dict[str, str] = {}
            spans_by_field: dict[str, list[tuple[int, int]]] = {}
            for field in fields:
                spans: list[tuple[int, int]] = []
                for start, end in sorted(
                    (m.start, m.end) for m in matches if m.field == field
                ):
                    if spans and start < spans[-1][1]:
                        spans[-1] = (spans[-1][0], max(spans[-1][1], end))
                    else:
                        spans.append((start, end))
                spans_by_field[field] = spans
            removal_count = sum(len(spans) for spans in spans_by_field.values())
            if strip_count + removal_count > config.max_strips_per_record:
                return ArtifactCheck(
                    record, "artifact_strip_limit", counts, strip_count
                )

            details: list[str] = []
            for field, spans in spans_by_field.items():
                before = getattr(record.payload, field)
                after = before
                for start, end in reversed(spans):
                    after = after[:start] + after[end:]
                changes[field] = after
                details.append(
                    f"field={field}, before_length={len(before)}, after_length={len(after)}"
                )
            payload = record.payload.model_copy(update=changes)
            record = record.model_copy(
                update={"payload": payload, "content_hash": content_hash(payload)}
            )
            record = record.with_event(
                stage=self.name, action="stripped", details="; ".join(details)
            )
            strip_count += removal_count
            if any(
                field in changes and not changes[field].strip()
                for field in ("instruction", "response")
            ):
                return ArtifactCheck(
                    record, "artifact_empty_content", counts, strip_count
                )
        return ArtifactCheck(record, None, counts, strip_count)
