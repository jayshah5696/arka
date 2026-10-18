from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import cast

from arka.pipeline.output import OutputWriter
from arka.privacy.models import PIIReport, PIISummary
from arka.privacy.policy import PIIPolicy, PrivacyError
from arka.records.identity import content_hash, record_id
from arka.records.models import ConversationRecord, GroundedChunkRecord, Record


def protect_record[RecordT: Record](
    record: RecordT, policy: PIIPolicy
) -> tuple[RecordT, PIISummary]:
    """Protect payload text in memory. Source coordinates retain their meaning."""
    if not isinstance(record, ConversationRecord | GroundedChunkRecord):
        raise PrivacyError(
            "PII protection requires a typed conversation or document record"
        )
    payload = record.payload.model_dump(mode="json")
    fields = (
        {"text": payload["text"]}
        if isinstance(record, GroundedChunkRecord)
        else {key: value for key, value in payload.items() if key != "turns"}
    )
    turns = payload.get("turns")
    if turns is not None:
        fields["turns"] = []
        for turn in turns:
            text_values = {key: turn[key] for key in ("content", "text") if key in turn}
            if any(
                value is not None and not isinstance(value, str)
                for value in text_values.values()
            ):
                raise PrivacyError(
                    "PII protection requires text-only conversation turns"
                )
            fields["turns"].append(text_values)
    result = policy.protect(fields)
    payload.update(result.value)
    if turns is not None:
        payload["turns"] = [
            {**turn, **protected}
            for turn, protected in zip(turns, result.value["turns"], strict=True)
        ]
    if isinstance(record, GroundedChunkRecord):
        payload["word_count"] = len(payload["text"].split())
    updated_payload = type(record.payload).model_validate(payload)
    if updated_payload != record.payload:
        rid = record_id(
            updated_payload, record.lineage if record.lineage.parent_ids else None
        )
        lineage = (
            record.lineage.model_copy(update={"root_id": rid})
            if not record.lineage.parent_ids
            else record.lineage
        )
        record = record.model_copy(
            update={
                "payload": updated_payload,
                "id": rid,
                "lineage": lineage,
                "content_hash": content_hash(updated_payload),
            }
        )
    summary = PIISummary(
        matched_bytes=result.matched_bytes,
        total_bytes=result.total_bytes,
        categories=result.categories,
        drop_reason=result.drop_reason,
    )
    return record, summary


def protect_records[RecordT: Record](
    records: list[RecordT], policy: PIIPolicy, work_dir: Path, stage: str
) -> list[RecordT]:
    """Write only protected rejections and payload-free counts for this boundary."""
    kept, dropped = [], []
    categories, reasons = Counter(), Counter()
    for record in records:
        protected, summary = protect_record(record, policy)
        categories.update(summary.categories)
        if summary.categories:
            protected = protected.with_event(
                stage=stage, action="redacted", details=summary.model_dump_json()
            )
        if summary.drop_reason is None:
            kept.append(cast(RecordT, protected))
        else:
            dropped.append(
                protected.dropped_by(
                    stage, summary.drop_reason, summary.model_dump_json()
                )
            )
            reasons[summary.drop_reason] += 1
    work_dir.mkdir(parents=True, exist_ok=True)
    report_path = work_dir / "pii_report.json"
    previous = (
        PIIReport.model_validate_json(report_path.read_text())
        if report_path.exists()
        else PIIReport(fingerprint=policy.fingerprint)
    )
    categories.update(previous.categories)
    reasons.update(previous.drop_reasons)
    report = PIIReport(
        checked_units=previous.checked_units + len(records),
        dropped_count=previous.dropped_count + len(dropped),
        categories=dict(categories),
        drop_reasons=dict(reasons),
        fingerprint=policy.fingerprint,
    )
    report_path.write_text(report.model_dump_json(indent=2))
    if dropped:
        path = work_dir / "pii_dropped.parquet"
        writer = OutputWriter()
        previous_drops = writer.read_parquet(path) if path.exists() else []
        writer.write_dropped_parquet(records=[*previous_drops, *dropped], path=path)
    return kept
