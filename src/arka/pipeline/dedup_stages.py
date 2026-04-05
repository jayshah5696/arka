from __future__ import annotations

import json
from typing import Any

import polars as pl

from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.pipeline.stages import Stage
from arka.records.models import ConversationRecord, Record, StageEvent


class ExactDedupStage(Stage):
    name = "02c_exact_dedup"
    stage_action = "deduplicated"

    def __init__(self) -> None:
        self._output_writer = OutputWriter()

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        if not ctx.config.dedup.exact.enabled:
            return records

        seen_content_hashes: dict[str, Record] = {}
        kept_records: list[Record] = []
        dropped_records: list[Record] = []
        clusters: list[dict[str, Any]] = []
        cluster_members: dict[str, list[Record]] = {}
        drop_reasons: dict[str, int] = {}

        for record in records:
            if not isinstance(record, ConversationRecord):
                kept_records.append(record)
                continue

            representative = seen_content_hashes.get(record.content_hash)
            if representative is None:
                seen_content_hashes[record.content_hash] = record
                kept_records.append(record)
                cluster_members[record.content_hash] = [record]
                continue

            cluster_members[record.content_hash].append(record)
            dropped_records.append(
                self._drop_record(
                    record=record,
                    reason_code="exact_duplicate",
                    details=f"duplicate_of={representative.id}",
                )
            )
            drop_reasons["exact_duplicate"] = drop_reasons.get("exact_duplicate", 0) + 1

        for content_hash, members in cluster_members.items():
            if len(members) < 2:
                continue
            representative = seen_content_hashes[content_hash]
            clusters.append(
                {
                    "cluster_id": content_hash,
                    "representative_id": representative.id,
                    "member_count": len(members),
                    "member_ids_json": json.dumps(
                        [member.id for member in members], separators=(",", ":")
                    ),
                }
            )

        self._write_artifacts(
            ctx=ctx,
            dropped_records=dropped_records,
            clusters=clusters,
            count_in=len(records),
            count_out=len(kept_records),
            drop_reasons=drop_reasons,
        )
        return kept_records

    def _write_artifacts(
        self,
        *,
        ctx: StageContext,
        dropped_records: list[Record],
        clusters: list[dict[str, Any]],
        count_in: int,
        count_out: int,
        drop_reasons: dict[str, int],
    ) -> None:
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        self._output_writer.write_dropped_parquet(
            records=dropped_records,
            path=ctx.work_dir / "dropped.parquet",
        )
        pl.DataFrame(
            clusters,
            schema={
                "cluster_id": pl.String,
                "representative_id": pl.String,
                "member_count": pl.Int64,
                "member_ids_json": pl.String,
            },
        ).write_parquet(ctx.work_dir / "clusters.parquet")
        stats = {
            "stage": self.name,
            "count_in": count_in,
            "count_out": count_out,
            "dropped_count": len(dropped_records),
            "drop_reasons": drop_reasons,
            "cluster_count": len(clusters),
        }
        (ctx.work_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    def _drop_record(self, record: Record, reason_code: str, details: str) -> Record:
        return record.model_copy(
            update={
                "stage_events": [
                    *record.stage_events,
                    StageEvent(
                        stage=self.name,
                        action="dropped",
                        reason_code=reason_code,
                        details=details,
                        seq=len(record.stage_events) + 1,
                    ),
                ]
            }
        )
