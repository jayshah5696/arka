from __future__ import annotations

import hashlib
import json
import re
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

        _write_artifacts(
            stage_name=self.name,
            output_writer=self._output_writer,
            ctx=ctx,
            dropped_records=dropped_records,
            clusters=clusters,
            count_in=len(records),
            count_out=len(kept_records),
            drop_reasons=drop_reasons,
        )
        return kept_records

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


class NearDedupStage(Stage):
    name = "02d_near_dedup"
    stage_action = "deduplicated"

    def __init__(self) -> None:
        self._output_writer = OutputWriter()

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        if not ctx.config.dedup.near.enabled:
            return records

        kept_records: list[Record] = []
        dropped_records: list[Record] = []
        clusters: list[dict[str, Any]] = []
        cluster_members: dict[str, list[ConversationRecord]] = {}
        representatives: dict[str, ConversationRecord] = {}
        drop_reasons: dict[str, int] = {}

        # PERF: [Add LSH bucketing to NearDedupStage] [Reduce O(n^2) comparisons by bucketing MinHash signatures] [reduces NearDedup from O(n^2) to O(n) average case]
        from collections import defaultdict

        threshold = ctx.config.dedup.near.jaccard_threshold
        shingle_size = ctx.config.dedup.near.shingle_size
        num_hashes = ctx.config.dedup.near.num_hashes
        num_bands = ctx.config.dedup.near.num_bands
        rows_per_band = ctx.config.dedup.near.rows_per_band

        lsh_index: dict[str, list[str]] = defaultdict(list)
        representative_signatures: dict[str, list[int]] = {}

        for record in records:
            if not isinstance(record, ConversationRecord):
                kept_records.append(record)
                continue

            tokens = _tokenize(record.payload.instruction)
            if not tokens:
                cluster_id = self._cluster_id(record.payload.instruction)
                representatives[cluster_id] = record
                cluster_members[cluster_id] = [record]
                kept_records.append(record)
                continue

            signature = _minhash_signature(
                tokens=tokens,
                shingle_size=shingle_size,
                num_hashes=num_hashes,
            )
            bands = self._lsh_bands(signature, num_bands, rows_per_band)

            matched_cluster_id: str | None = None
            matched_reason: str | None = None

            # Find candidate cluster IDs that share at least one band
            candidate_cluster_ids = set()
            for band in bands:
                for candidate_id in lsh_index[band]:
                    candidate_cluster_ids.add(candidate_id)

            # Only compare exact similarity with candidates
            for cluster_id in sorted(candidate_cluster_ids):
                rep_sig = representative_signatures[cluster_id]
                if _minhash_similarity(rep_sig, signature) >= threshold:
                    matched_cluster_id = cluster_id
                    matched_reason = "near_duplicate_minhash"
                    break

            if matched_cluster_id is None or matched_reason is None:
                cluster_id = self._cluster_id(record.payload.instruction)
                representatives[cluster_id] = record
                cluster_members[cluster_id] = [record]
                representative_signatures[cluster_id] = signature
                for band in bands:
                    lsh_index[band].append(cluster_id)
                kept_records.append(record)
                continue

            representative = representatives[matched_cluster_id]
            cluster_members[matched_cluster_id].append(record)
            dropped_records.append(
                self._drop_record(
                    record=record,
                    reason_code=matched_reason,
                    details=f"duplicate_of={representative.id}",
                )
            )
            drop_reasons[matched_reason] = drop_reasons.get(matched_reason, 0) + 1

        for cluster_id, members in cluster_members.items():
            if len(members) < 2:
                continue
            representative = representatives[cluster_id]
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "representative_id": representative.id,
                    "member_count": len(members),
                    "member_ids_json": json.dumps(
                        [member.id for member in members], separators=(",", ":")
                    ),
                }
            )

        _write_artifacts(
            stage_name=self.name,
            output_writer=self._output_writer,
            ctx=ctx,
            dropped_records=dropped_records,
            clusters=clusters,
            count_in=len(records),
            count_out=len(kept_records),
            drop_reasons=drop_reasons,
        )
        return kept_records

    def _lsh_bands(
        self, signature: list[int], num_bands: int, rows_per_band: int
    ) -> list[str]:
        if not signature:
            return []
        bands = []
        for i in range(num_bands):
            start = i * rows_per_band
            end = start + rows_per_band
            band_tuple = tuple(signature[start:end])
            bands.append(f"b{i}:{hash(band_tuple)}")
        return bands

    def _cluster_id(self, instruction: str) -> str:
        return hashlib.sha256(instruction.strip().encode("utf-8")).hexdigest()

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


def _write_artifacts(
    *,
    stage_name: str,
    output_writer: OutputWriter,
    ctx: StageContext,
    dropped_records: list[Record],
    clusters: list[dict[str, Any]],
    count_in: int,
    count_out: int,
    drop_reasons: dict[str, int],
) -> None:
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    output_writer.write_dropped_parquet(
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
        "stage": stage_name,
        "count_in": count_in,
        "count_out": count_out,
        "dropped_count": len(dropped_records),
        "drop_reasons": drop_reasons,
        "cluster_count": len(clusters),
    }
    (ctx.work_dir / "stats.json").write_text(json.dumps(stats, indent=2))


_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _feature_hash(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _shingles(tokens: list[str], size: int) -> set[tuple[str, ...]]:
    if not tokens:
        return set()
    if len(tokens) < size:
        return {tuple(tokens)}
    return {
        tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)
    }


def _minhash_signature(
    *,
    tokens: list[str],
    shingle_size: int,
    num_hashes: int,
) -> list[int]:
    shingles = _shingles(tokens, shingle_size)
    if not shingles:
        return []
    signature: list[int] = []
    for salt in range(num_hashes):
        signature.append(
            min(_feature_hash(f"{salt}:{' '.join(shingle)}") for shingle in shingles)
        )
    return signature


def _minhash_similarity(left_signature: list[int], right_signature: list[int]) -> float:
    if not left_signature or not right_signature:
        return 0.0
    matches = sum(
        1
        for left, right in zip(left_signature, right_signature, strict=True)
        if left == right
    )
    return matches / len(left_signature)
