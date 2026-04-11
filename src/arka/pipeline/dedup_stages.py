from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
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

        # PERF: Added LSH bucketing to reduce NearDedupStage from O(n^2) to O(n) average case
        lsh_index = LSHIndex(
            num_bands=ctx.config.dedup.near.num_bands,
            rows_per_band=ctx.config.dedup.near.rows_per_band,
        )
        representative_signatures: dict[str, list[int]] = {}

        for record in records:
            if not isinstance(record, ConversationRecord):
                kept_records.append(record)
                continue

            record_tokens = _tokenize(record.payload.instruction)
            if not record_tokens:
                kept_records.append(record)
                continue

            record_signature = _minhash_signature(
                tokens=record_tokens,
                shingle_size=ctx.config.dedup.near.shingle_size,
                num_hashes=ctx.config.dedup.near.num_hashes,
            )

            matched_cluster_id: str | None = None
            matched_reason: str | None = None

            candidate_cluster_ids = lsh_index.query(record_signature)

            for cluster_id in candidate_cluster_ids:
                representative_signature = representative_signatures.get(cluster_id)
                if not representative_signature:
                    continue

                similarity = _minhash_similarity(
                    representative_signature, record_signature
                )
                if similarity >= ctx.config.dedup.near.jaccard_threshold:
                    matched_cluster_id = cluster_id
                    matched_reason = "near_duplicate_minhash"
                    break

            if matched_cluster_id is None or matched_reason is None:
                cluster_id = self._cluster_id(record.payload.instruction)
                representatives[cluster_id] = record
                representative_signatures[cluster_id] = record_signature
                lsh_index.insert(cluster_id, record_signature)
                cluster_members[cluster_id] = [record]
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

    def _near_duplicate_reason(
        self,
        *,
        left: str,
        right: str,
        ctx: StageContext,
    ) -> str | None:
        left_tokens = _tokenize(left)
        right_tokens = _tokenize(right)
        if not left_tokens or not right_tokens:
            return None

        threshold = ctx.config.dedup.near.jaccard_threshold
        shingle_size = ctx.config.dedup.near.shingle_size
        num_hashes = ctx.config.dedup.near.num_hashes
        left_signature = _minhash_signature(
            tokens=left_tokens,
            shingle_size=shingle_size,
            num_hashes=num_hashes,
        )
        right_signature = _minhash_signature(
            tokens=right_tokens,
            shingle_size=shingle_size,
            num_hashes=num_hashes,
        )
        if _minhash_similarity(left_signature, right_signature) >= threshold:
            return "near_duplicate_minhash"
        return None

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


class LSHIndex:
    def __init__(self, num_bands: int, rows_per_band: int):
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.bands: list[dict[tuple[int, ...], set[str]]] = [
            defaultdict(set) for _ in range(num_bands)
        ]

    def insert(self, record_id: str, signature: list[int]) -> None:
        if not signature:
            return
        for i in range(self.num_bands):
            start = i * self.rows_per_band
            end = start + self.rows_per_band
            if end > len(signature):
                break
            band_tuple = tuple(signature[start:end])
            self.bands[i][band_tuple].add(record_id)

    def query(self, signature: list[int]) -> set[str]:
        candidates: set[str] = set()
        if not signature:
            return candidates
        for i in range(self.num_bands):
            start = i * self.rows_per_band
            end = start + self.rows_per_band
            if end > len(signature):
                break
            band_tuple = tuple(signature[start:end])
            candidates.update(self.bands[i].get(band_tuple, []))
        return candidates


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
