from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from arka.pipeline.output import OutputWriter
from arka.records.identity import content_hash, file_hash
from arka.records.models import ConversationRecord, GroundedChunkRecord, Record


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Invalid run or stage identifier: {value!r}")
    return value


def run_directory(project_root: Path, run_id: str) -> Path:
    runs = (project_root / "runs").resolve()
    path = runs / identifier(run_id)
    if path.resolve().parent != runs:
        raise ValueError("Run identifier resolves outside the runs directory")
    return path


@dataclass(frozen=True)
class RecordContext:
    source_key: str
    evidence: str | None = None


@dataclass
class Population:
    parent_run_id: str
    stage: str
    records: list[ConversationRecord]
    contexts: dict[str, RecordContext]
    fingerprint: str

    @property
    def groups(self) -> dict[str, list[ConversationRecord]]:
        groups: dict[str, list[ConversationRecord]] = {}
        for record in self.records:
            groups.setdefault(self.contexts[record.id].source_key, []).append(record)
        return groups


def load_population(
    project_root: Path, parent_run_id: str, stage: str | None
) -> Population:
    """Resolve persisted ancestry only; never reread the original raw source files."""
    directory = run_directory(project_root, parent_run_id)
    manifest = json.loads((directory / "manifest.json").read_text())
    stages = manifest.get("stage_names", [])
    if not stages or not all(isinstance(s, str) for s in stages):
        raise ValueError("Parent run has no saved stage list")
    selected = identifier(stage or stages[-1])
    if selected not in stages:
        raise ValueError(f"Unknown parent stage: {selected}")
    writer = OutputWriter()
    index: dict[str, Record] = {}
    fingerprints: dict[str, str] = {}
    # A rebuilt run carries the ancestry run references needed by later audits.
    ancestors = manifest.get("ancestry_runs", [])
    if not isinstance(ancestors, list) or not all(
        isinstance(x, str) for x in ancestors
    ):
        raise ValueError("Invalid ancestry run references")
    for ancestor in [*ancestors, parent_run_id]:
        ancestor_dir = run_directory(project_root, ancestor)
        config_path = ancestor_dir / "config.resolved.yaml"
        if config_path.exists():
            saved_config = yaml.safe_load(config_path.read_text())
            if not isinstance(saved_config, dict):
                raise ValueError("Parent resolved configuration must be a mapping")
            if saved_config.get("pii") is not None:
                raise ValueError(
                    "PII-protected parent runs require the existing source_audit and quality_repair pipeline stages; standalone recovery does not yet support PII policies"
                )
            fingerprints[f"{ancestor}/config.resolved.yaml"] = file_hash(config_path)
        ancestor_manifest = json.loads((ancestor_dir / "manifest.json").read_text())
        names = ancestor_manifest["stage_names"]
        if ancestor == parent_run_id:
            names = stages[: stages.index(selected) + 1]
        for name in names:
            path = ancestor_dir / "stages" / identifier(name) / "data.parquet"
            if not path.exists():
                raise ValueError(f"Missing source artifact: {path}")
            fingerprints[f"{ancestor}/{name}"] = file_hash(path)
            for item in writer.read_parquet(path):
                previous = index.get(item.id)
                # Cleanup stages may change a conversation payload while keeping
                # its id. Chunk evidence must have one unambiguous persisted value.
                if isinstance(previous, GroundedChunkRecord) and (
                    previous.export_payload() != item.export_payload()
                    or previous.source.source_hash != item.source.source_hash
                ):
                    raise ValueError(
                        f"Ambiguous changed source evidence for record {item.id}"
                    )
                index[item.id] = item
    selected_path = directory / "stages" / selected / "data.parquet"
    records = [
        r
        for r in writer.read_parquet(selected_path)
        if isinstance(r, ConversationRecord)
    ]
    if not records:
        raise ValueError("Selected stage contains no conversation records")
    if len({r.id for r in records}) != len(records):
        raise ValueError("Selected stage contains duplicate record identifiers")
    contexts = {r.id: _context(r, index) for r in records}
    return Population(parent_run_id, selected, records, contexts, digest(fingerprints))


def _context(record: ConversationRecord, index: dict[str, Record]) -> RecordContext:
    if content_hash(record.payload) != record.content_hash:
        raise ValueError(
            f"Record content has changed without updated identity: {record.id}"
        )
    if record.source.seed_file_hash and not record.source.doc_id:
        return RecordContext(f"seed:{record.source.seed_file_hash}")
    pending = list(record.lineage.parent_ids or [record.lineage.root_id])
    visited = {record.id}
    chunks: dict[str, GroundedChunkRecord] = {}
    while pending:
        parent_id = pending.pop()
        if parent_id in visited:
            continue
        visited.add(parent_id)
        parent = index.get(parent_id)
        if parent is None:
            raise ValueError(f"Missing source ancestor evidence: {parent_id}")
        if isinstance(parent, GroundedChunkRecord):
            # PDFSourceStage hashes chunk text, unlike conversation payloads.
            if (
                hashlib.sha256(parent.payload.text.encode()).hexdigest()
                != parent.content_hash
            ):
                raise ValueError(f"PDF source evidence changed: {parent_id}")
            chunks[parent.id] = parent
        else:
            pending.extend(parent.lineage.parent_ids)
    if len(chunks) != 1:
        raise ValueError(
            f"Exact PDF source evidence is missing or ambiguous: {record.id}"
        )
    chunk = next(iter(chunks.values()))
    if not chunk.source.source_hash:
        raise ValueError(
            f"PDF source evidence lacks its original file hash: {chunk.id}"
        )
    if record.source.doc_id != chunk.source.doc_id:
        raise ValueError(
            f"PDF source evidence belongs to a different document: {record.id}"
        )
    return RecordContext(f"pdf:{chunk.source.source_hash}", chunk.payload.text)
