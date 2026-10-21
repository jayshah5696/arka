"""Small shared helpers for quality operation identities and source grouping."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from arka.common.models import StrictModel
from arka.records.models import Record


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_json(path: Path, value: StrictModel) -> None:
    """Persist a completed operation step with an atomic file replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(value.model_dump_json(indent=2))
    temporary.replace(path)


def source_group(record: Record) -> str:
    source = record.source
    if source.doc_id:
        return f"document:{source.doc_id}"
    if source.seed_file_hash:
        return f"seeds:{source.seed_file_hash}"
    return "ungrouped"
