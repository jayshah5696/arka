"""Canonical identity for Records, Pipelines, and seed files.

Three rules previously scattered across Stages and the runner:

  - ``content_hash(payload)``: sha256 of a Record's payload, used by
    Exact Dedup to decide whether two Records are the "same".
  - ``record_id(payload, lineage)``: sha256 of payload + lineage, used as
    the Record's primary key. A Record with the same payload but a
    different parent (different Lineage) gets a different id.
  - ``config_hash(config)``: sha256 of a ResolvedConfig dump, used by
    Generators to scope Checkpoint resume and by the runner for the Run
    Manifest's ``config_hash``.

Plus one boundary helper: ``file_hash(path)`` for source seed files.

Before this module existed each rule was inlined in three or four places.
Changing identity semantics required hunting through the pipeline package
and hoping you got them all -- and the runner's ``_config_hash`` was the
same algorithm written twice.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from arka.config.models import ResolvedConfig
from arka.records.models import RecordLineage


def content_hash(payload: object) -> str:
    """Stable hash of a Record payload (e.g. ConversationPayload).

    Accepts any Pydantic model. Hashes the canonical JSON dump with
    ``exclude_none=True`` so optional payload fields do not destabilise the
    hash when they happen to be unset.
    """
    return hashlib.sha256(
        payload.model_dump_json(exclude_none=True).encode("utf-8")  # type: ignore[attr-defined]
    ).hexdigest()


def record_id(payload: object, lineage: RecordLineage | None = None) -> str:
    """Stable id derived from payload + lineage.

    ``lineage=None`` is treated as an empty source lineage (no parents,
    no operator), matching the historical ``SeedSourceStage`` behaviour.
    """
    if lineage is None:
        lineage_payload = {
            "parent_ids": [],
            "operator": None,
            "round": None,
            "depth": None,
        }
    else:
        lineage_payload = lineage.model_dump(mode="json", exclude_none=True)
    identity_payload = {
        "payload": payload.model_dump(mode="json", exclude_none=True),  # type: ignore[attr-defined]
        "lineage": lineage_payload,
    }
    return hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def config_hash(config: ResolvedConfig) -> str:
    """Stable hash of a ResolvedConfig used for Run identity and Checkpoint scope.

    Used by ``PipelineRunner`` for the Manifest's ``config_hash`` and by
    Generators that scope their resume buckets per resolved config.
    """
    return hashlib.sha256(
        json.dumps(config.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def file_hash(path: Path) -> str:
    """Sha256 of a file's bytes, used to fingerprint seed source files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
