from __future__ import annotations

from pydantic import Field

from arka.common.models import StrictModel


class PIISummary(StrictModel):
    matched_bytes: int = 0
    total_bytes: int = 0
    categories: dict[str, int] = Field(default_factory=dict)
    drop_reason: str | None = None


class PIIReport(StrictModel):
    checked_units: int = 0
    dropped_count: int = 0
    categories: dict[str, int] = Field(default_factory=dict)
    drop_reasons: dict[str, int] = Field(default_factory=dict)
    fingerprint: str
