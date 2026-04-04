from __future__ import annotations

from abc import ABC, abstractmethod

from arka.pipeline.models import StageContext
from arka.records.models import Record


class Stage(ABC):
    name: str

    @abstractmethod
    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        """Transform records for a pipeline stage."""
