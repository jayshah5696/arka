from __future__ import annotations

from arka.pipeline.stages import Stage
from arka.records.models import Record


class ExampleStage(Stage):
    name = "01_example"

    def run(self, records: list[Record], ctx) -> list[Record]:
        return records


def test_stage_protocol_exposes_name() -> None:
    stage = ExampleStage()

    assert stage.name == "01_example"
