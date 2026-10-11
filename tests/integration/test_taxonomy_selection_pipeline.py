import json
from pathlib import Path

from arka.config.models import ResolvedConfig
from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.pipeline.runner import PipelineRunner
from arka.pipeline.stage_builder import StageBuilder
from arka.pipeline.stages import Stage
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    Record,
    RecordLineage,
    RecordScores,
    RecordSource,
)


class TaggedSource(Stage):
    name = "01_source"

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        return [
            ConversationRecord(
                id=topic,
                content_hash=topic,
                source=RecordSource(type="generated"),
                lineage=RecordLineage(root_id=topic, parent_ids=[]),
                payload=ConversationPayload(instruction=topic, response=topic),
                scores=RecordScores(
                    quality=score,
                    quality_per_dim={"taxonomy_nodes": {"topic": [topic]}},
                ),
                config_hash="test",
                created_at="2026-10-10T00:00:00Z",
            )
            for topic, score in (("science", 0.1), ("arts", 1), ("business", 0.9))
        ]


def test_taxonomy_selection_exports_and_resumes_protected_records(
    tmp_path: Path,
) -> None:
    config = ResolvedConfig(
        version="1",
        llm={
            "provider": "openai",
            "model": "test",
            "api_key": "test",
            "base_url": "https://api.openai.com/v1",
        },
        executor={"mode": "threadpool", "max_workers": 1},
        pipeline=[
            {
                "type": "select",
                "target_count": 2,
                "weights": {"quality": 1},
                "coverage": [{"factor": "topic", "path": ["science"]}],
            }
        ],
        output={"format": "chatml", "path": "dataset.jsonl"},
    )
    stages = [TaggedSource(), *StageBuilder(config, tmp_path).build()]
    runner = PipelineRunner(tmp_path)
    runner.run(config, stages, "coverage")
    rows = [
        json.loads(line)
        for line in (tmp_path / "dataset.jsonl").read_text().splitlines()
    ]
    assert [row["messages"][0]["content"] for row in rows] == ["arts", "science"]
    stage_dir = tmp_path / "runs/coverage/stages/05_composite_select"
    kept = OutputWriter().read_parquet(stage_dir / "data.parquet")
    assert kept[1].scores.quality_per_dim["taxonomy_nodes"] == {"topic": ["science"]}
    dropped = OutputWriter().read_parquet(stage_dir / "dropped.parquet")
    assert [record.id for record in dropped] == ["business"]
    before = (tmp_path / "dataset.jsonl").read_bytes()
    runner.run(config, stages, "coverage", resume=True)
    assert (tmp_path / "dataset.jsonl").read_bytes() == before
