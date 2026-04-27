"""StageArtifacts: the typed seam between Stages and the on-disk per-stage files.

A Stage emits two kinds of Artifact: a typed ``stats.json`` (described by
:class:`StageReport`) and an optional ``dropped.parquet`` of dropped Records.
Some stages also write extra artifacts (e.g. ``clusters.parquet`` for dedup);
those are exposed via ``extras``.

Before this module existed, every Stage hand-rolled the same ``stats = {...};
(work_dir / "stats.json").write_text(json.dumps(...))`` block, and the runner
re-parsed those dicts defensively. The schema was real but invisible. Now the
schema is :class:`StageReport` -- one Pydantic model that both producer
(Stage) and consumer (PipelineRunner) round-trip through.

Keep field names stable: downstream tests and the run report read the JSON
directly, so changing a field name here breaks compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.records.models import Record


class StageReport(BaseModel):
    """The typed contract behind every per-stage ``stats.json``.

    ``model_config(extra="allow")`` lets a Stage attach a small number of
    well-known optional fields without inventing a subclass per Stage. Today's
    optional fields: ``cost_usd``, ``quality_distribution``,
    ``reward_distribution``, ``ifd_distribution``, ``cluster_count``,
    ``scored_count``, ``generated_count``. They are accessed via
    ``model_dump()`` so callers see them as plain dict entries, exactly the
    way they did when ``stats.json`` was a hand-built dict.
    """

    model_config = ConfigDict(extra="allow")

    stage: str
    count_in: int
    count_out: int
    dropped_count: int = 0
    drop_reasons: dict[str, int] = Field(default_factory=dict)


class StageArtifacts:
    """Single seam for writing per-Stage Artifacts to the Run Directory.

    Usage from inside a Stage::

        StageArtifacts(ctx).write(
            report=StageReport(
                stage=self.name,
                count_in=count_in,
                count_out=len(kept),
                dropped_count=len(dropped),
                drop_reasons=drop_reasons,
                cost_usd=total_cost,                # optional extras as kwargs
            ),
            dropped=dropped,
            extras={"clusters.parquet": clusters_df},  # optional
        )

    The class deliberately stays small: it owns ``stats.json``,
    ``dropped.parquet`` and any extras a Stage names. It does *not* own the
    main per-stage ``data.parquet`` -- that one is written by ``PipelineRunner``
    after the Stage returns its kept Records.
    """

    def __init__(self, ctx: StageContext, writer: OutputWriter | None = None) -> None:
        self._ctx = ctx
        self._writer = writer or OutputWriter()

    def write(
        self,
        *,
        report: StageReport,
        dropped: list[Record] | None = None,
        extras: dict[str, pl.DataFrame] | None = None,
    ) -> None:
        self._ctx.work_dir.mkdir(parents=True, exist_ok=True)
        if dropped is not None:
            # Match the historical behaviour: filter stages skipped writing
            # dropped.parquet entirely when the dropped list was empty, while
            # other stages always wrote it. Preserve "always write" -- callers
            # that want to skip can pass dropped=None.
            self._writer.write_dropped_parquet(
                records=dropped,
                path=self._ctx.work_dir / "dropped.parquet",
            )
        if extras:
            for filename, df in extras.items():
                df.write_parquet(self._ctx.work_dir / filename)
        (self._ctx.work_dir / "stats.json").write_text(
            report.model_dump_json(indent=2, exclude_none=False)
        )

    @staticmethod
    def load_report(path: Path) -> StageReport | None:
        """Read a previously-written ``stats.json`` back into a StageReport.

        Returns ``None`` when the file does not exist (e.g. a Stage failed
        before writing). Used by ``PipelineRunner`` while building per-stage
        ``StageStat`` summaries and the run manifest.
        """
        if not path.exists():
            return None
        return StageReport.model_validate_json(path.read_text())


def get_extra(report: StageReport | None, key: str, default: Any = None) -> Any:
    """Read an optional extras field from a StageReport (or ``None``).

    Convenience for the runner, which currently asks for ``cost_usd`` and
    ``quality_distribution`` without caring whether a given Stage emitted
    them.
    """
    if report is None:
        return default
    return report.model_dump().get(key, default)
