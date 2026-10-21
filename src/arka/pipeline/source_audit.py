"""Deterministic source audits preserve records and checkpoint each judgment."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field

from arka.common.models import StrictModel
from arka.config.models import SourceAuditConfig
from arka.pipeline.artifacts import StageReport
from arka.pipeline.decision_stages import decision_usage
from arka.pipeline.models import StageContext, StageResult
from arka.pipeline.output import OutputWriter
from arka.pipeline.quality_artifacts import digest, save_json, source_group
from arka.pipeline.quality_judging import QualityJudging, QualityJudgment
from arka.pipeline.stages import Stage
from arka.records.identity import file_hash
from arka.records.models import ConversationRecord, GroundedChunkRecord, Record


class AuditEntry(StrictModel):
    judgment: QualityJudgment | None = None
    reason: Literal["missing_evidence"] | None = None


class AuditState(StrictModel):
    version: Literal[1] = 1
    settings: str
    inputs: str
    sample_ids: list[str]
    entries: dict[str, AuditEntry] = Field(default_factory=dict)


class SourceAuditStage(QualityJudging, Stage):
    name = "03_source_audit"

    def __init__(self, config: SourceAuditConfig, project_root: Path):
        self.config = config
        self.project_root = project_root

    def _settings(self, ctx: StageContext) -> str:
        evidence = self._evidence_path(ctx)
        return digest(
            {
                "config": ctx.config.model_dump(mode="json"),
                "stage": self.config.model_dump(mode="json"),
                "rubric": self._rubric().hash,
                "privacy": ctx.privacy.fingerprint if ctx.privacy else None,
                "evidence": file_hash(evidence) if evidence.exists() else None,
            }
        )

    @staticmethod
    def _inputs(records: list[Record], ctx: StageContext) -> str:
        if ctx.privacy:
            from arka.privacy.records import protect_record

            protected = [protect_record(r, ctx.privacy) for r in records]
            records = [r for r, summary in protected if not summary.drop_reason]
        return digest(
            [
                r.model_dump(mode="json", exclude={"created_at", "stage_events"})
                for r in records
            ]
        )

    def validate_resume(
        self, ctx: StageContext, records: list[Record] | None = None
    ) -> None:
        path = ctx.work_dir / "audit.json"
        if not path.exists():
            return
        if not ctx.resume:
            raise ValueError("Audit artifacts exist; resume or use a new run ID")
        state = AuditState.model_validate_json(path.read_text())
        if state.settings != self._settings(ctx) or (
            records is not None and state.inputs != self._inputs(records, ctx)
        ):
            raise ValueError("Audit inputs or configuration changed; use a new run ID")
        if set(state.entries) - set(state.sample_ids) or len(state.sample_ids) != len(
            set(state.sample_ids)
        ):
            raise ValueError("Audit checkpoint is inconsistent; use a new run ID")

    def run(self, records: list[Record], ctx: StageContext) -> StageResult:
        self.validate_resume(ctx, records)
        records = ctx.protect_records(records)
        if len({r.id for r in records}) != len(records):
            raise ValueError("Source audit does not accept duplicate record IDs")
        rubric = self._rubric()
        populations: dict[str, list[Record]] = {}
        for record in records:
            populations.setdefault(source_group(record), []).append(record)
        selected = []
        by_source = {}
        for group, population in sorted(populations.items()):
            eligible = [
                r
                for r in population
                if isinstance(r, ConversationRecord) and not r.payload.turns
            ]
            sample = sorted(
                eligible, key=lambda r: digest([self.config.sample_seed, group, r.id])
            )[: self.config.sample_size]
            selected.extend(sample)
            by_source[group] = {
                "population": len(population),
                "eligible_population": len(eligible),
                "unsupported": len(population) - len(eligible),
                "sampled": len(sample),
                "census": len(sample) == len(population),
            }
        path = ctx.work_dir / "audit.json"
        if path.exists():
            state = AuditState.model_validate_json(path.read_text())
            if state.sample_ids != [r.id for r in selected]:
                raise ValueError("Audit sample changed; use a new run ID")
        else:
            state = AuditState(
                settings=self._settings(ctx),
                inputs=self._inputs(records, ctx),
                sample_ids=[r.id for r in selected],
            )
            save_json(path, state)
        evidence_path = self._evidence_path(ctx)
        chunks = (
            [
                r
                for r in OutputWriter().read_parquet(evidence_path)
                if isinstance(r, GroundedChunkRecord)
            ]
            if evidence_path.exists()
            else []
        )
        for record in selected:
            if record.id in state.entries:
                continue
            evidence = self._evidence(record, chunks, ctx)
            entry = (
                AuditEntry(reason="missing_evidence")
                if evidence is None
                else AuditEntry(judgment=self._judge(record, evidence, rubric, ctx))
            )
            state.entries[record.id] = entry
            save_json(path, state)
        outcomes: dict[str, Counter] = {group: Counter() for group in by_source}
        for record in selected:
            entry = state.entries[record.id]
            counts = outcomes[source_group(record)]
            if entry.reason:
                counts[entry.reason] += 1
            else:
                judgment = entry.judgment
                counts["judged"] += 1
                if judgment.drop_reason:
                    counts["privacy_rejected"] += 1
                elif judgment.uncertain:
                    counts["uncertain"] += 1
                elif self._failed(judgment):
                    counts["failed"] += 1
                else:
                    counts["passed"] += 1
        for group, stats in by_source.items():
            counts = outcomes[group]
            stats.update(
                {
                    key: counts[key]
                    for key in (
                        "judged",
                        "failed",
                        "passed",
                        "missing_evidence",
                        "uncertain",
                        "privacy_rejected",
                    )
                }
            )
            valid = counts["passed"] + counts["failed"]
            stats["failure_rate"] = counts["failed"] / valid if valid else None
            stats["incomplete"] = valid != stats["sampled"]
        judgments = [
            entry.judgment for entry in state.entries.values() if entry.judgment
        ]
        llm = [j for j in judgments if j.decision is None]
        billable = [
            j for j in judgments if j.decision is None or j.decision.requests_sent
        ]
        costs = [j.usage.cost_usd for j in billable]
        return StageResult(
            kept=records,
            report=StageReport(
                stage=self.name,
                count_in=len(records),
                count_out=len(records),
                by_source=by_source,
                sampled_count=len(selected),
                sample_seed=self.config.sample_seed,
                llm_usage={
                    "requests": len(llm),
                    "prompt_tokens": sum(j.usage.prompt_tokens for j in llm),
                    "completion_tokens": sum(j.usage.completion_tokens for j in llm),
                    "total_tokens": sum(j.usage.total_tokens for j in llm),
                },
                decision_usage=decision_usage(
                    [j.decision for j in judgments if j.decision]
                ),
                cost_usd=sum(c for c in costs if c is not None)
                if any(c is not None for c in costs)
                else None,
                cost_complete=all(c is not None for c in costs),
                rate_scope="Observed valid sample judgments only; no automatic intervention",
            ),
        )
