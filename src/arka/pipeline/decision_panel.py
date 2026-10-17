from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from arka.common.concurrency import bounded_worker_count
from arka.config.models import DecisionPanelConfig
from arka.decisions.factory import resolve_decision_config
from arka.pipeline.artifacts import StageArtifacts, StageReport
from arka.pipeline.decision_stages import (
    attach_answers,
    decision_questions,
    decision_usage,
    record_state,
)
from arka.pipeline.models import StageContext
from arka.pipeline.stages import Stage
from arka.records.models import Record


class DecisionPanelStage(Stage):
    name = "02p_decision_panel"
    stage_action = "scored"

    def __init__(self, config: DecisionPanelConfig, project_root: Path):
        self.config = config
        self.project_root = project_root

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        from arka.decisions.client import DecisionClientError

        # Only prefetch later stages with identical provider settings and evidence.
        index = next(
            i for i, cfg in enumerate(ctx.config.pipeline) if cfg is self.config
        )
        effective = resolve_decision_config(
            ctx.config.decisions, self.config.decision_override
        )
        later = [
            cfg
            for cfg in ctx.config.pipeline[index + 1 :]
            if getattr(cfg, "backend", None) == "decision"
            and (self.config.include == "auto" or cfg.type in self.config.include)
            and resolve_decision_config(ctx.config.decisions, cfg.decision_override)
            == effective
        ]
        out, used, failures = [], [], 0
        with ctx.decision_client(override=self.config.decision_override) as client:

            def evaluate(record):
                record_used, record_failures = [], 0
                questions = {}
                for cfg in later:
                    for question in decision_questions(cfg, record, self.project_root):
                        existing = questions.get(question.name)
                        if existing is not None and existing != question:
                            raise ValueError(
                                f"Panel has conflicting questions named {question.name}"
                            )
                        questions[question.name] = question
                shared = [q for q in questions.values() if not q.exclude_from_panel]
                groups = [
                    shared,
                    *[[q] for q in questions.values() if q.exclude_from_panel],
                ]
                updated = record
                for group in groups:
                    if not group:
                        continue
                    try:
                        result = client.decide(record_state(record), group)
                        record_used.append(result)
                        updated = attach_answers(updated, result.answers)
                    except DecisionClientError:
                        # A later decision stage retries directly. Never switch to an LLM.
                        record_failures += 1
                return (
                    updated,
                    record_used,
                    record_failures,
                    [len(group) for group in groups if group],
                )

            pool = ctx.executor or ThreadPoolExecutor(
                max_workers=bounded_worker_count(len(records), ctx.max_workers)
            )
            try:
                evaluated = list(pool.map(evaluate, records))
            finally:
                if ctx.executor is None:
                    pool.shutdown()
            question_counts = []
            for updated, results, failed, counts in evaluated:
                out.append(updated)
                used.extend(results)
                failures += failed
                question_counts.extend(counts)
        usage = decision_usage(used)
        StageArtifacts(ctx).write(
            report=StageReport(
                stage=self.name,
                count_in=len(records),
                count_out=len(out),
                decision_usage=usage,
                decision_failures=failures,
                records_sent=len(records),
                requests_sent=usage["requests"],
                questions_per_request={
                    "mean": sum(question_counts) / len(question_counts)
                    if question_counts
                    else 0,
                    "max": max(question_counts, default=0),
                },
                cost_usd=usage["cost_usd"],
            )
        )
        return out
