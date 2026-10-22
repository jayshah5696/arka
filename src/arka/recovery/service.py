from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from arka.labeling.rubric import RubricLoader
from arka.llm.models import TokenUsage
from arka.pipeline.output import OutputWriter
from arka.records.identity import content_hash, record_id
from arka.records.models import ConversationRecord, RecordLineage, RecordScores
from arka.recovery.journal import Journal
from arka.recovery.models import (
    Assessment,
    Candidate,
    CandidateOutcome,
    RecoveryConfig,
    RecoveryReport,
    SavedCandidate,
    SourceSummary,
)
from arka.recovery.providers import (
    PROMPT_VERSION,
    DecisionFactory,
    LLMFactory,
    RecoveryProviders,
)
from arka.recovery.sources import Population, digest, load_population, run_directory


class RecoveryService:
    """Audit persisted records and rebuild an explicitly selected source population."""

    def __init__(
        self,
        config: RecoveryConfig,
        *,
        project_root: Path,
        config_root: Path,
        llm_factory: LLMFactory | None = None,
        decision_factory: DecisionFactory | None = None,
    ):
        self.config = config
        self.project_root = project_root.resolve()
        self.config_root = config_root.resolve()
        self.llm_factory = llm_factory
        self.decision_factory = decision_factory
        self.writer = OutputWriter()

    def _providers(self) -> RecoveryProviders:
        # Reload on every operation so a changed rubric cannot evade resume checks.
        rubric = RubricLoader().load(self.config_root / self.config.judge.rubric_path)
        factories = {}
        if self.llm_factory is not None:
            factories["llm_factory"] = self.llm_factory
        if self.decision_factory is not None:
            factories["decision_factory"] = self.decision_factory
        return RecoveryProviders(self.config, rubric, **factories)

    def _judge_hash(self, providers: RecoveryProviders) -> str:
        return digest(
            [
                PROMPT_VERSION,
                self.config.judge.model_dump(mode="json"),
                providers.rubric.model_dump(mode="json"),
            ]
        )

    def audit(
        self,
        parent_run_id: str,
        run_id: str,
        *,
        stage: str | None = None,
        resume: bool = False,
    ) -> RecoveryReport:
        directory = run_directory(self.project_root, run_id)
        population = load_population(self.project_root, parent_run_id, stage)
        providers = self._providers()
        for record in population.records:
            providers.validate_context(record, population.contexts[record.id])
        identity = {
            "version": PROMPT_VERSION,
            "operation": "audit",
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "parent_stage": population.stage,
            "population_hash": population.fingerprint,
            "judge_hash": self._judge_hash(providers),
            "judge": self.config.judge.model_dump(mode="json"),
            "rubric": providers.rubric.model_dump(mode="json"),
            "sample_size": self.config.sample_size,
            "sample_seed": self.config.sample_seed,
        }
        samples = {
            key: sorted(
                records, key=lambda r: digest([self.config.sample_seed, key, r.id])
            )[: self.config.sample_size]
            for key, records in sorted(population.groups.items())
        }
        report = RecoveryReport(
            run_id=run_id,
            operation="audit",
            parent_run_id=parent_run_id,
            parent_stage=population.stage,
            total_count=len(population.records),
        )
        try:
            with Journal(directory, identity, resume=resume) as journal:
                try:
                    for records in samples.values():
                        for record in records:
                            key = f"original:{record.id}"
                            if journal.get(key, Assessment) is None:
                                assessment = providers.judge(
                                    record, population.contexts[record.id]
                                )
                                journal.save(key, assessment)
                    self._audit_report(report, journal, population, samples)
                    journal.report(report, ancestry_runs=self._ancestry(population))
                    return report
                except BaseException as exc:
                    self._audit_report(report, journal, population, samples)
                    self._failure(report, journal, population, exc)
                    raise
        finally:
            providers.close()

    def _audit_report(
        self,
        report: RecoveryReport,
        journal: Journal,
        population: Population,
        samples: dict[str, list[ConversationRecord]],
    ) -> None:
        report.sources = []
        groups = population.groups
        for key, records in samples.items():
            completed = [
                (r.id, journal.get(f"original:{r.id}", Assessment)) for r in records
            ]
            completed = [
                (rid, assessment)
                for rid, assessment in completed
                if assessment is not None
            ]
            failed = sum(not a.passed for _, a in completed)
            report.sources.append(
                SourceSummary(
                    key=key,
                    total_count=len(groups[key]),
                    sampled_count=len(completed),
                    failed_count=failed,
                    failure_rate=failed / len(completed) if completed else None,
                    sample_ids=[rid for rid, _ in completed],
                )
            )
        assessments = journal.values("original:", Assessment)
        report.sampled_count = len(assessments)
        report.failed_count = sum(not a.passed for a in assessments)
        report.assessed_count = len(assessments)
        self._usage(report, journal)

    def rebuild(
        self,
        audit_run_id: str,
        run_id: str,
        *,
        sources: list[str],
        resume: bool = False,
    ) -> RecoveryReport:
        replacement = self.config.replacement
        if replacement is None:
            raise ValueError("A replacement configuration is required for rebuild")
        directory = run_directory(self.project_root, run_id)
        audit_directory = run_directory(self.project_root, audit_run_id)
        audit_identity = json.loads((audit_directory / "operation.json").read_text())
        if audit_identity.get("operation") != "audit":
            raise ValueError("The referenced operation is not a source audit")
        audit_report = RecoveryReport.model_validate_json(
            (audit_directory / "report/recovery_report.json").read_text()
        )
        if audit_report.status != "completed":
            raise ValueError("Resume and complete the audit before rebuilding sources")
        population = load_population(
            self.project_root,
            audit_identity["parent_run_id"],
            audit_identity["parent_stage"],
        )
        providers = self._providers()
        if (
            population.fingerprint != audit_identity["population_hash"]
            or self._judge_hash(providers) != audit_identity["judge_hash"]
        ):
            raise ValueError(
                "Audit input, source evidence, or judge configuration changed; incompatible rebuild"
            )
        selected = sorted(set(sources))
        if not selected or set(selected) - population.groups.keys():
            raise ValueError("Select one or more source keys from the audit report")
        selected_records = [
            r
            for r in population.records
            if population.contexts[r.id].source_key in selected
        ]
        for record in selected_records:
            providers.validate_context(record, population.contexts[record.id])
        identity = {
            "version": PROMPT_VERSION,
            "operation": "rebuild",
            "run_id": run_id,
            "parent_run_id": population.parent_run_id,
            "parent_stage": population.stage,
            "audit_run_id": audit_run_id,
            "audit_hash": digest(audit_identity),
            "population_hash": population.fingerprint,
            "judge_hash": self._judge_hash(providers),
            "replacement": replacement.model_dump(mode="json"),
            "selected_sources": selected,
        }
        report = RecoveryReport(
            run_id=run_id,
            operation="rebuild",
            parent_run_id=population.parent_run_id,
            parent_stage=population.stage,
            audit_run_id=audit_run_id,
            selected_sources=selected,
            sources=audit_report.sources,
            total_count=len(population.records),
            sampled_count=audit_report.sampled_count,
        )
        try:
            with Journal(directory, identity, resume=resume) as journal:
                try:
                    self._assess_selected(
                        journal,
                        audit_directory,
                        population,
                        selected_records,
                        providers,
                    )
                    original_assessments = {
                        r.id: journal.get(f"original:{r.id}", Assessment)
                        for r in selected_records
                    }
                    retained = [
                        r
                        for r in population.records
                        if r.id not in original_assessments
                        or original_assessments[r.id].passed
                    ]
                    seen = {self._pair_key(r) for r in retained}
                    accepted: dict[str, ConversationRecord] = {}
                    exhausted: list[ConversationRecord] = []
                    for record in selected_records:
                        assessment = original_assessments[record.id]
                        if assessment.passed:
                            continue
                        candidate = self._repair(
                            journal, record, assessment, population, providers, seen
                        )
                        if candidate is None:
                            exhausted.append(
                                self._scored_record(
                                    record, assessment, providers
                                ).dropped_by(
                                    "rebuild",
                                    "rebuild_exhausted",
                                    f"No accepted answer after {replacement.max_attempts} attempts",
                                )
                            )
                        else:
                            accepted[record.id] = candidate
                            seen.add(self._pair_key(candidate))
                    output = []
                    exhausted_ids = {r.id for r in exhausted}
                    for record in population.records:
                        if record.id in accepted:
                            output.append(accepted[record.id])
                        elif record.id not in exhausted_ids:
                            output.append(record)
                    self._write_output(directory, output, exhausted, journal, providers)
                    self._rebuild_report(report, journal, population)
                    report.final_count = len(output)
                    report.dataset_path = str(directory / "dataset.jsonl")
                    journal.report(report, ancestry_runs=self._ancestry(population))
                    return report
                except BaseException as exc:
                    self._rebuild_report(report, journal, population)
                    self._failure(report, journal, population, exc)
                    raise
        finally:
            providers.close()

    def _assess_selected(
        self,
        journal: Journal,
        audit_directory: Path,
        population: Population,
        records: list[ConversationRecord],
        providers: RecoveryProviders,
    ) -> None:
        # Read the audit journal without creating or modifying its database.
        connection = sqlite3.connect(
            (audit_directory / "recovery.sqlite3").as_uri() + "?mode=ro", uri=True
        )
        try:
            for record in records:
                key = f"original:{record.id}"
                if journal.get(key, Assessment) is not None:
                    continue
                row = connection.execute(
                    "SELECT value FROM results WHERE key = ?", (key,)
                ).fetchone()
                if row:
                    assessment = Assessment.model_validate_json(row[0])
                    assessment = assessment.model_copy(
                        update={"usage": TokenUsage(), "requests": 0}
                    )
                else:
                    assessment = providers.judge(record, population.contexts[record.id])
                journal.save(key, assessment)
        finally:
            connection.close()

    def _repair(
        self,
        journal: Journal,
        original: ConversationRecord,
        feedback: Assessment,
        population: Population,
        providers: RecoveryProviders,
        seen: set[str],
    ) -> ConversationRecord | None:
        previous = original.payload.response
        for attempt in range(1, self.config.replacement.max_attempts + 1):
            suffix = f"{original.id}:{attempt}"
            saved = journal.get(f"candidate:{suffix}", SavedCandidate)
            if saved is None:
                candidate, output, prompt_hash = providers.generate(
                    original,
                    population.contexts[original.id],
                    previous,
                    feedback,
                    attempt,
                )
                replacement_record = self._replacement_record(
                    original, candidate, attempt, journal.identity_hash
                )
                saved = SavedCandidate(
                    record=replacement_record,
                    usage=output.usage,
                    model=output.model,
                    provider=output.provider,
                    prompt_hash=prompt_hash,
                )
                journal.save(f"candidate:{suffix}", saved)
            candidate = saved.record
            outcome = journal.get(f"outcome:{suffix}", CandidateOutcome)
            assessment = journal.get(f"candidate_judge:{suffix}", Assessment)
            if outcome is None:
                if not candidate.payload.response.strip():
                    outcome = CandidateOutcome(accepted=False, reason="empty_response")
                elif candidate.payload.response == original.payload.response:
                    outcome = CandidateOutcome(
                        accepted=False, reason="unchanged_response"
                    )
                elif (
                    self.config.replacement.exact_dedup
                    and self._pair_key(candidate) in seen
                ):
                    outcome = CandidateOutcome(accepted=False, reason="duplicate")
                else:
                    if assessment is None:
                        assessment = providers.judge(
                            candidate, population.contexts[original.id]
                        )
                        journal.save(f"candidate_judge:{suffix}", assessment)
                    outcome = CandidateOutcome(
                        accepted=assessment.passed,
                        reason="accepted" if assessment.passed else "quality_failed",
                    )
                journal.save(f"outcome:{suffix}", outcome)
            if outcome.accepted:
                return self._scored_record(candidate, assessment, providers)
            previous = candidate.payload.response
            if assessment is not None:
                feedback = assessment
        return None

    @staticmethod
    def _replacement_record(
        original: ConversationRecord,
        candidate: Candidate,
        attempt: int,
        operation_hash: str,
    ) -> ConversationRecord:
        payload = original.payload.model_copy(
            update={"response": candidate.response, "reasoning": candidate.reasoning}
        )
        lineage = RecordLineage(
            root_id=original.lineage.root_id,
            parent_ids=[original.id],
            operator="source_rebuild",
            round=attempt,
            depth=(original.lineage.depth or 0) + 1,
        )
        # seed_domain and taxonomy labels describe the task. Answer scores do not.
        task_metadata = {
            key: value
            for key, value in original.annotations.items()
            if key in {"seed_domain", "taxonomy_nodes", "taxonomy_nodes_requested"}
        }
        replacement = original.model_copy(
            update={
                "id": record_id(payload, lineage),
                "content_hash": content_hash(payload),
                "payload": payload,
                "lineage": lineage,
                "source": original.source.model_copy(update={"type": "generated"}),
                "scores": RecordScores(),
                "stage_events": [],
                "config_hash": operation_hash,
                "created_at": datetime.now(UTC).isoformat(),
            }
        ).with_event(stage="rebuild", action="generated", details=f"attempt={attempt}")
        for key, value in task_metadata.items():
            replacement = replacement.with_annotation(key, value)
        return replacement

    @staticmethod
    def _scored_record(
        record: ConversationRecord, assessment: Assessment, providers: RecoveryProviders
    ) -> ConversationRecord:
        return record.model_copy(
            update={
                "scores": record.scores.model_copy(
                    update={
                        "quality": assessment.score,
                        "quality_per_dim": {
                            **record.scores.quality_per_dim,
                            **assessment.scores,
                        },
                        "rubric_hash": providers.rubric.hash,
                        "rubric_version": providers.rubric.version,
                        "judge_model": assessment.model,
                        "judge_prompt_hash": assessment.prompt_hash,
                    }
                )
            }
        ).with_event(stage="rebuild", action="scored")

    @staticmethod
    def _pair_key(record: ConversationRecord) -> str:
        return digest([record.payload.instruction, record.payload.response])

    def _write_output(
        self,
        directory: Path,
        output: list[ConversationRecord],
        exhausted: list[ConversationRecord],
        journal: Journal,
        providers: RecoveryProviders,
    ) -> None:
        stage_directory = directory / "stages/rebuild"
        attempts = []
        for candidate in journal.values("candidate:", SavedCandidate):
            record = candidate.record
            suffix = f"{record.lineage.parent_ids[0]}:{record.lineage.round}"
            assessment = journal.get(f"candidate_judge:{suffix}", Assessment)
            outcome = journal.get(f"outcome:{suffix}", CandidateOutcome)
            if assessment is not None:
                record = self._scored_record(record, assessment, providers)
            attempts.append(
                record.with_event(
                    stage="rebuild",
                    action="accepted" if outcome.accepted else "rejected",
                    reason_code=outcome.reason,
                )
            )
        for name, records in [
            ("data", output),
            ("dropped", exhausted),
            ("attempts", attempts),
        ]:
            path = stage_directory / f"{name}.parquet"
            temporary = path.with_suffix(".parquet.tmp")
            if name == "dropped":
                self.writer.write_dropped_parquet(records, temporary)
            else:
                self.writer.write_parquet(records, temporary)
            temporary.replace(path)
        dataset = directory / "dataset.jsonl"
        temporary = dataset.with_suffix(".jsonl.tmp")
        self.writer.write_jsonl(output, temporary)
        temporary.replace(dataset)

    def _rebuild_report(
        self, report: RecoveryReport, journal: Journal, population: Population
    ) -> None:
        originals = journal.values("original:", Assessment)
        outcomes = journal.values("outcome:", CandidateOutcome)
        report.assessed_count = len(originals)
        report.failed_count = sum(not a.passed for a in originals)
        report.attempt_count = len(journal.values("candidate:", SavedCandidate))
        report.accepted_replacements = sum(o.accepted for o in outcomes)
        report.duplicate_count = sum(o.reason == "duplicate" for o in outcomes)
        report.exhausted_count = 0
        limit = self.config.replacement.max_attempts
        for record in population.records:
            outcome = journal.get(f"outcome:{record.id}:{limit}", CandidateOutcome)
            report.exhausted_count += int(outcome is not None and not outcome.accepted)
        report.unresolved_count = sum(
            s.failed_count
            for s in report.sources
            if s.key not in report.selected_sources
        )
        groups = population.groups
        for source in report.sources:
            source.selected = source.key in report.selected_sources
            source.assessed_count = 0
            source.full_failed_count = 0
            source.attempt_count = 0
            source.accepted_replacements = 0
            source.exhausted_count = 0
            if not source.selected:
                continue
            for record in groups[source.key]:
                assessment = journal.get(f"original:{record.id}", Assessment)
                source.assessed_count += int(assessment is not None)
                source.full_failed_count += int(
                    assessment is not None and not assessment.passed
                )
                for attempt in range(1, limit + 1):
                    suffix = f"{record.id}:{attempt}"
                    source.attempt_count += int(
                        journal.get(f"candidate:{suffix}", SavedCandidate) is not None
                    )
                    outcome = journal.get(f"outcome:{suffix}", CandidateOutcome)
                    source.accepted_replacements += int(
                        outcome is not None and outcome.accepted
                    )
                    if attempt == limit:
                        source.exhausted_count += int(
                            outcome is not None and not outcome.accepted
                        )
        self._usage(report, journal)

    @staticmethod
    def _usage(report: RecoveryReport, journal: Journal) -> None:
        assessments = journal.values("original:", Assessment) + journal.values(
            "candidate_judge:", Assessment
        )
        candidates = journal.values("candidate:", SavedCandidate)
        usages = [a.usage for a in assessments if a.requests] + [
            c.usage for c in candidates
        ]
        report.requests = sum(a.requests for a in assessments) + len(candidates)
        costs = [u.cost_usd for u in usages if u.cost_usd is not None]
        report.unknown_cost_calls = sum(u.cost_usd is None for u in usages)
        report.cost_complete = not report.unknown_cost_calls
        report.usage = TokenUsage(
            prompt_tokens=sum(u.prompt_tokens for u in usages),
            completion_tokens=sum(u.completion_tokens for u in usages),
            total_tokens=sum(u.total_tokens for u in usages),
            cost_usd=sum(costs) if costs else None,
        )

    def _ancestry(self, population: Population) -> list[str]:
        manifest = json.loads(
            (
                run_directory(self.project_root, population.parent_run_id)
                / "manifest.json"
            ).read_text()
        )
        return list(
            dict.fromkeys(
                [*manifest.get("ancestry_runs", []), population.parent_run_id]
            )
        )

    def _failure(
        self,
        report: RecoveryReport,
        journal: Journal,
        population: Population,
        exc: BaseException,
    ) -> None:
        report.status = "failed"
        # Do not persist exception text that could include provider request data.
        report.error = type(exc).__name__
        journal.report(report, ancestry_runs=self._ancestry(population))
