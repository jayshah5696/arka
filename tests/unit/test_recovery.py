"""Recovery failure cases, specified before implementation.

Cover wrong source grouping, sampling drift, missing PDF evidence, incompatible
resume, repeated provider work, stale scores, ineffective retries, accidental
changes to unselected sources, hidden failures, and unsafe run identifiers.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import yaml
from pydantic import ValidationError

from arka.llm.models import LLMOutput, TokenUsage
from arka.pipeline.output import OutputWriter
from arka.records.identity import content_hash, record_id
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    GroundedChunkPayload,
    GroundedChunkRecord,
    RecordLineage,
    RecordScores,
    RecordSource,
)
from arka.recovery.models import RecoveryConfig
from arka.recovery.service import RecoveryService


def record(name, response="good", source="a"):
    payload = ConversationPayload(instruction=name, response=response)
    return ConversationRecord(
        id=name,
        content_hash=content_hash(payload),
        payload=payload,
        source=RecordSource(type="seed", seed_file_hash=source),
        lineage=RecordLineage(root_id=name, parent_ids=[]),
        scores=RecordScores(ifd=0.9, quality_per_dim={"seed_domain": "science"}),
        config_hash="parent",
        created_at="2026-10-21T00:00:00Z",
    )


def parent(root, records, chunks=()):
    run = root / "runs" / "parent"
    writer = OutputWriter()
    writer.write_parquet(list(chunks), run / "stages" / "source" / "data.parquet")
    writer.write_parquet(records, run / "stages" / "candidates" / "data.parquet")
    writer.write_parquet(records[:1], run / "stages" / "final" / "data.parquet")
    (run / "manifest.json").write_text(
        json.dumps(
            {"run_id": "parent", "stage_names": ["source", "candidates", "final"]}
        )
    )


def config(root, **overrides):
    rubric = {
        "version": "1",
        "description": "Answer quality",
        "dimensions": [
            {
                "name": "correct",
                "description": "Correct answer",
                "scale_min": 1,
                "scale_max": 3,
                "criteria": {1: "Wrong", 2: "Partial", 3: "Correct"},
            }
        ],
        "overall_weights": {"correct": 1.0},
    }
    (root / "rubric.yaml").write_text(yaml.safe_dump(rubric))
    llm = {
        "provider": "openai",
        "model": "fixture",
        "api_key": "private-test-key",
        "base_url": "http://127.0.0.1:1/v1",
    }
    values = {
        "judge": {
            "backend": "llm",
            "rubric_path": "rubric.yaml",
            "min_overall_score": 3,
            "llm": llm,
        },
        "replacement": {"llm": llm, "max_attempts": 2},
        "sample_size": 200,
        "sample_seed": 7,
    }
    values.update(overrides)
    return RecoveryConfig.model_validate(values)


class FakeLLM:
    def __init__(self, replacements=(), fail_judge_at=None):
        self.replacements = iter(replacements)
        self.fail_judge_at = fail_judge_at
        self.judged = []
        self.generated = []

    def complete_structured(self, messages, schema, **kwargs):
        state = json.loads(messages[-1]["content"])
        if "scores" in schema.model_fields:
            if len(self.judged) == self.fail_judge_at:
                raise RuntimeError("provider unavailable")
            self.judged.append(state)
            parsed = schema(
                scores={"correct": 3 if state["response"].startswith("good") else 1},
                reasoning="Evaluate the answer",
            )
        else:
            self.generated.append(state)
            parsed = schema(
                response=next(self.replacements), reasoning="Fresh reasoning"
            )
        return LLMOutput(
            parsed=parsed,
            model="fixture",
            provider="openai",
            latency_ms=1,
            usage=TokenUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.01
            ),
        )


def service(root, cfg, fake):
    return RecoveryService(
        cfg, project_root=root, config_root=root, llm_factory=lambda _: fake
    )


def test_default_final_stage_and_explicit_candidates(tmp_path):
    parent(tmp_path, [record("one"), record("two")])
    cleaned = record("one", "good cleaned")
    OutputWriter().write_parquet(
        [cleaned], tmp_path / "runs/parent/stages/final/data.parquet"
    )
    svc = service(tmp_path, config(tmp_path), FakeLLM())
    final = svc.audit("parent", "audit-final")
    candidates = svc.audit("parent", "audit-candidates", stage="candidates")
    assert (final.total_count, candidates.total_count) == (1, 2)
    assert final.parent_stage == "final"


def test_sampling_is_repeatable_and_smaller_sources_are_complete(tmp_path):
    records = [record(f"a-{i}") for i in range(20)] + [record("b", source="b")]
    parent(tmp_path, records)
    fake = FakeLLM()
    svc = service(tmp_path, config(tmp_path, sample_size=3), fake)
    first = svc.audit("parent", "audit-a", stage="candidates")
    sampled = [x["instruction"] for x in fake.judged]
    fake.judged.clear()
    svc.audit("parent", "audit-b", stage="candidates")
    assert sampled == [x["instruction"] for x in fake.judged]
    assert sorted(s.sampled_count for s in first.sources) == [1, 3]
    assert first.sampled_count == 4


def test_rebuild_only_failed_selected_records_and_reuse_judgments(tmp_path):
    originals = [
        record("keep"),
        record("fix", "bad"),
        record("unselected", "bad", "b"),
        record("exhaust", "bad"),
    ]
    originals[1] = originals[1].with_annotation(
        "taxonomy_nodes", {"topic": ["science"]}
    )
    parent(tmp_path, originals)
    fake = FakeLLM(["bad again", "good repaired", "bad still", "bad last"])
    cfg = config(tmp_path)
    svc = service(tmp_path, cfg, fake)
    audit = svc.audit("parent", "audit", stage="candidates")
    source = next(s.key for s in audit.sources if s.total_count == 3)
    report = svc.rebuild("audit", "rebuild", sources=[source])
    output = OutputWriter().read_parquet(
        tmp_path / "runs/rebuild/stages/rebuild/data.parquet"
    )
    assert [r.payload.instruction for r in output] == ["keep", "fix", "unselected"]
    assert output[0] == originals[0] and output[2] == originals[2]
    replacement = output[1]
    assert replacement.payload.response == "good repaired"
    assert replacement.id != originals[1].id
    assert replacement.id == record_id(replacement.payload, replacement.lineage)
    assert replacement.content_hash == content_hash(replacement.payload)
    assert replacement.lineage.parent_ids == [originals[1].id]
    assert replacement.scores.ifd is None
    assert replacement.scores.quality == 3
    assert replacement.scores.quality_per_dim["seed_domain"] == "science"
    assert replacement.scores.annotations["taxonomy_nodes"] == {"topic": ["science"]}
    assert replacement.payload.reasoning == "Fresh reasoning"
    assert report.accepted_replacements == 1 and report.exhausted_count == 1
    assert report.unresolved_count == 1 and report.final_count == 3
    assert report.attempt_count == 4 and report.assessed_count == 3
    assert len(fake.judged) == 8  # Four audit judgments, four candidate judgments.
    assert fake.generated[1]["previous_response"] == "bad again"
    assert report.usage.total_tokens == 120  # Reused audit work is not charged again.
    assert report.usage.cost_usd == pytest.approx(0.08)
    exported = [
        json.loads(line)
        for line in (tmp_path / "runs/rebuild/dataset.jsonl").read_text().splitlines()
    ]
    assert exported[1]["response"] == "good repaired"
    assert (
        "private-test-key" not in (tmp_path / "runs/rebuild/operation.json").read_text()
    )


def test_full_assessment_judges_unsampled_selected_records(tmp_path):
    parent(tmp_path, [record(str(i), "bad") for i in range(4)])
    fake = FakeLLM(["good"] * 4)
    svc = service(tmp_path, config(tmp_path, sample_size=1), fake)
    audit = svc.audit("parent", "audit", stage="candidates")
    result = svc.rebuild("audit", "rebuilt", sources=[audit.sources[0].key])
    assert len(fake.judged) == 8
    assert result.accepted_replacements == 4


def test_provider_failure_resumes_after_saved_candidate_without_regeneration(tmp_path):
    parent(tmp_path, [record("fix", "bad")])
    cfg = config(tmp_path)
    fake = FakeLLM(["good fixed"], fail_judge_at=1)
    svc = service(tmp_path, cfg, fake)
    audit = svc.audit("parent", "audit")
    with pytest.raises(RuntimeError, match="provider unavailable"):
        svc.rebuild("audit", "rebuilt", sources=[audit.sources[0].key])
    fake.fail_judge_at = None
    result = svc.rebuild(
        "audit", "rebuilt", sources=[audit.sources[0].key], resume=True
    )
    assert len(fake.generated) == 1 and len(fake.judged) == 2
    assert result.accepted_replacements == 1
    svc.rebuild("audit", "rebuilt", sources=[audit.sources[0].key], resume=True)
    assert len(fake.generated) == 1 and len(fake.judged) == 2


def test_audit_resume_does_not_repeat_completed_judgments(tmp_path):
    parent(tmp_path, [record("first"), record("second")])
    fake = FakeLLM(fail_judge_at=1)
    svc = service(tmp_path, config(tmp_path), fake)
    with pytest.raises(RuntimeError):
        svc.audit("parent", "audit", stage="candidates")
    fake.fail_judge_at = None
    svc.audit("parent", "audit", stage="candidates", resume=True)
    assert len(fake.judged) == 2


@pytest.mark.parametrize("change", ["input", "rubric", "model", "attempts", "stage"])
def test_resume_rejects_changed_inputs_or_configuration(tmp_path, change):
    parent(tmp_path, [record("fix", "bad")])
    cfg = config(tmp_path)
    svc = service(tmp_path, cfg, FakeLLM(["good"]))
    audit = svc.audit("parent", "audit")
    source = audit.sources[0].key
    svc.rebuild("audit", "rebuilt", sources=[source])
    if change == "input":
        OutputWriter().write_parquet(
            [record("changed")], tmp_path / "runs/parent/stages/final/data.parquet"
        )
    elif change == "rubric":
        path = tmp_path / "rubric.yaml"
        path.write_text(path.read_text().replace("Correct answer", "A new criterion"))
    elif change == "model":
        cfg.replacement.llm.model = "different"
    elif change == "attempts":
        cfg.replacement.max_attempts = 3
    with pytest.raises(ValueError, match="changed|incompatible"):
        if change == "stage":
            svc.audit("parent", "audit", stage="candidates", resume=True)
        else:
            svc.rebuild("audit", "rebuilt", sources=[source], resume=True)


def pdf_records():
    payload = GroundedChunkPayload(
        text="Exact evidence: the answer is 42.",
        doc_id="document",
        chunk_idx=0,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=36,
        word_count=7,
        chunk_strategy="fixed",
    )
    chunk = GroundedChunkRecord(
        id="chunk",
        content_hash=hashlib.sha256(payload.text.encode()).hexdigest(),
        payload=payload,
        source=RecordSource(
            type="pdf_chunk", doc_id="document", source_hash="pdf-file-hash"
        ),
        lineage=RecordLineage(root_id="chunk", parent_ids=[]),
        scores=RecordScores(),
        config_hash="parent",
        created_at="2026-10-21T00:00:00Z",
    )
    generated = record("What is the answer?", "bad")
    generated.source = RecordSource(
        type="generated",
        doc_id="document",
        chunk_id="document:0",
        source_hash=chunk.content_hash,
    )
    generated.lineage = RecordLineage(
        root_id="chunk", parent_ids=["chunk"], operator="prompt_based", round=1, depth=1
    )
    return chunk, generated


def test_pdf_group_and_exact_evidence_reach_judge_and_replacement(tmp_path):
    chunk, generated = pdf_records()
    parent(tmp_path, [generated], [chunk])
    fake = FakeLLM(["good 42"])
    svc = service(tmp_path, config(tmp_path), fake)
    audit = svc.audit("parent", "audit")
    assert audit.sources[0].key == "pdf:pdf-file-hash"
    svc.rebuild("audit", "rebuilt", sources=[audit.sources[0].key])
    assert all(
        x["source_evidence"] == chunk.payload.text for x in fake.judged + fake.generated
    )


def test_missing_pdf_evidence_fails_before_provider_call(tmp_path):
    _, generated = pdf_records()
    parent(tmp_path, [generated])
    fake = FakeLLM()
    with pytest.raises(ValueError, match="evidence|ancestor"):
        service(tmp_path, config(tmp_path), fake).audit("parent", "audit")
    assert not fake.judged


def test_changed_pdf_evidence_invalidates_rebuild(tmp_path):
    chunk, generated = pdf_records()
    parent(tmp_path, [generated], [chunk])
    fake = FakeLLM()
    svc = service(tmp_path, config(tmp_path), fake)
    audit = svc.audit("parent", "audit")
    chunk.payload.text = "Changed evidence"
    OutputWriter().write_parquet(
        [chunk], tmp_path / "runs/parent/stages/source/data.parquet"
    )
    with pytest.raises(ValueError, match="changed|evidence"):
        svc.rebuild("audit", "rebuilt", sources=[audit.sources[0].key])


def test_duplicate_candidate_retries_and_exhausts(tmp_path):
    first = record("same", "good existing", "b")
    second = record("same", "bad", "a")
    second.id = "other-id"
    second.lineage.root_id = second.id
    parent(tmp_path, [first, second])
    fake = FakeLLM(["good existing", "good existing"])
    svc = service(tmp_path, config(tmp_path), fake)
    svc.audit("parent", "audit", stage="candidates")
    result = svc.rebuild("audit", "rebuilt", sources=["seed:a"])
    assert result.final_count == 1 and result.exhausted_count == 1
    assert result.attempt_count == 2 and result.duplicate_count == 2


@pytest.mark.parametrize("run_id", ["../parent", "/tmp/escape", "a/b", ".", ".."])
def test_unsafe_run_ids_rejected(tmp_path, run_id):
    parent(tmp_path, [record("one")])
    with pytest.raises(ValueError, match="identifier"):
        service(tmp_path, config(tmp_path), FakeLLM()).audit("parent", run_id)


def test_unknown_source_and_existing_operation_rejected(tmp_path):
    parent(tmp_path, [record("one")])
    fake = FakeLLM()
    svc = service(tmp_path, config(tmp_path), fake)
    svc.audit("parent", "audit")
    with pytest.raises(ValueError, match="exists"):
        svc.audit("parent", "audit")
    with pytest.raises(ValueError, match="source"):
        svc.rebuild("audit", "rebuilt", sources=["absent"])
    assert len(fake.judged) == 1


def test_rebuild_rejects_changed_audit_judge(tmp_path):
    parent(tmp_path, [record("one")])
    cfg = config(tmp_path)
    svc = service(tmp_path, cfg, FakeLLM())
    audit = svc.audit("parent", "audit")
    cfg.judge.min_overall_score = 2
    with pytest.raises(ValueError, match="changed|incompatible"):
        svc.rebuild("audit", "rebuilt", sources=[audit.sources[0].key])


@pytest.mark.parametrize(
    "override",
    [
        {"sample_size": 0},
        {"sample_seed": -1},
        {"replacement": {"max_attempts": 0}},
        {
            "judge": {
                "backend": "llm",
                "rubric_path": "r",
                "min_overall_score": float("nan"),
            }
        },
    ],
)
def test_invalid_limits_and_missing_backend_configuration(tmp_path, override):
    with pytest.raises(ValidationError):
        config(tmp_path, **override)


def test_empty_or_unsupported_population_fails(tmp_path):
    parent(tmp_path, [])
    with pytest.raises(ValueError, match="conversation"):
        service(tmp_path, config(tmp_path), FakeLLM()).audit("parent", "audit")


def test_replacement_required_only_for_rebuild(tmp_path):
    parent(tmp_path, [record("one")])
    cfg = config(tmp_path, replacement=None)
    svc = service(tmp_path, cfg, FakeLLM())
    audit = svc.audit("parent", "audit")
    with pytest.raises(ValueError, match="replacement"):
        svc.rebuild("audit", "rebuilt", sources=[audit.sources[0].key])
