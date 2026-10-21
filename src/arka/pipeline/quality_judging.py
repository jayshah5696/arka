"""Shared evidence-preserving judging for repair and source audits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from arka.common.models import StrictModel
from arka.common.security import sanitize_for_prompt
from arka.decisions.models import DecisionResult, ScoreAnswer
from arka.labeling.judges import JudgeResponse
from arka.labeling.prompting import build_single_judge_messages
from arka.labeling.rubric import Rubric, RubricLoader
from arka.llm.models import TokenUsage
from arka.pipeline.decision_stages import decision_questions, judge_answers
from arka.pipeline.models import StageContext
from arka.records.models import ConversationRecord, GroundedChunkRecord


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class QualityJudgment(StrictModel):
    scores: dict[str, float]
    reasoning: str | None = None
    model: str
    provider: str
    prompt_hash: str
    usage: TokenUsage
    decision: DecisionResult | None = None
    uncertain: bool = False
    drop_reason: str | None = None


class QualityJudging:
    """Use the owning stages bound config and project root."""

    def _rubric(self) -> Rubric:
        rubric = RubricLoader().load(self.project_root / self.config.rubric_path)
        dimensions = {d.name: d for d in rubric.dimensions}
        if set(self.config.min_scores) - dimensions.keys():
            raise ValueError("quality_repair min_scores contains unknown dimensions")
        if (
            set(getattr(self.config, "repair_dimensions", self.config.min_scores))
            - self.config.min_scores.keys()
        ):
            raise ValueError("repair_dimensions must be included in min_scores")
        for name, threshold in self.config.min_scores.items():
            dim = dimensions[name]
            if not dim.scale_min <= threshold <= dim.scale_max:
                raise ValueError(f"quality_repair threshold outside scale for {name}")
        if (
            not rubric.dimensions
            or len(dimensions) != len(rubric.dimensions)
            or any(d.scale_min >= d.scale_max for d in rubric.dimensions)
            or any(w < 0 for w in rubric.overall_weights.values())
            or sum(rubric.overall_weights.values()) <= 0
        ):
            raise ValueError("quality_repair requires dimensions and positive weights")
        return rubric

    def _evidence_path(self, ctx: StageContext) -> Path:
        return ctx.work_dir.parent / self.config.evidence_stage / "data.parquet"

    @staticmethod
    def _llm_client(ctx: StageContext, override: object) -> object:
        # Attempt files already checkpoint completed judgments and repairs.
        # Disable prompt-addressed SQLite caching so identical prompts across
        # records or repaired candidates never collide on fresh client instances.
        client = ctx.llm_client(override=override)
        target = getattr(client, "client", client)
        if hasattr(target, "cache_db_path"):
            target.cache_db_path = None
        if hasattr(target, "read_cache"):
            target.read_cache = False
        return client

    def _evidence(
        self,
        record: ConversationRecord,
        chunks: list[GroundedChunkRecord],
        ctx: StageContext,
    ) -> str | None:
        if not (record.source.doc_id or record.source.chunk_id):
            return ""
        matches = [
            c
            for c in chunks
            if c.id == record.lineage.root_id
            and c.source.doc_id == record.source.doc_id
            and c.source.chunk_id == record.source.chunk_id
        ]
        if len(matches) != 1:
            return None
        text = matches[0].payload.text
        if ctx.privacy is not None:
            result = ctx.privacy.protect(text)
            if result.drop_reason:
                return None
            text = result.value
        return text

    @staticmethod
    def _protect_messages(
        messages: list[dict[str, str]], ctx: StageContext
    ) -> list[dict[str, str]]:
        if ctx.privacy is None:
            return messages
        return [
            {**m, "content": ctx.privacy.protect(m["content"]).value} for m in messages
        ]

    def _judge(
        self,
        record: ConversationRecord,
        evidence: str,
        rubric: Rubric,
        ctx: StageContext,
    ) -> QualityJudgment:
        messages = build_single_judge_messages(
            record.payload.instruction, record.payload.response, rubric
        )
        messages[-1]["content"] += (
            f"\nTask system message: <text>{sanitize_for_prompt(record.payload.system or '')}</text>"
            f"\nPermitted source evidence: <text>{sanitize_for_prompt(evidence)}</text>"
            "\nFor a grounded task, check that the answer is supported by this evidence."
        )
        messages = self._protect_messages(messages, ctx)
        decision = None
        if self.config.backend == "decision":
            questions = decision_questions(self.config, record, self.project_root)
            state = json.dumps(
                {
                    "instruction": record.payload.instruction,
                    "response": record.payload.response,
                    "system": record.payload.system,
                    "evidence": evidence,
                },
                ensure_ascii=False,
            )
            if ctx.privacy is not None:
                state = ctx.privacy.protect(state).value
            client = ctx.decision_client(override=self.config.decision_override)
            try:
                # DecisionClient truncates state. Grounded evidence must be complete.
                if len(state) > ctx.config.decisions.max_state_chars:
                    raise ValueError(
                        "Quality repair evidence exceeds decision max_state_chars"
                    )
                decision = client.decide(state, questions)
            finally:
                client.close()
            if set(decision.answers) != {q.name for q in questions} or any(
                not isinstance(decision.answers[q.name], ScoreAnswer)
                or len(decision.answers[q.name].probabilities) != len(q.levels)
                for q in questions
            ):
                raise ValueError(
                    "Decision judge scores must match rubric dimensions and scales"
                )
            scored, outcome, _ = judge_answers(
                record, self.config, decision.answers, self.project_root
            )
            scores = {
                d.name: scored.scores.quality_per_dim[d.name] for d in rubric.dimensions
            }
            judgment = QualityJudgment(
                scores=scores,
                model=decision.model,
                provider=decision.provider,
                prompt_hash=_hash([state, [q.model_dump() for q in questions]]),
                usage=decision.usage,
                decision=decision,
                uncertain=outcome == "uncertain",
            )
        else:
            output = self._llm_client(
                ctx, self.config.judge_llm_override
            ).complete_structured(messages=messages, schema=JudgeResponse)
            if not isinstance(output.parsed, JudgeResponse):
                raise ValueError("Quality judge returned invalid scores")
            judgment = QualityJudgment(
                scores=output.parsed.scores,
                reasoning=output.parsed.reasoning,
                model=output.model,
                provider=output.provider,
                prompt_hash=_hash(messages),
                usage=output.usage,
                drop_reason=output.privacy.drop_reason if output.privacy else None,
            )
        dimensions = {d.name: d for d in rubric.dimensions}
        if set(judgment.scores) != dimensions.keys() or any(
            not dimensions[n].scale_min <= s <= dimensions[n].scale_max
            for n, s in judgment.scores.items()
        ):
            raise ValueError(
                "Quality judge scores must match rubric dimensions and scales"
            )
        return judgment

    def _failed(self, judgment: QualityJudgment) -> list[str]:
        return [
            n
            for n, minimum in self.config.min_scores.items()
            if judgment.scores[n] < minimum
        ]
