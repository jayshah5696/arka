"""Decision checks used inside sources, generators, and pair filters."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from arka.config.models import (
    BestOfNConfig,
    DecisionCheckConfig,
    EvolInstructGeneratorConfig,
    FidelityCheckConfig,
    PairwiseCheckConfig,
    TaxonomyLabelConfig,
)
from arka.decisions.models import (
    ChoiceQuestion,
    DecisionRequest,
    DecisionResult,
    ScoreQuestion,
    YesNoQuestion,
)
from arka.pipeline.decision_stages import apply_uncertain, attach_answers, record_state
from arka.pipeline.models import StageContext
from arka.records.models import ConversationRecord, Record


def filter_pdf_chunks(
    records: list[Record], ctx: StageContext, config: DecisionCheckConfig, stage: str
) -> tuple[list[Record], list[Record], list[DecisionResult]]:
    question = YesNoQuestion(
        name="pdf_source.is_front_or_back_matter",
        instructions="Is this chunk mainly navigation or front/back matter, such as contents, index, bibliography, copyright or a list of figures, rather than substantive body text?",
    )
    with ctx.decision_client(override=config.decision_override) as client:
        results = client.decide_many(
            [
                DecisionRequest(state=record_state(r), questions=[question])
                for r in records
            ],
            ctx.max_workers,
        )
    kept, dropped = [], []
    for record, result in zip(records, results, strict=True):
        updated = attach_answers(record, result.answers)
        if result.answers[question.name].probability >= config.reject_at:
            dropped.append(updated.dropped_by(stage, "pdf_front_back_matter"))
        else:
            kept.append(updated)
    return kept, dropped, results


def best_of_n(
    records: list[Record],
    ctx: StageContext,
    config: BestOfNConfig,
    stage: str,
    *,
    kind: str = "prompt_based_generator",
) -> tuple[list[Record], list[Record], list[DecisionResult]]:
    questions = [
        ScoreQuestion(
            name=f"{kind}.candidate_quality",
            instructions="How good is this instruction and response pair as supervised fine-tuning data?",
            levels=["Unusable", "Poor", "Acceptable", "Good", "Excellent"],
        ),
        YesNoQuestion(
            name=f"{kind}.answers_instruction",
            instructions="Does the response directly and fully answer the instruction?",
        ),
    ]
    with ctx.decision_client(override=config.decision_override) as client:
        results = client.decide_many(
            [
                DecisionRequest(state=record_state(r), questions=questions)
                for r in records
            ],
            ctx.max_workers,
        )
    groups = defaultdict(list)
    for index, (record, result) in enumerate(zip(records, results, strict=True)):
        updated = attach_answers(record, result.answers)
        group = tuple(record.lineage.parent_ids) or json.dumps(
            record.scores.quality_per_dim.get("taxonomy_nodes", {}), sort_keys=True
        )
        groups[group].append(
            (
                index,
                updated,
                result.answers[questions[0].name].score,
                result.answers[questions[1].name].probability,
            )
        )
    winners = set()
    for candidates in groups.values():
        candidates.sort(key=lambda row: (-row[2], -row[3], row[0]))
        winners.update(
            row[0]
            for row in candidates[: config.keep_per_seed]
            if config.min_score is None or row[2] >= config.min_score
        )
    updated_by_index = {
        row[0]: row[1] for candidates in groups.values() for row in candidates
    }
    kept = [updated_by_index[i] for i in range(len(records)) if i in winners]
    dropped = [
        updated_by_index[i].dropped_by(stage, "best_of_n_not_selected")
        for i in range(len(records))
        if i not in winners
    ]
    return kept, dropped, results


def taxonomy_fidelity(
    records: list[Record], ctx: StageContext, config: FidelityCheckConfig, stage: str
) -> tuple[list[Record], list[Record], list[DecisionResult]]:
    kept, dropped, results = [], [], []
    with ctx.decision_client(override=config.decision_override) as client:
        for record in records:
            nodes = record.scores.quality_per_dim.get("taxonomy_nodes", {})
            questions = [
                YesNoQuestion(
                    name=f"taxonomy_generator.reflects_{factor}",
                    instructions=f"Does this example clearly belong to {factor}: {' > '.join(path)}?",
                )
                for factor, path in nodes.items()
            ]
            if not questions:
                kept.append(record)
                continue
            result = client.decide(record_state(record), questions)
            results.append(result)
            updated = attach_answers(record, result.answers)
            failed = [
                factor
                for factor in nodes
                if result.answers[f"taxonomy_generator.reflects_{factor}"].probability
                < config.reject_at
            ]
            uncertain = [
                factor
                for factor in nodes
                if config.reject_at
                <= result.answers[f"taxonomy_generator.reflects_{factor}"].probability
                < config.accept_at
            ]
            if failed and config.relabel:
                dims = {
                    **updated.scores.quality_per_dim,
                    "taxonomy_nodes_requested": nodes,
                    "taxonomy_nodes": {
                        factor: path
                        for factor, path in nodes.items()
                        if factor not in failed
                    },
                }
                updated = updated.model_copy(
                    update={
                        "scores": updated.scores.model_copy(
                            update={"quality_per_dim": dims}
                        )
                    }
                )
            elif failed:
                dropped.append(
                    updated.dropped_by(
                        stage, "taxonomy_node_mismatch", json.dumps(failed)
                    )
                )
                continue
            if uncertain:
                updated, keep = apply_uncertain(
                    updated,
                    stage,
                    config,
                    "taxonomy_fidelity_uncertain",
                    json.dumps(uncertain),
                )
                if not keep:
                    dropped.append(updated)
                    continue
            kept.append(updated)
    return kept, dropped, results


def check_evolution(
    child: ConversationRecord,
    parent: ConversationRecord,
    ctx: StageContext,
    config: EvolInstructGeneratorConfig,
    stage: str,
) -> tuple[Record, str | None, DecisionResult]:
    questions = [
        YesNoQuestion(
            name="evol_instruct_generator.is_refusal",
            instructions="Does the candidate response refuse, deflect or avoid answering the instruction?",
        ),
        YesNoQuestion(
            name="evol_instruct_generator.harder_than_parent",
            instructions="Does the candidate instruction ask meaningfully more than the parent through added constraints or reasoning, rather than paraphrasing?",
        ),
        YesNoQuestion(
            name="evol_instruct_generator.still_solvable",
            instructions="Can a competent expert answer the candidate instruction without missing information or contradictions?",
        ),
    ]
    state = json.dumps(
        {
            "parent_instruction": parent.payload.instruction,
            "candidate": child.export_payload(),
        },
        sort_keys=True,
    )
    with ctx.decision_client(
        override=config.filter.decision_override or config.decision_override
    ) as client:
        result = client.decide(state, questions)
    updated = attach_answers(child, result.answers)
    if result.answers[questions[0].name].probability >= config.filter.refusal_reject_at:
        return updated, "evol_refusal", result
    for question, reason in zip(
        questions[1:], ["evol_not_harder", "evol_unsolvable"], strict=True
    ):
        p = result.answers[question.name].probability
        if p < config.filter.reject_at:
            return updated, reason, result
        if p < config.filter.evolution_accept_at:
            updated, keep = apply_uncertain(
                updated,
                stage,
                config.filter,
                reason + "_uncertain",
                json.dumps({question.name: p}),
            )
            if not keep:
                return updated, reason + "_uncertain", result
    return updated, None, result


def pairwise_preference(
    child: ConversationRecord,
    parent: ConversationRecord,
    ctx: StageContext,
    config: PairwiseCheckConfig,
    stage: str,
) -> tuple[Record, str | None, list[DecisionResult]]:
    question = ChoiceQuestion(
        name="pair_delta.better_response",
        instructions="Which response better answers the instruction with correctness, completeness and helpfulness? Do not prefer length.",
        choices={
            "A": "Response A is better",
            "B": "Response B is better",
            "tie": "Equally good",
        },
    )
    results, preference, ties = [], [], []
    with ctx.decision_client(override=config.decision_override) as client:
        for swapped in [False, True] if config.swap_order else [False]:
            state = json.dumps(
                {
                    "instruction": child.payload.instruction,
                    "A": parent.payload.response if swapped else child.payload.response,
                    "B": child.payload.response if swapped else parent.payload.response,
                },
                sort_keys=True,
            )
            result = client.decide(state, [question])
            results.append(result)
            distribution = result.answers[question.name].probabilities
            preference.append(distribution["B" if swapped else "A"])
            ties.append(distribution["tie"])
    p_child = sum(preference) / len(preference)
    agree = len(preference) == 1 or abs(preference[0] - preference[1]) <= 0.4
    dims = {
        **child.scores.quality_per_dim,
        "pair_preference": {
            "p_child_better": p_child,
            "p_tie": sum(ties) / len(ties),
            "orders_agree": agree,
        },
    }
    updated = attach_answers(child, results[0].answers)
    updated = updated.model_copy(
        update={"scores": updated.scores.model_copy(update={"quality_per_dim": dims})}
    )
    reason = (
        "pairwise_position_inconsistent"
        if not agree
        else "pairwise_not_preferred"
        if p_child < config.min_preference
        else None
    )
    return updated, reason, results


def label_taxonomy(
    records: list[Record],
    ctx: StageContext,
    config: TaxonomyLabelConfig,
    project_root: Path,
) -> tuple[list[Record], list[DecisionResult]]:
    from arka.taxonomy.models import TaxonomyBundle

    bundle = TaxonomyBundle.from_yaml(project_root / Path(config.taxonomy_path))
    out, used = [], []
    with ctx.decision_client(override=config.decision_override) as client:
        for record in records:
            if record.scores.quality_per_dim.get("taxonomy_nodes"):
                out.append(record)
                continue
            labels, answers = {}, {}
            for factor in bundle.factors:
                paths = factor.leaf_paths()
                if len(paths) <= 255:
                    choices = {" > ".join(path): " / ".join(path) for path in paths}
                    if not choices:
                        continue
                    question = ChoiceQuestion(
                        name=f"select.taxonomy_{factor.factor}",
                        instructions=f"Which leaf path in factor {factor.factor} best describes this example?",
                        choices=choices,
                    )
                    result = client.decide(record_state(record), [question])
                    used.append(result)
                    answers.update(result.answers)
                    answer = result.answers[question.name]
                    if answer.confidence >= config.min_confidence:
                        labels[factor.factor] = answer.choice.split(" > ")
                else:
                    # Follow one branch at a time for taxonomies with many leaves.
                    node, path = factor.root, []
                    while node.children:
                        question = ChoiceQuestion(
                            name=f"select.taxonomy_{factor.factor}_{len(path)}",
                            instructions=f"Choose the child under {' / '.join(path) or factor.factor} which best describes the example.",
                            choices={child.name: None for child in node.children},
                        )
                        result = client.decide(record_state(record), [question])
                        used.append(result)
                        answers.update(result.answers)
                        answer = result.answers[question.name]
                        if answer.confidence < config.min_confidence:
                            path = []
                            break
                        node = next(
                            child
                            for child in node.children
                            if child.name == answer.choice
                        )
                        path.append(node.name)
                    if path:
                        labels[factor.factor] = path
            updated = attach_answers(record, answers)
            dims = {
                **updated.scores.quality_per_dim,
                "taxonomy_nodes": labels,
                "taxonomy_labeled_by": "decision",
            }
            out.append(
                updated.model_copy(
                    update={
                        "scores": updated.scores.model_copy(
                            update={"quality_per_dim": dims}
                        )
                    }
                )
            )
    return out, used
