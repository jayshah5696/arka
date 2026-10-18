"""Shared generator execution engine and item-checkpointing core."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from arka.common.models import StrictModel
from arka.decisions.models import DecisionResult
from arka.llm.client import extract_json_text
from arka.llm.models import LLMOutput, TokenUsage
from arka.pipeline.checkpoint import CheckpointManager
from arka.records.models import Record

if TYPE_CHECKING:
    from arka.pipeline.models import StageContext

__all__ = [
    "RawGeneratorResponse",
    "apply_post_generation_decisions",
    "execute_generator_slots",
    "existing_rows_for_prompt",
    "extract_json_text",
    "load_raw_responses",
    "resolve_generator_llm_client",
]


class RawGeneratorResponse(StrictModel):
    """Raw response logged per generator plan item for resume and auditing."""

    plan_index: int
    seed_id: str
    generated_text: str
    prompt_hash: str
    model: str
    latency_ms: int
    usage: TokenUsage


def resolve_generator_llm_client(
    ctx: StageContext,
    stage_config: Any,
    default_client: Any | None = None,
) -> Any:
    """Resolve the LLM client, respecting StageLLMOverride if configured."""
    if default_client is not None:
        return default_client
    override = getattr(stage_config, "llm_override", None)
    return ctx.llm_client(override=override)


def load_raw_responses(responses_path: Path) -> list[RawGeneratorResponse]:
    """Load existing raw generator responses from JSONL."""
    if not responses_path.exists():
        return []
    rows: list[RawGeneratorResponse] = []
    with responses_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(RawGeneratorResponse.model_validate_json(stripped))
            except Exception:
                return []
    return rows


def existing_rows_for_prompt(
    responses_path: Path, prompt_hash: str
) -> list[RawGeneratorResponse]:
    """Return existing raw responses if their prompt hash matches."""
    if not responses_path.exists():
        return []
    rows = load_raw_responses(responses_path)
    if not rows:
        return []
    if all(row.prompt_hash == prompt_hash for row in rows):
        return rows
    return []


def execute_generator_slots(
    *,
    items: list[Any],
    execute_slot_fn: Callable[
        [Any], tuple[str, LLMOutput] | tuple[str, str, LLMOutput] | LLMOutput
    ],
    responses_path: Path,
    ctx: StageContext,
    stage_name: str,
    prompt_hash: str,
    get_item_id: Callable[[Any], str],
    checkpoint_manager: CheckpointManager | None = None,
) -> list[RawGeneratorResponse]:
    """Execute generator slots in parallel with per-item checkpointing and resume support."""
    checkpoint = checkpoint_manager or ctx.checkpoint_manager
    existing_rows = existing_rows_for_prompt(responses_path, prompt_hash)
    if not existing_rows and responses_path.exists():
        responses_path.unlink()

    done_indices = {row.plan_index for row in existing_rows}
    indexed_items = [
        (getattr(item, "plan_index", idx), item) for idx, item in enumerate(items)
    ]
    pending = [(idx, item) for idx, item in indexed_items if idx not in done_indices]

    if checkpoint is not None:
        checkpoint.save_generator(
            run_id=ctx.run_id,
            stage_name=stage_name,
            prompt_hash=prompt_hash,
            responses_path=responses_path,
            response_count=len(existing_rows),
            status="running",
        )

    if pending:
        responses_path.parent.mkdir(parents=True, exist_ok=True)
        written_count = len(existing_rows)

        def _run_slot(entry: tuple[int, Any]) -> tuple[int, Any, Any]:
            idx, item = entry
            result = execute_slot_fn(item)
            return idx, item, result

        evaluated = ctx.map_parallel(pending, _run_slot)
        # Preserve original item ordering when appending to JSONL
        evaluated.sort(key=lambda x: x[0])

        with responses_path.open("a", encoding="utf-8") as handle:
            for idx, item, result in evaluated:
                if isinstance(result, tuple):
                    if len(result) == 2:
                        text, output = result
                    else:
                        _, text, output = result
                else:
                    output = result
                    if getattr(output, "text", None) is not None:
                        text = output.text.strip()
                    elif getattr(output, "parsed", None) is not None:
                        text = output.parsed.model_dump_json()
                    else:
                        raise ValueError(
                            "Generator output returned neither text nor parsed"
                        )

                row = RawGeneratorResponse(
                    plan_index=idx,
                    seed_id=get_item_id(item),
                    generated_text=text,
                    prompt_hash=prompt_hash,
                    model=getattr(
                        output,
                        "model",
                        ctx.config.llm.model if ctx.config else "unknown",
                    ),
                    latency_ms=getattr(output, "latency_ms", 0),
                    usage=getattr(output, "usage", TokenUsage()),
                )
                handle.write(
                    json.dumps(row.model_dump(mode="json"), separators=(",", ":"))
                    + "\n"
                )
                handle.flush()
                written_count += 1
                if checkpoint is not None:
                    checkpoint.save_generator(
                        run_id=ctx.run_id,
                        stage_name=stage_name,
                        prompt_hash=prompt_hash,
                        responses_path=responses_path,
                        response_count=written_count,
                        status="running",
                    )

    all_raw = load_raw_responses(responses_path)
    if checkpoint is not None:
        checkpoint.save_generator(
            run_id=ctx.run_id,
            stage_name=stage_name,
            prompt_hash=prompt_hash,
            responses_path=responses_path,
            response_count=len(all_raw),
            status="completed",
        )
    return all_raw


def apply_post_generation_decisions(
    *,
    records: list[Record],
    ctx: StageContext,
    stage_config: Any,
    stage_name: str,
    kind: str = "prompt_based_generator",
) -> tuple[list[Record], list[Record], list[DecisionResult]]:
    """Run post-generation fidelity_check and best_of_n hooks if enabled."""
    kept = list(records)
    decision_dropped: list[Record] = []
    decision_results: list[DecisionResult] = []

    fidelity_check = getattr(stage_config, "fidelity_check", None)
    if fidelity_check is not None and getattr(fidelity_check, "enabled", False):
        from arka.pipeline.decision_checks import taxonomy_fidelity

        kept, dropped, results = taxonomy_fidelity(
            kept, ctx, fidelity_check, stage_name
        )
        decision_dropped.extend(dropped)
        decision_results.extend(results)

    best_of_n = getattr(stage_config, "best_of_n", None)
    if best_of_n is not None and getattr(best_of_n, "enabled", False):
        from arka.pipeline.decision_checks import best_of_n as run_best_of_n

        kept, dropped, results = run_best_of_n(
            kept, ctx, best_of_n, stage_name, kind=kind
        )
        decision_dropped.extend(dropped)
        decision_results.extend(results)

    return kept, decision_dropped, decision_results
