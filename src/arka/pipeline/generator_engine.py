"""Shared generator execution engine and item-checkpointing core."""

from __future__ import annotations

import json
import threading
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
    latency_ms: int = 0
    usage: TokenUsage = TokenUsage()


def resolve_generator_llm_client(
    ctx: StageContext,
    stage_config: Any,
    default_client: Any | None = None,
) -> Any:
    """Resolve the LLM client, respecting StageLLMOverride if configured.

    Generator stages checkpoint completed slots in raw_responses.jsonl, so
    prompt-addressed SQLite cache reads are disabled to preserve independent
    samples when multiple slots share a seed prompt.
    """
    if default_client is not None:
        return default_client
    override = getattr(stage_config, "llm_override", None)
    client = ctx.llm_client(override=override)
    if hasattr(client, "read_cache"):
        client.read_cache = False
    return client


def load_raw_responses(responses_path: Path) -> list[RawGeneratorResponse]:
    """Load existing raw generator responses from JSONL.

    Preserves valid rows before an incomplete final line (such as an interrupted
    append) and rewrites the file without the trailing fragment. Raises
    ValueError if corruption occurs elsewhere.
    """
    if not responses_path.exists():
        return []
    raw_text = responses_path.read_text(encoding="utf-8")
    non_empty = [
        (line_no, raw_line)
        for line_no, raw_line in enumerate(raw_text.splitlines(keepends=True), start=1)
        if raw_line.strip()
    ]
    rows: list[RawGeneratorResponse] = []
    truncated_final = False
    for idx, (line_no, raw_line) in enumerate(non_empty):
        stripped = raw_line.strip()
        try:
            rows.append(RawGeneratorResponse.model_validate_json(stripped))
        except Exception as exc:
            is_last = idx == len(non_empty) - 1
            is_incomplete = not raw_line.endswith(("\n", "\r")) or (
                stripped.startswith("{") and not stripped.endswith("}")
            )
            if is_last and is_incomplete:
                truncated_final = True
                break
            raise ValueError(
                f"Corrupted generator checkpoint in {responses_path} at line {line_no}"
            ) from exc

    if truncated_final:
        cleaned = "".join(
            json.dumps(row.model_dump(mode="json"), separators=(",", ":")) + "\n"
            for row in rows
        )
        responses_path.write_text(cleaned, encoding="utf-8")
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
    responses_path.unlink(missing_ok=True)
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
        write_lock = threading.Lock()

        def _to_row(idx: int, item: Any, result: Any) -> RawGeneratorResponse:
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

            return RawGeneratorResponse(
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

        def _run_slot(entry: tuple[int, Any]) -> RawGeneratorResponse:
            nonlocal written_count
            idx, item = entry
            result = execute_slot_fn(item)
            row = _to_row(idx, item, result)
            with write_lock:
                with responses_path.open("a", encoding="utf-8") as handle:
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
            return row

        ctx.map_parallel(pending, _run_slot)

    all_raw = sorted(load_raw_responses(responses_path), key=lambda row: row.plan_index)
    if pending and responses_path.exists():
        responses_path.write_text(
            "".join(
                json.dumps(row.model_dump(mode="json"), separators=(",", ":")) + "\n"
                for row in all_raw
            ),
            encoding="utf-8",
        )
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
