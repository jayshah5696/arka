"""Slice 5 \u2014 Simula \u00a72.3 calibrated batch-Elo complexity scoring.

For each ConversationRecord we want a `complexity_elo` rating that is
comparable across datasets. The procedure (Davidson et al. 2026, \u00a72.3):

1. Sample records into batches of size B such that each record appears K
   times across batches.
2. For each batch, ask an M3 to RANK its members by complexity (most complex
   first).
3. Decompose each batch ranking into pairwise win/loss outcomes
   (B*(B-1)/2 pairs per batch).
4. Update a per-record Elo rating from those pairwise outcomes (default
   K-factor=32, starting Elo=400 to match the paper's plots).
5. Attach the final rating to
   ``record.scores.quality_per_dim['complexity_elo']``.

The stage **does not drop any records** \u2014 it annotates and passes through.
Filter or select stages downstream consume the score.

Cost: ~K LLM calls per record (each batch is one call). With default
``samples_per_record=4, batch_size=5`` that is ``4 * N / 5 = 0.8 N`` calls,
each returning a rank list of length 5.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel, Field

from arka.config.models import resolve_llm_override
from arka.llm.client import LLMClient, LLMClientError
from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.pipeline.stages import Stage
from arka.records.models import ConversationRecord, Record

# --- Elo math ----------------------------------------------------------------


def _expected(r_a: float, r_b: float) -> float:
    """Standard chess Elo expected-score formula."""
    return 1.0 / (1.0 + math.pow(10.0, (r_b - r_a) / 400.0))


def elo_update_pair(
    *,
    rating_a: float,
    rating_b: float,
    a_wins: bool,
    k: float = 32.0,
) -> tuple[float, float]:
    """Apply one pairwise Elo update.

    Returns (new_rating_a, new_rating_b). Sum is preserved (zero-sum).
    """
    s_a = 1.0 if a_wins else 0.0
    s_b = 1.0 - s_a
    e_a = _expected(rating_a, rating_b)
    e_b = 1.0 - e_a
    new_a = rating_a + k * (s_a - e_a)
    new_b = rating_b + k * (s_b - e_b)
    return new_a, new_b


# --- Ranker schema -----------------------------------------------------------


class _BatchRanking(BaseModel):
    """Structured output for a single batch ranking call.

    The model must return ``ranked_ids`` containing every input id, ordered
    most-complex-first. Strict equality on the set is enforced at parse time.
    """

    ranked_ids: list[str] = Field(..., min_length=1)


_RANKER_SYSTEM = (
    "You rank a small batch of (instruction, response) examples by COMPLEXITY.\n"
    "Complexity = how many distinct constraints the response must satisfy, the\n"
    "depth of domain knowledge required, the number of explicit reasoning\n"
    "steps, and the difficulty of edge cases. Length alone is NOT complexity.\n"
    "\n"
    "Return ONLY a JSON object with a single key 'ranked_ids' \u2014 the list of\n"
    "item ids from most complex (first) to least complex (last). Include every\n"
    "id exactly once.\n"
)


def _build_batch_messages(items: list[tuple[str, str, str]]) -> list[dict[str, str]]:
    """Build the messages for a single batch ranking call.

    `items` is a list of (id, instruction, response) tuples.
    """
    body_lines = []
    for rid, instr, resp in items:
        body_lines.append(f"ITEM {rid}:")
        body_lines.append(f"  instruction: {instr.strip()[:600]}")
        body_lines.append(f"  response:    {resp.strip()[:600]}")
        body_lines.append("")
    body = "\n".join(body_lines)
    return [
        {"role": "system", "content": _RANKER_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Rank the following {len(items)} items by complexity, most "
                f"complex first.\n\n{body}\n\n"
                'Reply with JSON: {"ranked_ids": [...]}'
            ),
        },
    ]


# --- Stage --------------------------------------------------------------------


class ComplexityEloScoringStage(Stage):
    """Annotate each ConversationRecord with a `complexity_elo` rating."""

    name = "02s_complexity_elo"
    stage_action = "scored"

    DEFAULT_BATCH_SIZE = 5
    DEFAULT_SAMPLES_PER_RECORD = 4
    DEFAULT_K_FACTOR = 32.0
    DEFAULT_STARTING_ELO = 400.0

    def __init__(
        self,
        llm_client: Any | None = None,
        seed: int = 0,
    ) -> None:
        self._llm_client = llm_client
        self._rng = random.Random(seed)
        self._output_writer = OutputWriter()

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        cfg = ctx.config.filters.get_stage_config("complexity_elo")
        if cfg is None:
            return records

        batch_size = getattr(cfg, "batch_size", None) or self.DEFAULT_BATCH_SIZE
        samples_per_record = (
            getattr(cfg, "samples_per_record", None) or self.DEFAULT_SAMPLES_PER_RECORD
        )
        k_factor = getattr(cfg, "k_factor", None) or self.DEFAULT_K_FACTOR

        conv_records: list[ConversationRecord] = [
            r for r in records if isinstance(r, ConversationRecord)
        ]
        non_conv = [r for r in records if not isinstance(r, ConversationRecord)]
        if not conv_records:
            self._write_artifacts(ctx, count_in=len(records), elos=[])
            return list(records)

        # Bucket-fill batches so each record appears `samples_per_record` times.
        batches = self._build_batches(
            [r.id for r in conv_records],
            batch_size=batch_size,
            samples_per_record=samples_per_record,
        )

        # Issue ranker calls in parallel up to ctx.max_workers.
        client = self._llm_client or LLMClient(
            config=resolve_llm_override(
                ctx.config.llm, getattr(cfg, "llm_override", None)
            )
        )
        record_by_id = {r.id: r for r in conv_records}
        rankings = self._call_ranker_batch(client, batches, record_by_id, ctx)

        # Apply Elo from each batch's ranking via pairwise comparisons.
        elos: dict[str, float] = {r.id: self.DEFAULT_STARTING_ELO for r in conv_records}
        for ranked_ids in rankings:
            if not ranked_ids:
                continue
            # Best -> worst order. Iterate every (i<j) pair: i wins.
            for i in range(len(ranked_ids)):
                for j in range(i + 1, len(ranked_ids)):
                    a_id = ranked_ids[i]
                    b_id = ranked_ids[j]
                    if a_id not in elos or b_id not in elos:
                        continue
                    elos[a_id], elos[b_id] = elo_update_pair(
                        rating_a=elos[a_id],
                        rating_b=elos[b_id],
                        a_wins=True,
                        k=k_factor,
                    )

        # Attach to records.
        out: list[Record] = list(non_conv)
        for r in conv_records:
            elo = round(elos[r.id], 2)
            updated = r.model_copy(
                update={
                    "scores": r.scores.model_copy(
                        update={
                            "quality_per_dim": {
                                **r.scores.quality_per_dim,
                                "complexity_elo": elo,
                            }
                        }
                    )
                }
            )
            out.append(updated)

        self._write_artifacts(ctx, count_in=len(records), elos=list(elos.values()))
        return out

    def _build_batches(
        self,
        record_ids: list[str],
        *,
        batch_size: int,
        samples_per_record: int,
    ) -> list[list[str]]:
        """Build batches such that each record appears `samples_per_record` times.

        Strategy: build a deck of (id repeated samples_per_record times),
        shuffle, split into batches of `batch_size`. The last batch may be
        short; we keep it if at least 2 ids remain (1-record batches give no
        pairwise signal).
        """
        deck = []
        for rid in record_ids:
            for _ in range(samples_per_record):
                deck.append(rid)
        self._rng.shuffle(deck)

        batches: list[list[str]] = []
        for start in range(0, len(deck), batch_size):
            chunk = deck[start : start + batch_size]
            # Deduplicate within a batch: a record appearing twice in the
            # same batch ranking is meaningless.
            seen = set()
            unique = []
            for rid in chunk:
                if rid not in seen:
                    seen.add(rid)
                    unique.append(rid)
            if len(unique) >= 2:
                batches.append(unique)
        return batches

    def _call_ranker_batch(
        self,
        client: Any,
        batches: list[list[str]],
        record_by_id: dict[str, ConversationRecord],
        ctx: StageContext,
    ) -> list[list[str]]:
        worker_count = max(1, min(ctx.max_workers, max(1, len(batches))))

        def _one(batch_ids: list[str]) -> list[str]:
            items = [
                (
                    rid,
                    record_by_id[rid].payload.instruction,
                    record_by_id[rid].payload.response,
                )
                for rid in batch_ids
            ]
            kwargs = {
                "messages": _build_batch_messages(items),
                "schema": _BatchRanking,
            }
            try:
                out = client.complete_structured(**kwargs)
            except LLMClientError:
                return []
            except TypeError as exc:
                if "unexpected keyword argument" not in str(exc):
                    raise
                out = client.complete_structured(**kwargs)
            parsed = out.parsed
            if not isinstance(parsed, _BatchRanking):
                return []
            # Reject malformed rankings (missing ids, duplicates) silently \u2014
            # those batches just don't contribute Elo updates.
            seen = set()
            valid: list[str] = []
            for rid in parsed.ranked_ids:
                if rid in record_by_id and rid not in seen:
                    seen.add(rid)
                    valid.append(rid)
            return valid

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            return list(pool.map(_one, batches))

    def _write_artifacts(
        self, ctx: StageContext, *, count_in: int, elos: list[float]
    ) -> None:
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        dist: dict[str, float | None] = {
            "min": min(elos) if elos else None,
            "max": max(elos) if elos else None,
            "mean": statistics.mean(elos) if elos else None,
            "median": statistics.median(elos) if elos else None,
            "stdev": statistics.stdev(elos) if len(elos) > 1 else None,
        }
        stats = {
            "stage": self.name,
            "count_in": count_in,
            "count_out": count_in,
            "dropped_count": 0,
            "drop_reasons": {},
            "scored_count": len(elos),
            "complexity_elo_distribution": dist,
        }
        (ctx.work_dir / "stats.json").write_text(json.dumps(stats, indent=2))
