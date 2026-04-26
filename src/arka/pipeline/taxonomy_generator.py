"""Slice 3 \u2014 Simula taxonomy-driven generator.

For each generation slot we (a) pick a sampling strategy, (b) sample one
leaf-path per included factor, (c) build a meta-prompt that names the sampled
nodes alongside the user instructions, and (d) call the LLM for one structured
ConversationRecord. The sampled node-set is recorded on
``record.scores.quality_per_dim['taxonomy_nodes']`` so slice 4 (level-ratio
coverage) can compute the coverage metric directly from the dataset.

This stage is deliberately seedless: it does not consume seed records (Simula
\u00a72.2 is a seedless approach). Any non-seed records that arrive flow through
unchanged.

What is NOT implemented in this slice (tracked for slice 3.5):
- M3-driven Best-of-N taxonomy expansion (\u00a72.1)
- Per-strategy weighting beyond uniform
- Compatibility groups beyond \"include these factors\"
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from arka.llm.client import LLMClient
from arka.pipeline.models import StageContext
from arka.pipeline.output import OutputWriter
from arka.pipeline.stages import Stage
from arka.records.models import (
    ConversationPayload,
    ConversationRecord,
    Record,
    RecordLineage,
    RecordScores,
    RecordSource,
)
from arka.taxonomy.models import SamplingStrategy, TaxonomyBundle

logger = logging.getLogger(__name__)


class _GeneratedConversation(BaseModel):
    instruction: str
    response: str


_META_PROMPT = (
    "You are generating one supervised fine-tuning example.\n"
    "Use the SAMPLED ATTRIBUTES below as hard requirements. Each attribute names\n"
    "one concrete value the example must reflect; do not invent extra attributes.\n"
    "\n"
    "SAMPLED ATTRIBUTES:\n"
    "{attributes}\n"
    "\n"
    "Hard constraints:\n"
    "- The response must sound like a real person wrote it. No corporate fluff.\n"
    "- Avoid em dashes. Avoid magic adverbs ('simply', 'actually', 'really').\n"
    "- Keep the response between 60 and 600 characters.\n"
    "\n"
    'Return ONLY valid JSON with two keys: "instruction" and "response".\n'
)


def _format_attributes(sampled: dict[str, list[str]]) -> str:
    """Render sampled node-paths as a bulleted list for the meta-prompt.

    Example:
      domain: email
      tone: professional / brisk
    """
    lines = []
    for factor, path in sampled.items():
        chain = " / ".join(path) if path else "(root)"
        lines.append(f"  - {factor}: {chain}")
    return "\n".join(lines)


class TaxonomyGeneratorStage(Stage):
    name = "02_generate"
    stage_action = "generated"

    def __init__(
        self,
        llm_client: Any | None = None,
        seed: int = 0,
        project_root: Path | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._rng = random.Random(seed)
        self._output_writer = OutputWriter()
        self._project_root = project_root

    def run(self, records: list[Record], ctx: StageContext) -> list[Record]:
        gen_cfg = ctx.config.generator
        target = gen_cfg.target_count * gen_cfg.generation_multiplier
        if target <= 0:
            return list(records)
        if not gen_cfg.taxonomy_path:
            raise ValueError("TaxonomyGeneratorStage requires generator.taxonomy_path")

        # Resolve relative to project_root (set via StageBuilder), matching how
        # rubric_path is resolved by LabelingQualityFilterStage.
        raw_path = Path(gen_cfg.taxonomy_path)
        if raw_path.is_absolute() or self._project_root is None:
            tax_path = raw_path
        else:
            tax_path = self._project_root / raw_path
        bundle = TaxonomyBundle.from_yaml(tax_path)
        client = self._llm_client or LLMClient(config=ctx.config.llm)

        # Allocate per-strategy budgets uniformly. Simula \u00a72.2 supports
        # weighted strategies; that knob is deferred to a follow-up slice.
        strategies = bundle.strategies
        per_strategy = max(1, target // len(strategies))
        # Adjust to hit `target` exactly: distribute the remainder across the
        # first few strategies in order (deterministic for reproducibility).
        budgets = [per_strategy] * len(strategies)
        remaining = target - sum(budgets)
        for i in range(remaining):
            budgets[i % len(budgets)] += 1

        new_records: list[Record] = []
        for strategy, budget in zip(strategies, budgets, strict=True):
            for _ in range(budget):
                sampled = self._sample(bundle, strategy)
                rec = self._generate_one(client, sampled, ctx)
                if rec is not None:
                    new_records.append(rec)

        # Persist summary stats so the slice harness can read it like other stages.
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        stats = {
            "stage": self.name,
            "count_in": len(records),
            "count_out": len(records) + len(new_records),
            "dropped_count": 0,
            "drop_reasons": {},
            "generated_count": len(new_records),
            "taxonomy_path": gen_cfg.taxonomy_path,
            "strategies": [
                {"name": s.name, "budget": b}
                for s, b in zip(strategies, budgets, strict=True)
            ],
        }
        (ctx.work_dir / "stats.json").write_text(json.dumps(stats, indent=2))

        return list(records) + new_records

    def _sample(
        self, bundle: TaxonomyBundle, strategy: SamplingStrategy
    ) -> dict[str, list[str]]:
        sampled: dict[str, list[str]] = {}
        for factor_name in strategy.include_factors:
            tax = bundle.factor(factor_name)
            paths = tax.leaf_paths()
            if not paths:
                # Degenerate: a factor with only a root node yields an empty
                # path. Skip it rather than fail \u2014 the meta-prompt will simply
                # have one fewer attribute. Documented in the schema docstring.
                continue
            sampled[factor_name] = self._rng.choice(paths)
        return sampled

    def _generate_one(
        self,
        client: Any,
        sampled: dict[str, list[str]],
        ctx: StageContext,
    ) -> ConversationRecord | None:
        prompt = _META_PROMPT.format(attributes=_format_attributes(sampled))
        kwargs = {
            "messages": [{"role": "user", "content": prompt}],
            "schema": _GeneratedConversation,
            "temperature": ctx.config.generator.temperature,
            "max_tokens": ctx.config.generator.max_tokens,
        }
        try:
            output = client.complete_structured(**kwargs)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            kwargs.pop("temperature", None)
            kwargs.pop("max_tokens", None)
            output = client.complete_structured(**kwargs)

        parsed = output.parsed
        if not isinstance(parsed, _GeneratedConversation):
            logger.warning(
                "TaxonomyGeneratorStage: LLM did not return a parsed conversation; "
                "skipping sample %s",
                sampled,
            )
            return None

        # Stable id from the sampled attributes + prompt content. Not collision-
        # proof on its own; the downstream exact_dedup stage handles true dups.
        seed = json.dumps(sampled, sort_keys=True) + parsed.instruction
        rid = "tax-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]

        return ConversationRecord(
            id=rid,
            content_hash=hashlib.sha256(
                f"{parsed.instruction}\n{parsed.response}".encode()
            ).hexdigest(),
            source=RecordSource(type="taxonomy"),
            lineage=RecordLineage(
                root_id=rid,
                parent_ids=[],
                operator="taxonomy_prompt",
            ),
            payload=ConversationPayload(
                instruction=parsed.instruction,
                response=parsed.response,
            ),
            scores=RecordScores(
                quality_per_dim={"taxonomy_nodes": sampled},
            ),
            config_hash="taxonomy",  # hashing taxonomies is a slice-3.5 concern\n
            created_at=datetime.now(tz=UTC).isoformat(),
        )
