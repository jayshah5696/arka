# Simula Paper Review — Reasoning-Driven Synthetic Data Generation

**Paper:** Davidson et al., *Simula: Reasoning-Driven Synthetic Data Generation and Evaluation*, TMLR 03/2026.
**URL:** https://openreview.net/pdf?id=NALsdGEPhB
**Asked:** Is it in arka? Does it expand rubrics? Critique + next steps.

## Is it already in arka?

No. Grep'd `src/`, `docs/`, `rubrics/` for `simula | davidson | NALsdGEPhB | reasoning-first | taxonom(ic|y)` — zero relevant hits. One tangential text-rewriting-taxonomy research log exists but is unrelated to Simula.

## Does it "expand rubrics"?

Not exactly. Simula is a **synthetic-data-generation mechanism** paper, not a rubric paper. Rubrics (what `rubrics/sft_quality.yaml` is) score existing data on fixed human-authored dimensions. Simula's taxonomies are **generative scaffolds** — hierarchical factor/attribute trees used to construct prompts *and then* reused post-hoc as a coverage metric. Complementary to rubrics, not an extension.

## What Simula actually does

Three-stage pipeline:

1. **Taxonomy construction** — LLM (M3) proposes factors of variation for a dataset description `y`, then expands each factor breadth-first into a taxonomy tree using Best-of-N generation + a critic (add/merge/prune nodes) + optional per-level "granularity plan".
2. **Agentic generation** — sample nodes via compatibility-aware strategies → meta-prompt → optional "complexify" (fraction `c`) → generate → pointwise critic → **double-critic** (independent "is this correct?" + "is this incorrect?" passes) to mitigate sycophancy.
3. **Evaluation** — two new intrinsic metrics:
   - **Level-Ratio taxonomic coverage** (proportion of unique nodes hit at each level).
   - **Calibrated batch-wise Elo complexity scoring** (repeat items in mixed batches, pairwise reduce, Elo).

Experiments: Gemini 2.5 Flash teacher → Gemma-3-4B LoRA student on CTI-MCQ, CTI-RCM, LEXam, GSM8k, Global-MMLU, up to 512k samples.

**Headline findings:**
- Full Simula > Baseline at every scale on every dataset.
- Local (meta-prompt + complexify) and Global (deep taxonomy) diversification are **additive**.
- Double-critic gives a measurable empirical lift, cost grows with complexity.
- Complexity helps with a strong teacher (GSM8k) and *hurts* with a weak one (LEXam).
- Embedding cosine distance is a coarse diversity signal; taxonomic coverage is strictly more informative.

## Critique

**Strong**
- Taxonomy-as-coverage-scaffold is cleaner than Evol-Instruct / Self-Instruct and gives a real audit trail.
- Double-critic is a cheap, defensible anti-sycophancy trick. Easy to port.
- Elo batch complexity scoring survives the "complexity is relative" objection.
- "Properties matter more than size" is the most durable takeaway and well-supported.

**Weak / open**
- Single teacher family (Gemini 2.5 Flash) generates, critiques, *and* scores complexity. No clean separation of "Simula works" from "Flash reasons well."
- Taxonomy eval is LLM-judged on 6 topics, no IAA.
- ~5× inference cost vs baseline. Hand-waved as "training dominates."
- Coverage-metric circularity: they score real data's coverage against their own generated taxonomy.
- Complexity-hurts-with-weak-teacher is charitable framing; pragmatic reading is that complexification multiplies confident errors.
- Critic rejection rate swings 2–61%. No adaptive budget.
- **No head-to-head against Evol-Instruct / Magpie / Self-Instruct.** Only within-Simula ablations.

## Fit with arka

arka has: prompt + Evol-Instruct generators, single-judge rubric labeler, IFD scorer, MinHash/LSH dedup, canary/lang/length filters.

arka lacks: global coverage planning, taxonomy-driven generation, double-critic, reasoning-based complexity scoring.

| Simula idea | arka integration point | Effort |
|---|---|---|
| Double-critic | New `double_critic` filter stage next to `label_filter` | **Low** (~1 day) |
| Taxonomy-driven generator | New `taxonomy_generator` + YAML taxonomy schema | **Medium** (3–5 days) |
| Level-Ratio taxonomic coverage | Report stage; per-record taxonomy assignment → `stats.json` | **Low–Medium** |
| Batch-Elo complexity scoring | New stage in `scoring_stages.py`, attaches to `Record.scores` | **Medium** (2–3 days) |
| Complexification | New operator in `evol_instruct.py` registry | **Trivial** |

## Recommended next steps

1. **ADR** `docs/decisions/0002-taxonomy-and-double-critic.md` — cite Simula, adopt double-critic + coverage metric, keep Evol-Instruct, make taxonomy-gen opt-in.
2. **Ship double-critic first** (highest leverage, lowest risk, ~1 day, red/green TDD).
3. **Add complexification as a new Evol operator** (trivial, slots into existing registry).
4. **Prototype taxonomy-driven generator** behind a feature flag; start with user-authored YAML taxonomies before building the Best-of-N expansion loop.
5. **Add taxonomic coverage as a report metric** once the taxonomy schema lands.
6. **Defer Elo complexity scoring** — IFD already gives us per-sample difficulty at v0 scale.

**Do not** swallow "Simula > Evol-Instruct." The paper does not show that. Keep Simula-style generation optional; defaults stay cheap.
