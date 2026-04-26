# Simula Slice 3 — Taxonomy-Driven Generator

**Date:** 2026-04-25
**Branch:** `wt/simula`
**Slice:** 3 — Simula §2.1–2.2 user-authored YAML taxonomy + seedless generator.

## What landed

A complete taxonomy schema, loader, and generator stage, plus a real
production-grade taxonomy YAML for Jay's writing-assistant domain.

### New module: `src/arka/taxonomy/`

```
taxonomy/
├── __init__.py
└── models.py
```

Pydantic models:
- `TaxonomyNode` — recursive tree node; sibling-name uniqueness enforced.
- `Taxonomy` — one factor with a rooted tree; `.leaf_paths()` returns every
  root-to-leaf chain (the sample-able units).
- `SamplingStrategy` — names a subset of factors that should be sampled
  together (Simula §2.2 "compatible groups").
- `TaxonomyBundle` — full bundle; loads from YAML; auto-synthesises a default
  `[all factors]` strategy if the user omits the strategies section; rejects
  strategies that reference unknown factors at load time.

YAML accepts either `factor:` or `name:` for the factor key (alias) so the
file is comfortable to author.

### New stage: `src/arka/pipeline/taxonomy_generator.py`

`TaxonomyGeneratorStage` (registered as `generator.type: taxonomy_prompt`):
1. Loads the bundle.
2. Distributes the target_count across strategies (uniform, deterministic
   remainder).
3. For each generation slot, samples one leaf-path per included factor.
4. Builds a meta-prompt that lists the sampled attributes as hard
   requirements + the same anti-AI-slop constraints we've been using.
5. Calls `LLMClient.complete_structured` for one structured ConversationRecord.
6. Records the sampled node-set on
   `record.scores.quality_per_dim["taxonomy_nodes"]` so slice 4 (level-ratio
   coverage) can read it directly.

The stage is **seedless** by design — Simula's taxonomy approach does not need
seeds. Existing seed records flow through unchanged so dedup can still see them.

### New config field

`generator.taxonomy_path: str | None`. Required when
`generator.type == "taxonomy_prompt"`; ignored otherwise. Resolved relative
to project_root, mirroring how `rubric_path` is resolved by the labeling
filter.

### New artifacts

- `tools/simula_eval/taxonomies/personal_writing.yaml` — 4 factors (domain,
  intent, tone, constraint), 6 domain branches × 7 intents × 6 tones × 6
  constraints = ~1500 unique node combinations across two strategies
  (`default` covers all four; `short_form` skips the constraint factor for
  short-form writing).
- `tools/simula_eval/configs/03-taxonomy.yaml` — slice config.
- `tests/unit/test_taxonomy_models.py` — 7 schema tests.
- `tests/unit/test_taxonomy_generator.py` — 6 generator tests.

Full suite: **288 passing** (was 275, +13). Lint clean.

## Live run on humanize-rl seeds

The taxonomy generator does not consume seeds; it samples node-sets. We still
load the 50 seeds so the dedup stage can filter against them.

```
| stage             | slice 0 (baseline) | slice 3 (taxonomy)         |
|-------------------|--------------------|----------------------------|
| 02_generate       | 50 → 300 (prompt)  | 50 → 200 (taxonomy)        |
| 02d_near_dedup    | 300 → 217 [83]     | 200 → 200 [0]              |
| FINAL             | 217                | 199                        |
```

| metric | 00-baseline | 03-taxonomy | delta |
|---|---:|---:|---:|
| final_count | 217 | 199 | -8% |
| **avg pairwise cosine distance** | **0.2480** | **0.4124** | **+66%** ↑ |
| **near-dup drops** | 83 (27.7%) | **0** (0%) | **-100%** ↓ |
| text_chars_median | 396 | 470 | +19% |
| text_chars_max | 626 | 736 | +18% |
| wallclock | 5m | 3.5m | -30% |

This is the **strongest result of any slice so far**. Higher diversity than
slice 2 (complexify) AND output length stays in the seed-style range.

## Sample outputs

The taxonomy nodes show through cleanly. A few examples I sampled from the
output dataset:

```
sampled: { domain: technical_writing/rfc, intent: explain, tone: dryly_funny }
[user] Explain the purpose of an RFC in technical documentation.
[assistant] An RFC is a way to propose standards while pretending to be democratic.
            It is mostly a paper trail for future blame.

sampled: { domain: email/personal_brief, intent: explain, tone: casual,
           constraint: must_acknowledge_a_tradeoff }
[user] Write a brief email outlining the plan for my upcoming week and mention a
       potential compromise.
[assistant] ...I know this schedule means we have to pause work on the new marketing
            site for a few days, but getting the draft done is our top priority...

sampled: { domain: essay/reflective, intent: outline, tone: matter_of_fact,
           constraint: must_acknowledge_a_tradeoff }
[user] Outline a reflective essay about the decision to quit a stable job to
       pursue creative work.
[assistant] ...I must acknowledge the tradeoff between financial security and
            personal fulfillment...
```

Notice how the `must_acknowledge_a_tradeoff` constraint shows through even
when the domain and intent vary. This is the kind of attribute control Simula
promises and the prompt-based generator could not give us.

## Honest read

**What worked**
- Diversity jumped 66%. **Zero** near-duplicates — taxonomy sampling pushed
  generation across a much wider conceptual surface than seed-driven random
  variation.
- Length stayed close to seed style (470 chars median vs Jay's 396) — slice 3
  matches his actual seed character, unlike slice 2's RFC-length outputs.
- Wallclock was **fastest of any slice** (3.5m) because there's no
  generator-2-call back-and-forth (just one structured-output call per record),
  no critic, and the existing prompt-based stage's parallelism applies.
  (Wait — actually the taxonomy generator IS still serial inside the stage's
  for-loop. The 3.5m wallclock for 150 calls suggests gemini-3.1-flash-lite is
  much faster than gemini-3-flash-preview from slice 1.5. Same parallelism gap
  as evol-instruct. Tracked as followup.)
- The test suite proved out the cleanest TDD cycle of any slice: 13 tests,
  all green on the second pass after one trivial Pydantic alias addition.

**Honest limits**
- The taxonomy is user-authored. Simula's headline contribution is the
  **M3-driven Best-of-N taxonomy expansion** — the model proposes a
  taxonomy from a description like "stories about cats" via a generate-critic
  loop. That's slice 3.5 / a distinct next step. The user-authored path we
  shipped is a strict subset of Simula's design but it is the part that is
  most useful for arka's "config-driven SFT pipeline" framing.
- `generation_multiplier` is honored for the budget but the generator still
  samples-with-replacement, so for very large `target_count` the same
  node-set could be selected twice. Slice 0 baseline used `multiplier=2` for
  the same reason and relied on dedup. Acceptable in practice; documented
  inline.
- Per-strategy weighting is uniform (Simula supports weighted strategies). In
  practice for slice 3 we have only `[default]` active in the run; the
  `short_form` strategy is defined in the YAML but never used in the
  comparison. The first interesting use of multi-strategy is when we want a
  controlled mix (e.g. 70% short-form / 30% with-constraint). Tracked.
- The taxonomy YAML I authored is hand-tuned for Jay's domain. Building a
  reusable starter taxonomy for "instruction following" / "math" / "code"
  would lower the barrier to adopting this in other arka pipelines. Out of
  scope here.

## Followups

1. **M3-driven taxonomy expansion** (slice 3.5) — the actual paper headline.
   Implement Simula's Best-of-N + critic loop in
   `arka.taxonomy.expand.expand_factor(...)`, gated by a CLI command
   `arka taxonomy expand --description "..." --depth 3 --out tax.yaml`.
2. **Per-strategy weights** — the YAML schema can grow a `weight: float`
   field on `SamplingStrategy` without breaking back-compat.
3. **Parallelize the generator loop** — same fix as evol-instruct; bound by
   `executor.max_workers`, mirror the `double_critic` ThreadPoolExecutor.
4. **Starter taxonomies under `arka.taxonomy.starters/`** — instruction
   following, math, code, RAG QA. Any user can drop one in to start.

## Reproduce

```bash
cd /Users/jshah/Documents/GitHub/.arka-worktrees/simula
export OPENROUTER_API_KEY=sk-...

uv run arka --config tools/simula_eval/configs/03-taxonomy.yaml \
            --run-id slice-03-taxonomy

uv run python tools/simula_eval/metrics.py \
  tools/simula_eval/configs/runs/slice-03-taxonomy \
  scratch/simula-eval/03-taxonomy/dataset.jsonl --name 03-taxonomy

uv run python tools/simula_eval/compare.py \
  scratch/simula-eval/00-baseline/metrics.json \
  scratch/simula-eval/03-taxonomy/metrics.json \
  --out scratch/simula-eval/03-taxonomy/comparison.md
```
