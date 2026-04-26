# ADR 0002: Simula Adoption — Reasoning-Driven Synthetic Data Generation

- Status: Accepted (proposed for merge via the `wt/simula` PR)
- Date: 2026-04-25
- Branch: `wt/simula` — 7 commits ahead of `main`
- Author: Jay Shah (with Pi as pair)
- Reference paper: Davidson et al., *Simula: Reasoning-Driven Synthetic Data Generation and Evaluation*, TMLR 03/2026 ([OpenReview](https://openreview.net/pdf?id=NALsdGEPhB))

## Context

Arka started as a config-driven SFT generation framework with prompt-based and
Evol-Instruct generators, MinHash dedup, length/language/IFD/judge filters, and
a single-judge LabelingEngine. The Simula paper proposes a more principled
"reasoning-first" pipeline whose three key ideas — taxonomy-driven coverage,
double-critic anti-sycophancy, and calibrated batch-Elo complexity — are
**orthogonal** to what arka already does and slot in cleanly as new stages.

This ADR records what we adopted, what we deliberately deferred, and why.

## Decision

Adopt five Simula primitives as additive components in arka, NOT as a
replacement for the existing generator/dedup/filter chain. Each primitive
shipped as one slice, red→green TDD, with a live measurement against the
slice-0 baseline.

## What we adopted

| # | Primitive | Where it lives | Status |
|---|---|---|---|
| 1 | **Double-critic filter** (§2.2) | `src/arka/pipeline/double_critic_stage.py` | shipped |
| 1.5 | Stage-level `llm_override` for the critic | wired through `resolve_llm_override(...)` | shipped |
| 2 | **Complexify operator** (§2.2 local complexity) | new entry in `src/arka/pipeline/evol_instruct.py` registry | shipped |
| 3 | **Taxonomy-driven generator** (§2.1–2.2) | new module `src/arka/taxonomy/` + `src/arka/pipeline/taxonomy_generator.py` | shipped (user-authored YAML; M3-driven expansion deferred) |
| 4 | **Level-Ratio Coverage metric** (§2.3) | `src/arka/taxonomy/coverage.py` + `tools/simula_eval/metrics.py` `--taxonomy` flag | shipped |
| 5 | **Calibrated Batch-Elo Complexity Scoring** (§2.3) | `src/arka/pipeline/complexity_elo_stage.py` | shipped |

Each is opt-in via YAML config — none changes the default behavior of an
existing pipeline.

## What we deliberately deferred

| # | Item | Why deferred |
|---|---|---|
| A | **M3-driven Best-of-N taxonomy expansion** (§2.1) | Slice 3 ships only the user-authored YAML path. The Best-of-N + critic loop that generates a taxonomy from a description is a meaningful extension worth its own slice (3.5). For arka's "config-driven" framing the user-authored path is the more useful starting point. |
| B | **M3-driven taxonomy assignment** for arbitrary records | Without this, slice 4's coverage metric only works on data that already carries `taxonomy_nodes`. Add as slice 4.5 to score baselines. |
| C | **Per-record critic checkpointing** | Existing arka resume is per-stage. With ~400 critic calls in slice 1.5, partial-stage resume would save real money. The `embeddings_cache` SQLite table shows the pattern; mirror it for `critic_cache` as a future slice. |
| D | **Cost telemetry** (`StageStat.cost_usd`) | Field exists; OpenRouter usage headers are reachable from `LLMOutput`. Easy follow-up. |
| E | **Parallelized evol-instruct + taxonomy generators** | Both are serial `for-record-for-branch` loops today. Wrap in `ThreadPoolExecutor` (mirror `double_critic`) to bring wallclock down. Pre-existing arka issue surfaced by this work; not Simula-specific. |
| F | **Simula `c` parameter** (complexification fraction) | Today, "complexify only X% of branches" requires brittle proxy configs. A first-class `complexification_fraction: float` field on `GeneratorConfig` is the right answer. |
| G | **Bootstrap CIs on Elo + different-family ranker + length-vs-Elo regression** | Slice 5 ships K=3 samples/record with a same-family ranker. Stronger statistical guarantees and bias detection are clear next steps. |

The deferred items are tracked in the per-slice logs and on the followup queue.
**None of them block merging the current PR.**

## Headline numbers (Jay's `humanize-rl` 50 seeds)

```
| metric                      | 00-base | 01-DC | 01b-strong | 02-cmplx | 03-tax | 05-elo |
|-----------------------------|--------:|------:|-----------:|---------:|-------:|-------:|
| final count                 |     217 |   211 |        197 |      198 |    199 |    197 |
| avg pairwise cosine dist    |   0.248 | 0.244 |      0.240 |    0.363 |  0.412 |  0.408 |
| text_chars_median           |     396 |   418 |        416 |     1297 |    470 |    459 |
| near-dup drop %             |    27.7 |  29.3 |       23.0 |      0.5 |    0.0 |    1.0 |
| critic rejection %          |     n/a |   0.5 |       14.7 |      n/a |    n/a |    n/a |
| taxonomy coverage L1 / L2   |     n/a |   n/a |        n/a |      n/a | 100% / 100% | 100% / 100% |
| complexity Elo spread (pts) |     n/a |   n/a |        n/a |      n/a |    n/a |    261 |
```

**Best per-axis result:**

- diversity → slice 3 (taxonomy)
- complexity stacking → slice 2 (complexify)
- anti-sycophancy → slice 1.5 (strong critic)
- coverage transparency → slice 4 (level-ratio)
- per-sample complexity ranking → slice 5 (Elo)

## Tests

- 299 passing in the unit suite (was 262 on `main`; +37 new).
- 90% coverage on changed code.
- Lint clean (ruff check + format).
- One pre-existing integration test still fails on `main` for unrelated YAML
  reasons; not in scope for this PR.

## Costs (rough OpenRouter spend across all slice runs)

- Slice 0: ~$0.05 (50 seeds × 6 generations on lite)
- Slice 1 + 1.5: ~$0.50 (most of it on the gemini-3-flash-preview critic)
- Slice 2: ~$0.20 (300 evol calls on lite)
- Slice 3: ~$0.10 (150 taxonomy generations on lite)
- Slice 4: $0 (read-only metric)
- Slice 5: ~$0.40 (~150 batch rankings × 4-item batches on flash-3-preview)

Total: ~$1.25 for the whole adoption work. Cheaper than I estimated.

## Files added by this PR (high-level)

```
src/arka/
├── pipeline/
│   ├── double_critic_stage.py        (slice 1)
│   ├── taxonomy_generator.py         (slice 3)
│   └── complexity_elo_stage.py       (slice 5)
└── taxonomy/
    ├── __init__.py
    ├── models.py                     (slice 3)
    └── coverage.py                   (slice 4)

tests/unit/
├── test_double_critic_stage.py       (slice 1, 1.5)
├── test_complexify_operator.py       (slice 2)
├── test_taxonomy_models.py           (slice 3)
├── test_taxonomy_generator.py        (slice 3)
├── test_taxonomy_coverage.py         (slice 4)
└── test_complexity_elo_stage.py      (slice 5)

tools/simula_eval/
├── README.md
├── metrics.py
├── compare.py
├── configs/
│   ├── 00-baseline.yaml
│   ├── 01-double-critic.yaml
│   ├── 01b-double-critic-strong.yaml
│   ├── 02-complexify.yaml
│   ├── 03-taxonomy.yaml
│   └── 05-elo.yaml
└── taxonomies/
    └── personal_writing.yaml

docs/logs/2026/04/
├── 2026-04-25-15-32_simula-paper-review.md         (initial review)
├── 2026-04-25-15-50_simula-slice-0-baseline.md
├── 2026-04-25-17-30_simula-slice-1-double-critic.md
├── 2026-04-25-17-50_simula-slice-1-5-strong-critic.md
├── 2026-04-25-18-00_simula-slice-2-complexify.md
├── 2026-04-25-18-15_simula-slice-3-taxonomy.md
├── 2026-04-25-18-30_simula-slice-4-coverage.md
└── 2026-04-25-18-40_simula-slice-5-elo.md
```

Modified: `src/arka/config/models.py`, `src/arka/pipeline/stage_builder.py`,
`src/arka/pipeline/evol_instruct.py`, `tools/simula_eval/metrics.py`,
`.gitignore`.

## Commits on `wt/simula`

```
69eceda Simula slice 5: batch-Elo complexity scoring (red/green TDD)
a02ca37 Simula slice 4: level-ratio taxonomic coverage metric (red/green TDD)
ce957a5 Simula slice 3: taxonomy-driven generator (red/green TDD)
ac5080c Simula slice 2: complexify operator (red/green TDD)
e5a67fa Simula slice 1.5: wire llm_override on double-critic for strong critic
29818d6 Simula slice 1: double-critic filter (red/green TDD)
4b16a45 Simula slice 0: baseline pipeline + comparison harness
```

## Honest read

The slice-by-slice results validate Simula's framing on Jay's domain: the
taxonomy-driven generator is the single biggest diversity lever (66% jump in
average pairwise cosine distance, 100% leaf coverage from 50 generated
samples), the double-critic earns its keep only with a stronger model than
the generator (15% rejection rate vs 0.5% same-family), and the calibrated
Elo distribution actually reflects depth/density rather than length.

What we did NOT prove: any of this improves downstream model performance on a
held-out task. The paper does that with Gemma-3-4B LoRA training on five
benchmarks; arka is not yet in that loop. The next obvious experiment is to
take a slice-3 dataset, fine-tune a small model, and compare against a
slice-0 baseline on a held-out set. That would close the loop on Simula's
"properties matter more than size" claim for our context.

## Followup queue

In rough priority order (highest leverage first):

1. **Per-record critic checkpointing** (item C) — saves real money on the
   next slice 5-style run that gets killed.
2. **Parallelize evol-instruct + taxonomy generators** (item E) — every
   later slice will amplify the wallclock cost.
3. **M3-driven taxonomy assignment** (item B) — unlocks coverage scoring
   on baseline datasets, makes slice 4 useful beyond slice 3 outputs.
4. **Cost telemetry** (item D) — surface per-stage `cost_usd`.
5. **Bootstrap CIs on Elo** (item G) — slice 5 results need this for
   honest reporting.
6. **M3-driven taxonomy expansion** (item A, slice 3.5) — the actual
   headline of the paper.
7. **Simula `c` parameter** (item F) — usability win for slice 2.

## Out of scope

- Replacing Evol-Instruct or any existing arka primitive.
- M3-driven taxonomy generation.
- Downstream training on the Simula-generated datasets.
- Cost optimization or production hardening of any of the new stages.
