# Simula Slice 1 — Double-Critic Filter

**Date:** 2026-04-25
**Branch:** `wt/simula`
**Slice:** 1 — Simula §2.2 double-critic, anti-sycophancy filter.

## What landed

A new pipeline stage `03_double_critic` that, for each (instruction, response) pair,
runs **two independent** structured LLM calls:

1. `_YES_SYSTEM` — "Is the response **correct**? yes / no + one-line reason."
2. `_NO_SYSTEM`  — "Is the response **incorrect**? yes / no + one-line reason."

ACCEPT iff `yes_verdict == "yes"` AND `no_verdict == "no"`. Anything else is dropped
with reason `double_critic_disagreement`. LLM failures drop with reason
`double_critic_llm_error` (we never silently accept).

The full audit trail
`{yes_verdict, no_verdict, yes_reason, no_reason}` is attached to
`record.scores.quality_per_dim["double_critic"]` for downstream inspection.

### Files

**New:**
- `src/arka/pipeline/double_critic_stage.py` — stage + `CriticVerdict` schema + frozen prompts
- `tests/unit/test_double_critic_stage.py` — 6 tests
- `tools/simula_eval/configs/01-double-critic.yaml` — slice config (extends slice 0)
- `tools/simula_eval/compare.py` — render side-by-side metrics tables

**Modified:**
- `src/arka/config/models.py` — added `DoubleCriticFilterConfig` to the discriminated union
- `src/arka/pipeline/stage_builder.py` — registered the stage

### Tests

```
6 passed (red→green TDD)
- accepts when both critics agree (yes=YES & no=NO)
- rejects when yes-critic says NOT correct
- catches sycophancy (yes-critic agrees but no-critic flags incorrect)
- writes drop_reasons["double_critic_disagreement"] to stats.json
- passes through non-conversation records untouched
- yes-prompt and no-prompt are semantically inverse
```

Full suite: **268 passed** (was 262 before; +6).
Coverage on the new file: **87%**.

## Live run on humanize-rl seeds

Same 50 seeds, same generator (`gemini-3.1-flash-lite-preview`), same dedup. Only
the critic stage is added.

```
| stage             | slice 0 (baseline) | slice 1 (double-critic) |
|-------------------|--------------------|-------------------------|
| 02_generate       | 300 → 300          | 300 → 300               |
| 02d_near_dedup    | 300 → 217 [83]     | 300 → 212 [88]          |
| 02a_length_filter | 217 → 217          | 212 → 212               |
| 02b_language_filter | 217 → 217        | 212 → 212               |
| 03_double_critic  | (n/a)              | 212 → 211 [1]           |
```

| metric | 00-baseline | 01-double-critic | delta |
|---|---:|---:|---:|
| final_count | 217 | 211 | -6 (-2.8%) |
| avg pairwise cosine distance | 0.2480 | 0.2444 | -0.0036 (-1.5%) |
| text_chars_median | 396 | 418 | +22 |
| `double_critic_disagreement` drops | — | **1** | — |
| wallclock | (resumed) | 7m54s | +2N LLM calls |

### Note on the dedup-count delta

slice-0 dropped 83, slice-1 dropped 88 — same generator, why? The generator is
non-deterministic and was re-run for slice 1 (no resume across seed boundary).
The 5-record swing is sampling noise on the LLM, not a real signal. The
`02_generate` count is identical (300 → 300), so the diff is downstream
clustering noise.

### The single rejection

The one `double_critic_disagreement` was an email draft about delaying a
dashboard launch. The yes-critic said "no" (not correct) because the response
was **truncated mid-sentence** — the generator's 600-char cap clipped it. The
no-critic said "no" (not incorrect) because the visible content was on-topic
and reasonable.

This is the right behavior: a half-finished response failed the strict
"correct?" gate even though it wasn't outright wrong. **The cost-of-being-wrong
asymmetry is exactly what Simula is designed to surface.**

## Honest read

**What worked**
- Plumbing is clean: stage, config schema, registry, tests, audit trail, drop
  reasons, parquet artifacts — all in place and consistent with the rest of the
  arka stage idiom.
- Real run completed and produced a meaningful catch on the first try.
- The audit trail (yes_reason / no_reason per record) gives us a debuggable
  signal for free; we can build on this when slice 5 (Elo complexity) lands.

**What didn't move much**
- **Only 1 of 212 records was dropped.** Same model on both sides
  (generator and critic both `gemini-3.1-flash-lite-preview`) is the
  worst-case for the sycophancy-detection effect — the critic shares the
  generator's blind spots. The Simula paper used a stronger model (Gemini 2.5
  Flash) as the critic against itself; even there the empirical lift was lower
  than the controlled lift.
- The critic is also being **lenient on Jay's seed domain** (open-ended
  writing). "Correct" is fuzzy for "draft an email" or "outline ML docs". The
  paper's strongest critic results were on MATH and MCQ where correctness is
  binary.

**What the metric did NOT show**
- Embedding diversity barely moved (0.248 → 0.244). We don't expect the critic
  to alter diversity — it filters, doesn't transform. Same shape, slightly
  fewer points. Confirms the critic is doing what it claims and nothing else.

## Followups

1. **Use a different / stronger critic model** via `llm_override` on the
   `DoubleCriticFilterConfig` (the field is already there but I haven't wired
   the runner to honor it yet). Try `gemini-3-flash` as the critic against
   `gemini-3.1-flash-lite-preview` as the generator. Expected: rejection rate
   moves to 5–15% on Jay's seeds, similar to the paper's empirical-setting
   range.
2. **Test on a math seed** (e.g. 10 GSM8k-style problems) — the headline domain
   for sycophancy, where binary correctness lets the critic earn its keep.
3. **Cost telemetry** — log per-stage `cost_usd` so we can compute "$ per
   accepted record" and track Simula's claimed 5× generation cost.
4. **Speed up.** 7m54s for 212 records × 2 calls is much slower than necessary
   given `max_workers=6`. Likely the OpenRouter call latency dominates and the
   ThreadPoolExecutor is not being saturated by the gemini side. Worth a
   profiling pass before slice 5 amplifies the problem.

## Reproduce

```bash
cd /Users/jshah/Documents/GitHub/.arka-worktrees/simula
export OPENROUTER_API_KEY=sk-...

# (one-time) copy seeds
mkdir -p scratch/simula-eval/seeds
cp /Users/jshah/Documents/GitHub/humanize-rl/seeds/human_seeds_v01.jsonl \
   scratch/simula-eval/seeds/human_seeds.jsonl

# slice 0 baseline (already in main)
uv run arka --config tools/simula_eval/configs/00-baseline.yaml --run-id slice-00-baseline
uv run python tools/simula_eval/metrics.py \
  tools/simula_eval/configs/runs/slice-00-baseline \
  scratch/simula-eval/00-baseline/dataset.jsonl --name 00-baseline

# slice 1 double-critic
uv run arka --config tools/simula_eval/configs/01-double-critic.yaml --run-id slice-01-double-critic
uv run python tools/simula_eval/metrics.py \
  tools/simula_eval/configs/runs/slice-01-double-critic \
  scratch/simula-eval/01-double-critic/dataset.jsonl --name 01-double-critic

# comparison
uv run python tools/simula_eval/compare.py \
  scratch/simula-eval/00-baseline/metrics.json \
  scratch/simula-eval/01-double-critic/metrics.json \
  --out scratch/simula-eval/01-double-critic/comparison.md
```
