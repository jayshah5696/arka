# Simula Slice 1.5 — Strong Critic via `llm_override`

**Date:** 2026-04-25
**Branch:** `wt/simula`
**Slice:** 1.5 — wire stage-level `llm_override` so the double-critic can use a stronger model than the generator.

## Why this slice exists

Slice 1's symmetric setup (`gemini-3.1-flash-lite-preview` for both generator
and critic) only rejected 1 of 212 records. The Simula paper warned about
exactly this: when the critic shares the generator's blind spots, it
under-rejects. The paper used a stronger critic model on purpose. We replicate.

## Plumbing

- `src/arka/pipeline/double_critic_stage.py` now reads
  `filter_config.llm_override` and merges it onto `ctx.config.llm` via the
  existing `resolve_llm_override(...)` helper (same pattern as
  `RewardModelScoringStage`). The override is applied ONLY when the user
  doesn't inject an `llm_client` via constructor (test-friendly).
- New unit test `test_double_critic_uses_llm_override_when_set` —
  monkeypatches the module-level `LLMClient` symbol and asserts the captured
  config carries the override model. Strong by-construction proof, no live API
  call required.
- New config `tools/simula_eval/configs/01b-double-critic-strong.yaml`:
  generator stays on `gemini-3.1-flash-lite-preview`, critic upgraded to
  `gemini-3-flash-preview` via the override.

Tests: 7 passing on the file (was 6, +1). Full suite still green.

## Live run on humanize-rl seeds

```
| stage             | slice 0          | slice 1 (lite-vs-lite) | slice 1.5 (lite-vs-flash3) |
|-------------------|------------------|------------------------|----------------------------|
| 02_generate       | 300 → 300        | 300 → 300              | 300 → 300                  |
| 02d_near_dedup    | 300 → 217 [83]   | 300 → 212 [88]         | 300 → 231 [69]             |
| 03_double_critic  | (n/a)            | 212 → 211 [1]          | 231 → 197 [34]             |
```

| metric | 00-baseline | 01 (lite critic) | 01b (flash-3 critic) |
|---|---:|---:|---:|
| final_count | 217 | 211 | 197 |
| critic rejection rate | n/a | **0.5%** (1/212) | **14.7%** (34/231) |
| avg pairwise cosine distance | 0.2480 | 0.2444 | **0.2404** |
| text_chars_median | 396 | 418 | 416 |
| wallclock (full run) | 5m | 7m54s | 4m (with resume) |

A 30× higher rejection rate by swapping just the critic model. Right inside the
paper's predicted 5-15% empirical-setting band.

The dataset-wide cosine distance dropped a bit further (0.244 → 0.240) — the
critic is preferentially rejecting records from the more "uniform" parts of
the distribution (incomplete email drafts, all looking similar). Diversity
metric still essentially unchanged (rounding noise), but worth noting.

## What the critic is actually catching

I sampled the 34 drops. 30/34 share the same shape: `yes=no, no=no`. The
yes-critic flags incomplete email format (no subject line, no
greeting/sign-off), the no-critic says the content isn't wrong. Result: drop.

```
inst: Draft an email to a project manager explaining why the deadline...
resp: The current timeline for the API migration does not account for...

yes_reason: "...lacks basic email components such as a subject line and formal greeting/closing."
no_reason:  "...clearly explains why the deadline is unrealistic..."
```

This is **a real signal**, not a critic bug. It surfaces a genuine
prompt-design question on Jay's seeds: do we want body-only drafts or
full-email-format drafts? Either choice is defensible; the critic just
exposed that the generator was producing the former while the strong critic
expected the latter.

## Honest read

**What worked**
- Wiring took 4 lines of `run()` + 1 import + 1 unit test (RED then GREEN).
- The override fired exactly as designed: live run errored with
  `'google/gemini-3-flash is not a valid model ID'` on the first attempt
  (proving the override was applied; I had the wrong slug — corrected to
  `gemini-3-flash-preview`).
- Resume worked end-to-end: after the bad-model run failed, `--resume` skipped
  the upstream stages and only re-ran the critic. Saved ~50 generation calls.
- 14.7% rejection rate matches the paper's empirical setting band.

**What's still weak**
- The critic stage is monolithic — if killed mid-flight, resume rolls back to
  the start of the stage and redoes all 432 calls. Per-record critic caching
  (keyed by `record_id` + critic prompt hash) would fix this; the existing
  `embeddings_cache` table in `checkpoint.py` shows the pattern. Tracked as
  followup, NOT done in this slice.
- No cost telemetry yet. The `cost_usd` field exists on `StageStat` but the
  critic stage doesn't populate it. Easy follow-up once OpenRouter usage
  numbers are accessible from `LLMOutput`.

## Followups (kept distinct from slice 2)

1. **Per-record critic checkpoint** — new `critic_cache` table keyed by
   `(stage_name, record_id, critic_kind, prompt_hash) → verdict_json`. Crash
   recovery moves from per-stage to per-record granularity for any LLM filter.
2. **Cost telemetry** — populate `StageStat.cost_usd` from OpenRouter
   `usage` headers.
3. **Speed.** 4m for 432 calls × 6 workers ≈ 5–6s per call (gemini-3-flash is
   a thinking model). Possible win: structured-output strategies that don't
   require the model to think.
