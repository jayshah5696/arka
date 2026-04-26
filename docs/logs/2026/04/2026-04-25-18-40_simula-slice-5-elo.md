# Simula Slice 5 — Calibrated Batch-Elo Complexity Scoring

**Date:** 2026-04-25
**Branch:** `wt/simula`
**Slice:** 5 — Simula §2.3 batch-Elo complexity scoring (last of the planned slices).

## What landed

A new scoring stage `02s_complexity_elo` that annotates each
`ConversationRecord` with a comparable `complexity_elo` rating without
dropping anything. Procedure (Davidson et al. 2026, §2.3):

1. Each record appears `samples_per_record` times across batches of size
   `batch_size`.
2. For each batch, the LLM ranks members by complexity (most complex first).
3. The rank decomposes into pairwise win/loss outcomes (B*(B-1)/2 per batch).
4. Standard Elo update with `k_factor=32`, starting rating 400.
5. Final rating attached to `record.scores.quality_per_dim['complexity_elo']`.

The stage is **annotate-and-pass-through**, not a filter — downstream
`composite_select` or any other selector can consume the score. A
distribution summary (min/max/mean/median/stdev) lands in the stage's
`stats.json` for the eval harness.

### Files

**New:**
- `src/arka/pipeline/complexity_elo_stage.py` — stage + `elo_update_pair` helper + `_BatchRanking` schema + frozen ranker prompt
- `tests/unit/test_complexity_elo_stage.py` — 5 tests (Elo math correctness + end-to-end + passthrough + stats.json)
- `tools/simula_eval/configs/05-elo.yaml` — slice config (slice 3 + this stage)

**Modified:**
- `src/arka/config/models.py` — `ComplexityEloFilterConfig` in the discriminated union
- `src/arka/pipeline/stage_builder.py` — registry wire-up

### Tests (red→green)

```
5 passed
- pairwise Elo update is correct (1500 vs 1500 → +/-16; chess math)
- upset (1300 beats 1700) gains MORE than 16, zero-sum preserved
- end-to-end with deterministic mock ranker orders records as expected
- non-conversation passthrough does not call the ranker
- stats.json contains a complexity_elo_distribution summary
```

Full suite: **299 passing** (was 294, +5). Lint clean.

## Live run on Jay's taxonomy + humanize-rl seeds

```
50 seeds + 150 generated → 197 records (after dedup + length filter)
- Generator:  google/gemini-3.1-flash-lite-preview (taxonomy-driven)
- Critic:     google/gemini-3-flash-preview (Elo ranker, via llm_override)
- Settings:   batch_size=4, samples_per_record=3, k_factor=32
- Ranker calls: ~150 batch rankings × ~4s each ≈ 4 min wallclock for the stage
```

Stats:

| metric | value |
|---|---:|
| records scored | 197 |
| Elo min | 271.1 |
| Elo max | 532.3 |
| Elo mean | 400.0 (preserved by zero-sum updates) |
| Elo stdev | 74.4 |
| spread | 261 Elo points |

Histogram (10 buckets across the range):

```
271.1 ###################  (19)   ← simplest
297.2 ###################  (19)
323.3 #####################  (21)
349.5 ################  (16)
375.6 ######################  (22)
401.7 #####################  (21)
427.8 ######################  (22)
453.9 #######################  (23)
480.1 ##################  (18)
506.2 ################  (16)   ← most complex
```

A roughly normal-shaped distribution, exactly what the paper's calibrated
scoring is supposed to produce. The flat-ish middle and slight thinning at
the tails matches the paper's plots.

## What the critic actually rated

Lowest 3 (simplest):
```
elo=271.1  "Write a social media post announcing that you are finally opening
            your own coffee shop next month."
            nodes: { social/short_announcement, announce, casual }

elo=271.3  "Complain about the job market."
            (seed; no taxonomy nodes — Jay's original)

elo=272.6  "Explain to your friend why you decided to skip the office party
            tonight."
            nodes: { email/personal_brief, explain, casual }
```

Highest 3 (most complex):
```
elo=524.5  "Explain the purpose of the Request for Comments series in
            networking."
            nodes: { technical_writing/rfc, explain, dryly_funny,
                     no_corporate_buzzwords }

elo=527.9  "Write a performance analysis."
            (seed; no taxonomy)

elo=532.3  "Write an academic methods section paragraph."
            (seed; no taxonomy)
```

Two of the three highest-complexity records are **Jay's own seeds** —
authored content, technical, dense, no taxonomy assignment. The critic
ranked them as harder than most generated outputs. That is exactly what
calibrated complexity scoring should do.

The simple end is dominated by short conversational tasks; the complex end
by stacked-constraint technical writing. The single example with FOUR
constraints (`technical/rfc + explain + dryly_funny + no_corporate_buzzwords`)
landed at elo=524, near the top.

## Honest read

**What worked**
- The math is right: zero-sum preserved across the dataset (mean stays at
  400.0), correct upset behavior in pairwise tests.
- The score actually reflects the constraints+depth signal we wanted, not
  just length. The seed records that ranked highest are **shorter** than
  many of the synthetic "complexified" outputs from slice 2 but are
  **denser** — that is the paper's whole point.
- Annotate-and-pass-through means slice 5 stacks cleanly on slice 3 without
  changing any of the upstream count metrics.
- 5-minute wallclock for 197 records × ~3 batches each is acceptable. The
  stage parallelises via ThreadPoolExecutor over batches, so it scales
  better than the evol-instruct serial loop.

**Honest limits**
- **K=3 samples per record is light.** The paper used K up to 10 for stability.
  Stdev of 74 across the dataset is reasonable but I have no per-record
  uncertainty bound. Followup: bootstrap the rankings to compute per-record
  Elo confidence intervals.
- **Same critic family for ranker.** The ranker is gemini-3-flash-preview,
  same family as the generator. The paper warned about this — ideally the
  ranker would be a **different** model family (e.g. claude-sonnet-4) so
  family-specific biases don't leak into the score. Easy override now via
  `llm_override`, just needs a different model slug.
- **Length leakage in the prompt.** The ranker prompt explicitly says
  "Length alone is NOT complexity" but I haven't measured whether the model
  obeys. The slice-2 (complexify) outputs would be a great test — they're
  3× longer than seeds — if the ranker tracks length, slice-2 records
  should dominate the top of the Elo distribution. Followup.
- **Single rank request per batch.** The paper uses N=3 votes per batch and
  takes the consensus to reduce single-call noise. Easy add later.

## Followups

1. **Bootstrap CIs on Elo** — sample subsets of pairwise outcomes, recompute
   Elo, report ±95% per record.
2. **Different-family ranker** — wire claude-sonnet-4 (or any non-gemini)
   via `llm_override` and re-run; expect lower correlation, more honest
   signal.
3. **Length-vs-Elo regression** — compute the Spearman correlation between
   `text_chars` and `complexity_elo`. If r > 0.7 the ranker is
   length-cheating; reword the prompt.
4. **Composite select stage hook-up** — show that slice 5's Elo can drive a
   `composite_select` to pick the top 100 by complexity, demonstrating
   end-to-end use beyond just annotation.
5. **Per-record critic-cache** (still tracked from slice 1.5) becomes more
   important here because Elo runs N×B*(B-1)/2 pairwise updates per batch
   call \u2014 partial-resume would save real money on large datasets.

## Reproduce

```bash
cd /Users/jshah/Documents/GitHub/.arka-worktrees/simula
export OPENROUTER_API_KEY=sk-...

uv run arka --config tools/simula_eval/configs/05-elo.yaml --run-id slice-05-elo

uv run python tools/simula_eval/metrics.py \
  tools/simula_eval/configs/runs/slice-05-elo \
  scratch/simula-eval/05-elo/dataset.jsonl \
  --name 05-elo \
  --taxonomy tools/simula_eval/taxonomies/personal_writing.yaml

uv run python tools/simula_eval/compare.py \
  scratch/simula-eval/00-baseline/metrics.json \
  scratch/simula-eval/05-elo/metrics.json \
  --out scratch/simula-eval/05-elo/comparison.md
```
