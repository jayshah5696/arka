# Simula Slice 0 — BEFORE Snapshot

**Date:** 2026-04-25
**Branch:** `wt/simula` (worktree at `/Users/jshah/Documents/GitHub/.arka-worktrees/simula`)
**Slice:** 0 — establish baseline before any Simula component is added.

## Setup

- **Worktree:** created via `nwt simula main` (zsh plugin `nwt` installed via oh-my-zsh symlink; backup of `~/.zshrc` at `~/.zshrc.bak.simula`).
- **Shared resources:** `.venv`, `.env`, `.pi`, `.rtk` symlinked from main checkout via `.nwt/shared.conf`.
- **Seed data:** copied 50 records from `/Users/jshah/Documents/GitHub/humanize-rl/seeds/human_seeds_v01.jsonl` into `scratch/simula-eval/seeds/human_seeds.jsonl`. Domains: blog (14), email (11), essay (8), technical (7), social (6), academic (4).
- **Models:** generator `google/gemini-3.1-flash-lite-preview` via OpenRouter; judge `google/gemini-3-flash` (deferred to Slice 1).
- **Comparison harness:** `tools/simula_eval/metrics.py` — uses fastembed `BAAI/bge-small-en-v1.5` to compute avg pairwise cosine distance (paper's "Embedding Diversity" metric); reads stage `stats.json` for yields/drop reasons. See `tools/simula_eval/README.md` for usage.

## Pipeline (slice 0 = control)

```
seeds (50)
  ─▶ normalize
  ─▶ prompt_based generate (target=150, multiplier=2)
  ─▶ exact dedup
  ─▶ near dedup (lsh_bands=16)
  ─▶ length filter (40–4096 chars response)
  ─▶ language filter (en)
  ─▶ output (chatml)
```

Quality-filter stage **omitted** for the baseline. Reason: the existing single-judge `labeling_engine` filter has a pre-existing `KeyError: 'response_quality'` when the judge LLM returns a partial scores dict. That bug is independent of Simula and tracked separately. The baseline now matches the paper's experimental protocol — generation + dedup, no critic — so Slice 1 (double-critic) can be compared against a clean control.

Config: `examples/simula/00-baseline.yaml`.

## Results

| Stage | in | out | dropped |
|---|---:|---:|---:|
| `01_source` | 0 | 50 | 0 |
| `02_normalize` | 50 | 50 | 0 |
| `02_generate` | 50 | 300 | 0 |
| `02c_exact_dedup` | 300 | 300 | 0 |
| `02d_near_dedup` | 300 | **217** | 83 (27.7%) |
| `02a_length_filter` | 217 | 217 | 0 |
| `02b_language_filter` | 217 | 217 | 0 |

Final dataset: `scratch/simula-eval/00-baseline/dataset.jsonl` — 217 chatml records.

### Intrinsic metrics (the Simula comparison surface)

| Metric | Value |
|---|---|
| Final count | **217** |
| Avg pairwise cosine distance (fastembed bge-small) | **0.248** |
| Text length min / median / max (chars) | 249 / 396 / 626 |
| Near-dup cluster count | 41 |
| Exact-dup count | 0 |

These are the numbers each subsequent slice will be compared against.

## Observations

1. **The generator is well-behaved on Jay's seeds.** First spot-checked sample: clean email reschedule, no AI-fluff opener, no em dashes — the prompt-template constraints in the config worked.
2. **Near-dedup is doing real work.** ~28% of generations were near-duplicates of each other or of the seeds. Without it, the diversity metric below would be inflated by repetition. Expected to drop further once Slice 3 (taxonomy generator) forces global coverage.
3. **No exact duplicates.** The generator never repeated itself verbatim in this run.
4. **Pre-existing bug to file:** `labeling_engine` filter crashes on partial judge responses. Should land in a separate ticket on `main`; not blocking Simula work.

## What this enables

- Slice 1 will add a `double_critic` filter immediately after `language_filter`. We expect: lower final count (some answers will fail correct/incorrect), no change to upstream generation/dedup.
- Slice 2 (complexification) will alter the generation step itself. We expect: higher complexity Elo (when slice 5 lands), small drop in average pairwise distance because complexified outputs of the same node-set cluster.
- Slices 3–5 will add taxonomy-driven generation, taxonomic coverage, and Elo complexity — at which point the comparison table grows three more columns.

## Files added in this slice

**Tracked:**
- `examples/simula/00-baseline.yaml` (config)
- `tools/simula_eval/metrics.py` (comparison harness)
- `tools/simula_eval/README.md`
- `.gitignore` updated to ignore worktree symlinks (`.pi`, `.rtk`, `.venv`)
- `docs/logs/2026/04/2026-04-25-15-50_simula-slice-0-baseline.md` (this log)

**Local-only (gitignored under `scratch/`):**
- `.nwt/shared.conf` and `.git/info/exclude` (worktree-local)
- `scratch/simula-eval/seeds/human_seeds.jsonl` (copied from humanize-rl)
- `scratch/simula-eval/00-baseline/dataset.jsonl` (run output)
- `scratch/simula-eval/00-baseline/metrics.json`
- `scratch/simula-eval/configs/runs/slice-00-baseline/...` (full arka run artifacts)
