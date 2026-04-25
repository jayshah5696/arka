# Simula evaluation harness

Reusable comparison rig for the Simula slices (double-critic, complexification,
taxonomy generator, level-ratio coverage, batch-Elo complexity).

Pipeline outputs and seed copies stay under `scratch/simula-eval/` (gitignored).
Configs live next to this README under `tools/simula_eval/configs/` so they
are versioned but not subject to the user-facing `examples/` header policy.

## One-time setup

```bash
# Copy your personal writing-assistant seeds into the (gitignored) scratch dir
mkdir -p scratch/simula-eval/seeds
cp /Users/jshah/Documents/GitHub/humanize-rl/seeds/human_seeds_v01.jsonl \
   scratch/simula-eval/seeds/human_seeds.jsonl

export OPENROUTER_API_KEY=sk-...
```

## Run a slice

```bash
# Slice 0 — baseline (no critic, no complexification, no taxonomy)
uv run arka --config tools/simula_eval/configs/00-baseline.yaml --run-id slice-00-baseline

# Compute metrics
uv run python tools/simula_eval/metrics.py \
  tools/simula_eval/configs/runs/slice-00-baseline \
  scratch/simula-eval/00-baseline/dataset.jsonl \
  --name 00-baseline
```

(Subsequent slices follow the same pattern; configs land at
`tools/simula_eval/configs/0N-<slice>.yaml` as they are added.)

## Metrics computed

| Metric | Definition | Slice that depends on it |
|---|---|---|
| `final_count` | rows in output JSONL | all |
| `stage_counts` | per-stage in/out/dropped/drop-reasons (read from arka's `stats.json`) | all |
| `avg_pairwise_cosine_distance` | mean cosine distance over fastembed `BAAI/bge-small-en-v1.5` embeddings (paper's "Embedding Diversity") | all |
| `text_chars_*` | character-length stats on flattened messages | sanity |

Slice 4 will add `coverage_by_level` (level-ratio taxonomic coverage).
Slice 5 will add `complexity_elo_distribution`.
