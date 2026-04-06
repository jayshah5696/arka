# Validation Matrix

This document defines the **currently supported** config/runtime matrix for arka and the quality checks we expect before calling a synthetic-data run healthy.

## What is exercised today

The automated matrix covers the currently implemented vertical slice.

### Config surface covered by automated tests

- **seed input formats**
  - JSONL
  - CSV
- **executor modes accepted by config and propagated into stage context**
  - `threadpool`
  - `realtime`
  - `provider_batch`
- **output formats**
  - `jsonl`
  - `chatml`
  - `alpaca`
- **dedup combinations**
  - dedup off
  - exact dedup only
  - near dedup only
  - exact + near dedup together
- **generation quality gate**
  - prompt-based generation
  - single-judge labeling filter
  - run report with `samples.jsonl`, `canaries.json`, `quality_distribution`, and `diversity_score`

Automated matrix file:
- `tests/integration/test_supported_options_matrix.py`

## What "supported" means right now

A config/runtime option is considered supported when all of the following are true:

1. the config validates through `ConfigLoader`
2. `StageBuilder` wires the expected stages
3. the pipeline completes end-to-end
4. the expected artifacts are written
5. the output is formatted correctly for the selected export format
6. drop reasons and run-report fields are inspectable

## Current quality bar for generation runs

For the current SFT slice, a run is in good standing when these checks pass.

### Required checks

- `run_report.json.status == "completed"`
- final dataset is non-empty
- `samples.jsonl` exists and is human-reviewable
- `canaries.json` exists
- when labeling is enabled, canary status is `pass` or any warning is explicitly reviewed
- `drop_reasons` are present for records removed by quality or dedup stages
- no unexpected parse/auth failures occurred in report or stage stats

### Strongly recommended checks

- `03_label_quality/stats.json` shows the filter is actually separating stronger vs weaker examples
- `low_quality_score` drops appear in verification runs that intentionally include weaker examples
- `diversity_score` is non-null when embeddings are configured and reachable
- `samples.jsonl` is manually spot-checked before training
- `clusters.parquet` is inspected when exact or near dedup is enabled

### Acceptable current limitations

- `diversity_score` may be `null` if embeddings are unavailable or fail; the run should not fail because of that
- near dedup is currently **MinHash-only** in implementation
- `labeling_engine.mode: multi` is not yet implemented end-to-end even though multi-judge is part of the longer-term spec
- `executor.mode` values other than `threadpool` are currently config/runtime compatibility surfaces, not distinct execution engines

## OSS-style validation workflow

### Fast local validation

```bash
just matrix
```

This runs the supported-option integration matrix only.

### Full local validation

```bash
just check
```

This runs lint, format-check, and the full test suite.

### Live verification against a real hosted model

Minimal example:

```bash
uv run arka --config examples/01-minimal.yaml --run-id verify-minimal
```

Resume/debug workflow:

```bash
uv run arka --config examples/07-resume-debug.yaml --run-id openrouter-debug-v1
uv run arka --config examples/07-resume-debug.yaml --run-id openrouter-debug-v1 --resume
```

Dedup + quality example:

```bash
uv run arka --config examples/06-dedup-quality-filter.yaml --run-id dedup-quality-check
```

## Recommended release checklist for this slice

Before claiming the current SFT slice is healthy:

- run `just check`
- run `just matrix`
- run one real provider-backed verification config
- inspect:
  - `run_report.json`
  - `samples.jsonl`
  - `canaries.json`
  - `stages/*/stats.json`
  - `dropped.parquet` for quality/dedup stages

## Out of scope for this matrix

These are planned or partially specified, but not yet part of the supported matrix:

- multi-judge conflict detection
- semantic dedup stage
- contamination checking
- non-seed data sources beyond current CSV/JSONL seed ingestion
- preference, triplet, reranker, eval, and trajectory outputs
- Anthropic-native adapter
