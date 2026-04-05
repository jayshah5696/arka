# Config example catalog

This project should maintain a growing set of commented example configs.

## Current examples

### Root configs

- `config.smoke.yaml`
  - smallest runnable JSONL seed smoke path
  - source -> normalize -> prompt-based generate

- `config.example.yaml`
  - baseline OpenAI-compatible example
  - includes prompt-based generation and optional exact dedup toggle
  - good starting point for local edits

- `config.openrouter.yaml`
  - OpenRouter-backed prompt generation + single-judge quality filter example
  - closer to the current live vertical slice

### Commented examples

- `config.examples.verify-openrouter.yaml`
  - small, debug-friendly live OpenRouter verification run
  - designed to exercise both kept and dropped paths
  - best for validating `dropped.parquet`, `stats.json`, and `run_report.json`

- `config.examples.resume-openrouter.yaml`
  - same overall path, but framed around resume/debug workflow
  - best for repeated runs using a stable `--run-id`

- `config.examples.csv-seeds.yaml`
  - CSV seed ingestion example
  - best for teams with spreadsheet-style starting data

- `config.examples.dedup-quality.yaml`
  - prompt generation + exact dedup + cheap filters + quality filter example
  - best for exercising the current end-to-end synthetic-data path

## Recommended future examples

Add these as slices land:

- `config.examples.multi-judge.yaml`
- `config.examples.preference-pairs.yaml`
- `config.examples.pdf-grounded.yaml`
- `config.examples.contamination-audit.yaml`

## Design rules for every example config

Each example should explain:

- what it is for
- when to use it
- required env vars
- expected input files
- expected output artifacts
- what stages/features are enabled
- common pitfalls

## Naming rule

Use:

- root `config.*.yaml` for the few canonical entry examples
- `config.examples.*.yaml` for the expanding catalog of commented use-case configs
