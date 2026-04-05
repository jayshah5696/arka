# arka (अर्क)

Config-driven synthetic data generation framework built from first principles.

## Quick Start

```bash
just setup
export OPENAI_API_KEY=your_key_here
printf '{"instruction":"Say hello","response":"Hello"}\n' > seeds.jsonl
uv run arka --config config.smoke.yaml --run-id smoke-run
```

Artifacts are written under `runs/<run_id>/` plus the configured final JSONL output path.

## Common Commands

- `just test` — run tests
- `just check` — lint, format-check, test
- `just run` — run `arka` with default `config.yaml`
- `uv run arka --config config.smoke.yaml --run-id smoke-run` — run smoke pipeline

## Current Implemented Slice

- typed Pydantic boundary models and internal stage protocol
- seed source stage for JSONL/CSV input
- normalize transform stage
- resumable pipeline runner with SQLite checkpoint state
- Parquet stage artifacts plus final JSONL dataset output
- `manifest.json` and `run_report.json`
- single-judge labeling quality filter with:
  - rubric-based scoring
  - dropped-record persistence (`dropped.parquet`)
  - stage stats (`stats.json`)
  - failure classification for label-path errors
- OpenAI-compatible client path with:
  - exponential retry backoff
  - provider-native structured output preferred
  - OpenRouter JSON-schema structured-output path
  - prompt-parse fallback for degraded compatibility

## Practical Provider Story

The canonical config model is still `provider: openai`, but practical live verification already runs through OpenAI-compatible routing as well, including OpenRouter-backed paths. In other words: the client interface is OpenAI-shaped, while OpenRouter-compatible usage is a supported real path today.

## Key Files

- `config.smoke.yaml` — simplest runnable config
- `config.example.yaml` — baseline OpenAI example
- `config.openrouter.yaml` — OpenRouter-compatible example
- `docs/config-examples.md` — commented config catalog
- `rubrics/sft_quality.yaml` — starter labeling rubric
- `docs/SPEC.md` — approved engineering spec
- `docs/decisions/0001-boundary-modeling.md` — boundary-modeling ADR
