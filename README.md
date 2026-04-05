# arka (अर्क)

Config-driven synthetic data generation framework built from first principles.

## Quick Start

```bash
just setup
export OPENAI_API_KEY=your_key_here
printf '{"instruction":"Say hello","response":"Hello"}\n' > seeds.jsonl
uv run arka --config config.smoke.yaml --run-id smoke-run
```

Artifacts will be written under `runs/smoke-run/` and `output/smoke-dataset.jsonl`.

## Common Commands

- `just test` — run tests
- `just check` — lint, format-check, test
- `just run` — run `arka` with default `config.yaml`
- `uv run arka --config config.smoke.yaml --run-id smoke-run` — run smoke pipeline

## Current Implemented Slice

- typed records and stage protocol
- seed source stage (JSONL/CSV)
- normalize transform stage
- resumable pipeline runner
- Parquet stage artifacts + JSONL output
- manifest + `run_report.json`
- LabelingEngine single-judge scaffold

## Key Files

- `config.smoke.yaml` — simplest runnable config
- `config.example.yaml` — basic OpenAI config
- `config.openrouter.yaml` — OpenRouter-compatible example
- `docs/config-examples.md` — catalog of commented example configs
- `rubrics/sft_quality.yaml` — starter labeling rubric
- `docs/SPEC.md` — approved engineering spec
