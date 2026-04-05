# arka (अर्क)
Config-driven synthetic data generation framework built from first principles.

## Quick Start
```bash
just setup
export OPENAI_API_KEY=your_key_here
printf '{"instruction":"Say hello","response":"Hello"}\n' > seeds.jsonl
uv run arka --config config.smoke.yaml --run-id smoke-run
```
Artifacts land in `runs/<run_id>/` plus the configured JSONL output path.

## Common Commands
- `just test` — run tests
- `just matrix` — run the supported config/runtime validation matrix
- `just check` — lint, format-check, test
- `just run` — run `arka` with default `config.yaml`
- `uv run arka --config config.smoke.yaml --run-id smoke-run` — run smoke pipeline

## Current Implemented Path
- seed source (JSONL/CSV) → normalize → prompt-based generate → exact dedup → near dedup
- cheap filters: length, language
- single-judge labeling quality filter
- resumable runner with SQLite checkpoints
- artifacts: `data.parquet`, `dropped.parquet`, `clusters.parquet`, `stats.json`, `manifest.json`, `run_report.json`, `samples.jsonl`, `canaries.json`
- diversity embeddings: local HuggingFace-style model by default (`all-MiniLM-L6-v2` via FastEmbed), configurable to provider/OpenAI-compatible APIs
- OpenAI-compatible client with structured-output strategy chain

## Example Configs
- `config.smoke.yaml` — smallest runnable path
- `config.example.yaml` — baseline OpenAI example
- `config.openrouter.yaml` — OpenRouter-compatible example
- `config.examples.dedup-quality.yaml` — generator + exact dedup + cheap filters + quality filter
- `docs/config-examples.md` — config catalog
- `docs/validation-matrix.md` — supported options, quality bar, and release checks
- `config.examples.pdf-grounded.yaml` — PDF chunk to grounded generation example
- `config.examples.evol-instruct.yaml` — multi-round Evol-Instruct example

OpenAI-compatible routing, including OpenRouter-backed paths, is supported in practice.
