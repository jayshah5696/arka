# Examples

Runnable example configs live here.

## What this directory is for

These files are user-facing examples, not test fixtures.
They are organized from simplest to more advanced workflows.

## Example order

1. `01-minimal.yaml` — smallest runnable seed-to-SFT path
2. `02-openrouter-quickstart.yaml` — OpenRouter + local embeddings quickstart
3. `03-csv-seeds.yaml` — CSV seed ingestion
4. `04-evol-instruct.yaml` — multi-round Evol-Instruct
5. `05-pdf-grounded.yaml` — PDF chunk to grounded generation
6. `06-dedup-quality-filter.yaml` — dedup + quality filter
7. `07-resume-debug.yaml` — resume-oriented debug workflow

## Seeds and PDFs

- `examples/seeds/` contains example seed inputs used by the configs.
- `examples/pdfs/` is where PDF demo inputs should live.

## Future placeholders

See `examples/future/README.md` for planned-but-not-yet-implemented example shapes.

## Validation

Run:

```bash
uv run python scripts/validate_examples.py
```

This checks:
- all example YAML files parse
- required header fields are present
- cost values are valid
- OpenRouter env-var usage is consistent
- output paths are relative
- referenced example seed files exist
- future placeholders include TODO markers
