# Examples

Runnable example configs live here.

## Run a local chat example

From the repository root, run the following commands. The first writes the included
five-message conversation to `examples/output/00-local-chat-dataset.jsonl`.
The second reuses the two named checkpoints. Neither command needs an API key.

```bash
uv run arka --config examples/00-local-chat.yaml --run-id local-chat
uv run arka --config examples/00-local-chat.yaml --run-id local-chat --resume
```

For answer-only SFT, set `output.include_reasoning: false` to omit Arka’s separate
reasoning field. Literal text already in assistant messages stays in the dataset.

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
7. `07-resume-debug.yaml` uses a named model and version 2 stage IDs. Resume with the same config and run ID plus `--resume`. Start a new run ID when moving from the old version 1 example.
8. `08-privacy-guardrails.yaml` checks similarity and canary phrases.
9. `09-decision-models.yaml` uses seed gating, candidate ranking, and decision judges without an LLM fallback.
10. `10-quality-repair.yaml` repairs eligible failed answers once and checks replacements before deduplication.
11. `11-distilled-quality-train.yaml` trains a classifier after you supply labeled artifacts from independent sources.
12. `12-distilled-quality-score.yaml` annotates answers using the saved model from example 11.
13. `13-source-audit.yaml` judges a deterministic sample per source without filtering records.

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
