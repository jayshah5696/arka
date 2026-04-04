# Slice 1 Scaffold and TDD Progress

Implemented the first thin Slice 1 foundation scaffold using uv and red/green TDD.

## User decisions applied

- Full `uv` project setup: yes
- Data layer: `polars`
- LLM provider: OpenAI first
- Config should allow alternate OpenAI-compatible base URLs for future OpenRouter-style use
- Use red/green TDD explicitly

## Changes made

### Project/setup
- Ran `uv init --package --python 3.12 .`
- Added dependencies:
  - runtime: `openai`, `polars`, `pydantic`, `pyyaml`
  - dev: `pytest`, `pytest-cov`, `ruff`
- Added `.gitignore`
- Added `Makefile`
- Added `.agents/napkin.md`
- Added `STATUS.md`

### AGENTS
Updated `AGENTS.md` to explicitly require red/green TDD for logic changes.

### Tests written first
Created failing tests for:
- `tests/unit/test_config_loader.py`
- `tests/unit/test_paths.py`
- `tests/unit/test_checkpoint.py`
- `tests/unit/test_pipeline_runner.py`
- `tests/unit/test_openai_client.py`
- `tests/unit/test_cli.py`

### Minimal implementation added
- `src/arka/config/models.py`
- `src/arka/config/loader.py`
- `src/arka/core/paths.py`
- `src/arka/pipeline/checkpoint.py`
- `src/arka/pipeline/models.py`
- `src/arka/pipeline/output.py`
- `src/arka/pipeline/runner.py`
- `src/arka/llm/openai_client.py`
- `src/arka/cli.py`

## What works now

- YAML config loads and validates through Pydantic
- `${ENV_VAR}` resolution works
- unknown config keys fail validation
- run directory layout is created under `runs/<run_id>/`
- SQLite checkpoint registry works
- pipeline stages can execute and write Parquet stage artifacts
- final JSONL output is written
- resume mode skips already-materialized stage Parquet files
- OpenAI client factory respects configurable `base_url`
- CLI can run from a local `config.yaml`

## Validation

Commands run successfully:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Final test status:
- 9 tests passed

## Important current limitations

This is intentionally only the first thin scaffold, not full Slice 1 completion yet.
Still missing from the spec:
- real `LLMClient` interface with retries / typed outputs / error taxonomy
- provider capability registry
- stage stats artifacts
- richer manifest/run metadata
- failure recovery details and partial-write behavior
- proper CLI args like `--config` and `--resume`
- explicit stage protocol/base abstractions

## Recommended next implementation step

Continue Slice 1 with the next red/green cycle:

1. write failing tests for a real `LLMClient`
2. implement:
   - `complete()`
   - retry behavior
   - auth vs retryable error classification
   - OpenAI-compatible base URL support
3. then add CLI argument parsing for:
   - `--config`
   - `--run-id`
   - `--resume`

That keeps progress aligned with the approved spec while staying thin-sliced.
