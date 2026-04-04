# Source Stage and Smoke Pipeline Implementation

After committing the Slice 1 scaffold snapshot, completed the next recommended step: a real source stage plus a smokeable end-to-end pipeline path.

## Commit completed

Created commit:

```text
Scaffold Slice 1 foundation with typed records and runner artifacts
```

## Local noise cleanup

Removed local ignored artifacts after commit:
- `.coverage`
- `.DS_Store`
- `.pytest_cache/`
- `.ruff_cache/`
- `src/arka/__pycache__/`
- `tests/unit/__pycache__/`

## What was implemented

### 1. Real seed source stage
Added:
- `src/arka/pipeline/source_stages.py`

Implemented `SeedSourceStage`:
- reads JSONL seed files
- reads CSV seed files
- emits typed `ConversationRecord` records
- computes `content_hash`
- computes `seed_file_hash`
- sets `source.type = "seed"`
- sets basic lineage metadata
- trims instruction/response text into normalized payloads

### 2. Simple normalize transform stage
Added:
- `src/arka/pipeline/transforms.py`

Implemented `NormalizeConversationStage`:
- trims conversation payload fields for `ConversationRecord`
- leaves non-conversation records unchanged

### 3. CLI wired to a real smoke pipeline path
Updated:
- `src/arka/cli.py`

Behavior now:
- if `data_source.type == "seeds"`, CLI builds a real stage list:
  - `SeedSourceStage`
  - `NormalizeConversationStage`

This turns the CLI from a config-only scaffold into a runnable tiny vertical slice.

### 4. Smoke config
Added:
- `config.smoke.yaml`

This gives the repo a simple example specifically for smoke execution.

## Tests added first

Added or updated tests for:
- JSONL seed loading into typed `ConversationRecord`s
- CSV seed loading into typed `ConversationRecord`s
- smoke config loading
- CLI tests with actual seed files present
- integration smoke pipeline run from config + seed file to final output

### New tests
- `tests/unit/test_seed_source_stage.py`
- `tests/integration/test_smoke_pipeline.py`

## Smoke behavior verified

The new smoke pipeline test proves:
- config loads
- source stage emits typed records
- stage Parquet artifact is written
- final JSONL is written
- `run_report.json` is written
- normalized output payload matches expectation

## Validation

All passing:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Final status:
- 29 tests passed

## Why this matters

This closes the gap the review called out.
The project now has a real tiny vertical path instead of only scaffold abstractions.

You can now point the system at a seed file and get:
- typed records
- stage artifacts
- output JSONL
- run report

That is a much stronger base for the next phase.

## Recommended next step

Now that the smoke path exists, the next sensible direction is:
1. improve `LLMOutput.error` / provider error taxonomy
2. add a tiny generator stage or source-to-generator smoke extension
3. begin Slice 2 LabelingEngine scaffolding
