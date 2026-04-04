# Scaffold Finish Cycle

Completed the next scaffold-finishing cycle focused on the remaining thin-slice foundation gaps:
- stage event emission
- run report artifact
- example config files
- threadpool-backed batch completion

## Changes made

### 1. Automatic stage events in the runner
Updated:
- `src/arka/pipeline/runner.py`

Behavior now:
- after each stage output, the runner appends a `StageEvent`
- events are stored per record with sequential `seq` values per record
- current action value is `transformed`

This makes stage traversal inspectable and moves the runtime closer to the approved record lineage/event model.

### 2. `run_report.json` artifact
Updated:
- `src/arka/core/paths.py`
- `src/arka/pipeline/runner.py`

Added report output at:
- `runs/<run_id>/report/run_report.json`

Current report includes:
- `run_id`
- `config_hash`
- `timestamp`
- `stage_yields`
- `final_count`
- `dataset_path`

This is still a minimal skeleton, but it closes the gap between pure manifest-only output and the spec’s expected report artifact direction.

### 3. Example config files
Added:
- `config.example.yaml`
- `config.openrouter.yaml`

These load through the actual config system and are covered by tests.

`config.openrouter.yaml` is set up for the future OpenRouter path with:
- `base_url: https://openrouter.ai/api/v1`
- model example: `google/gemini-3.1-flash-lite-preview`
- OpenRouter-compatible headers under `openai_compatible`

### 4. Threadpool-backed `complete_batch`
Updated:
- `src/arka/llm/client.py`

`complete_batch()` now supports:
- `max_workers`
- concurrent execution via `ThreadPoolExecutor`
- preserving output order to match input order

### 5. Config model alignment
Updated:
- `src/arka/config/models.py`

Added:
- `run_id: str | None = None`

This allows the example config shape to stay aligned with the spec’s YAML direction.

## Tests added/updated first

Added or expanded tests for:
- stage event emission
- `run_report.json` writing
- example config loading for both OpenAI and OpenRouter-style config files
- `complete_batch(..., max_workers=...)`

## Validation

All passing:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Final status:
- 25 tests passed

## Current scaffold state

The project now has a strong Slice 1 scaffold:
- uv + just project setup
- typed config models
- typed records
- explicit stage protocol
- resumable pipeline runner
- checkpointing metadata
- Parquet + JSONL outputs
- manifest + minimal run report
- stage event emission
- tenacity-based retry logic
- threadpool batch completion
- OpenAI-compatible config path for OpenRouter use later
- explicit Anthropic defer note

## Still intentionally unfinished

Not yet complete relative to the full approved spec:
- richer `LLMOutput.error` propagation strategy
- full report fields like cost, canaries, quality distributions, etc.
- provider-native structured output APIs
- more record subclasses beyond the initial typed base/conversation shapes
- Anthropic adapter

## Suggested next implementation area

The next step is no longer scaffold work. It should move into true Slice 1-to-2 productization, likely in this order:
1. enrich `LLMOutput.error` and provider error taxonomy
2. add a real source stage + dummy smoke pipeline command
3. start Slice 2 LabelingEngine scaffolding
