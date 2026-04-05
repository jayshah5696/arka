# Fixes: run_id, stage_action, cost_usd, supports_json_schema

**Date:** 2026-04-05  
**Commit:** 001e3bd

## Changes Made

### ① run_id auto-generation (CLI fix)
- `--run-id` default changed from `"manual-run"` to `None`
- New `_resolve_run_id(cli_run_id, config_run_id)` function with cascade: CLI > config > UUID4
- Config `run_id: null` now correctly triggers auto-generation instead of silently being ignored
- 4 new tests: integration test for auto-generated UUID dir, 3 unit tests for `_resolve_run_id`

### ② cost_usd extraction from API response
- `_usage_from_response()` now reads `usage.total_cost` when present (e.g., OpenRouter)
- `cost_usd` remains `None` when provider doesn't include it — no behavior change for standard OpenAI
- 2 new tests: cost present case, cost absent case

### ③ stage_action — runner reads from stage instead of hardcoding
- Added `stage_action: str = "transformed"` class attribute to `Stage` ABC
- Source stages override to `"sourced"`, filter stages to `"filtered"`
- Runner's `_append_stage_events()` now takes `action` parameter, reads `stage.stage_action`
- Dropped records still set `action="dropped"` inside their own stage implementations (unchanged)
- Existing test updated to assert on action values

### ④ supports_json_schema config flag
- Added `supports_json_schema: bool | None = None` to `LLMConfig`
- `OpenAICompatibleJsonSchemaStrategy.is_applicable()` checks flag first:
  - `True` → use json_schema strategy (works for Together AI, Fireworks, etc.)
  - `False` → skip even for OpenRouter
  - `None` → legacy URL-pattern fallback (`"openrouter.ai" in base_url`)
- 2 new tests: flag=True with non-OpenRouter URL, flag=False with OpenRouter URL

## Test Results
- 94 tests passing (was 87), 95% coverage
- Lint clean (ruff check + format)

## Files Changed (11)
- `src/arka/cli.py` — run_id resolution
- `src/arka/config/models.py` — supports_json_schema field
- `src/arka/llm/client.py` — strategy applicability + cost extraction
- `src/arka/pipeline/stages.py` — stage_action default
- `src/arka/pipeline/runner.py` — read stage_action
- `src/arka/pipeline/source_stages.py` — stage_action = "sourced"
- `src/arka/pipeline/cheap_filters.py` — stage_action = "filtered"
- `src/arka/pipeline/filter_stages.py` — stage_action = "filtered"
- `tests/unit/test_cli.py` — 4 new tests
- `tests/unit/test_llm_client.py` — 4 new tests (2 strategy + 2 cost)
- `tests/unit/test_reports_and_stage_events.py` — action assertions added
