# StageBuilder extraction and cheap filter stages

Date: 2026-04-05 11:25

## Summary

Executed the prioritized task list from `2026-04-05-10-30_implementation-review-and-next-steps.md`. All immediate and next-feature items completed.

## What was done

### Immediate cleanup

1. **Committed 1018 lines of uncommitted work** → `9e55145`
2. **Removed dead `OpenAIClientFactory`** → `fb5da4d` (tests updated to use `build_openai_client` directly)

### Structural: StageBuilder

3. **Extracted `StageBuilder(config, project_root).build()`** → `e771695`
   - `src/arka/pipeline/stage_builder.py`: takes `ResolvedConfig` + `project_root`, returns `list[Stage]`
   - CLI reduced from manual stage assembly to single `StageBuilder().build()` call
   - Source stage selection and filter composition are now in `_source_stages()` / `_filter_stages()`
   - Raises `ValueError` for unsupported `data_source.type`
   - 5 tests with 100% coverage

### Code quality

4. **Documented `_extract_json_text` greedy regex limitation** → `2c51ed1`
   - Added code comments explaining the known limitation
   - 9 edge-case tests: bare JSON, surrounding text, code fences, nested braces, two-object greedy match, no braces fallback, unclosed fence, whitespace

5. **Made canary check frequency configurable** → `c94caf7`
   - `label_batch()` now accepts `run_canary=True` (default)
   - 1 new test verifying canary skip behavior

### Feature: Cheap filter stages

6. **Added length and language filter stages** → `32889da`
   - `LengthFilterStage` (`02a_length_filter`): drops records outside configurable instruction/response char bounds
   - `LanguageFilterStage` (`02b_language_filter`): drops non-Latin-script records when `allowed: [en]` using 70% Latin char threshold heuristic
   - Both write `dropped.parquet` and `stats.json` following existing stage artifact pattern
   - Config: `filters.length` and `filters.language` sections with safe defaults (disabled by default)
   - StageBuilder orders: length → language → labeling (cheapest first)
   - 18 new tests (15 filter + 3 stage builder integration)

## Test count progression

55 → 54 (removed redundant factory test) → 59 (stage builder) → 68 (JSON extract) → 69 (canary) → 87 (cheap filters)

## Current state

- 87 tests, all passing
- Ruff clean
- 6 commits since session start

## Files changed

- `src/arka/pipeline/stage_builder.py` — New: stage assembly from config
- `src/arka/pipeline/cheap_filters.py` — New: LengthFilterStage, LanguageFilterStage
- `src/arka/config/models.py` — Added LengthFilterConfig, LanguageFilterConfig to FiltersConfig
- `src/arka/pipeline/stage_builder.py` — Wired cheap filters into build()
- `src/arka/llm/openai_client.py` — Removed dead OpenAIClientFactory
- `src/arka/llm/client.py` — Documented _extract_json_text limitation
- `src/arka/labeling/engine.py` — Added run_canary parameter
- `src/arka/cli.py` — Simplified to use StageBuilder
- `config.example.yaml` — Added length/language filter config examples
- `tests/unit/test_stage_builder.py` — New: 8 tests
- `tests/unit/test_cheap_filters.py` — New: 15 tests
- `tests/unit/test_llm_client.py` — 9 new extraction edge-case tests
- `tests/unit/test_labeling_engine.py` — 1 new canary skip test
- `tests/unit/test_openai_client.py` — Updated to use build_openai_client directly

## What is next (from the review doc, deferred items)

1. Multi-judge / conflict detection
2. Anthropic adapter
3. Exact/near dedup stages (SHA-256 exact, SimHash/MinHash fuzzy)
4. Generator stages (Evol-Instruct, Magpie)
5. docs/SPEC.md sync with implementation reality
