# Filter artifact dedup

Date: 2026-04-05 04:02 UTC

## Summary

Refactored duplicated cheap-filter artifact writing into a shared helper.

## What changed

- Updated `src/arka/pipeline/cheap_filters.py`
- Added shared `write_filter_artifacts(...)` helper
- `LengthFilterStage` now calls the shared helper
- `LanguageFilterStage` now calls the shared helper
- Removed duplicate `_write_artifacts()` implementations from both stages

## Why

Both cheap filters were writing the same:
- `dropped.parquet`
- `stats.json`
- identical count/drop-reason payload structure

This was the small cleanup called out in the next-steps doc.

## Validation

- `uv run pytest -q tests/unit/test_cheap_filters.py` → passed
- `uv run pytest -q` → **99 passed**
- `uv run ruff check .` → clean
- `uv run ruff format --check src tests` → clean

## Files changed

- `src/arka/pipeline/cheap_filters.py`
