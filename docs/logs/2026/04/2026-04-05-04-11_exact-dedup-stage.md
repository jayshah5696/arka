# Exact dedup stage

Date: 2026-04-05 04:11 UTC

## Summary

Implemented the next thin slice after generator + cheap-filter cleanup: exact dedup as a first-class stage.

## What changed

### Config
- Added `dedup` config section to `ResolvedConfig`
- Added:
  - `ExactDedupConfig`
  - `DedupConfig`
- Current shape:

```yaml
dedup:
  exact:
    enabled: false
```

Defaults are safe/off so existing configs continue to work.

### Stage
- Added `src/arka/pipeline/dedup_stages.py`
- New `ExactDedupStage` (`02c_exact_dedup`)
- When enabled, keeps the first `ConversationRecord` seen for a `content_hash` and drops later duplicates
- Dropped records get a stage event with:
  - `action = "dropped"`
  - `reason_code = "exact_duplicate"`
  - `details = "duplicate_of=<representative_id>"`

### Artifacts
- Writes:
  - `dropped.parquet`
  - `clusters.parquet`
  - `stats.json`
- `clusters.parquet` currently contains:
  - `cluster_id`
  - `representative_id`
  - `member_count`
  - `member_ids_json`

### StageBuilder wiring
- `StageBuilder` now inserts exact dedup after generation and before cheap filters
- Current order:
  - `01_source`
  - `02_normalize`
  - `02_generate`
  - `02c_exact_dedup` (optional)
  - cheap filters
  - labeling filter

## Tests

Added:
- `tests/unit/test_dedup_stages.py`
  - disabled pass-through
  - duplicate drop behavior
  - dropped artifact assertions
  - cluster artifact assertions
  - stats assertions

Updated config-bearing tests to include the new optional `dedup` section where appropriate.

## Validation

- `uv run pytest -q` → **102 passed**
- `uv run ruff check .` → clean
- `uv run ruff format --check src tests` → clean
- Coverage remains **95%**

## Notes

This thin slice dedups by existing `content_hash`, which in current record models is payload-based. That matches the current implementation reality and gets exact dedup into the pipeline now.

A future refinement can switch exact dedup to normalized instruction-text hashing if we want it to align more literally with the spec wording.

## Files changed

- `src/arka/config/models.py`
- `src/arka/pipeline/dedup_stages.py` — new
- `src/arka/pipeline/stage_builder.py`
- `tests/unit/test_dedup_stages.py` — new
- `tests/unit/test_stage_builder.py`
- `tests/unit/test_config_loader.py`
- `tests/unit/test_labeling_config.py`
- `tests/unit/test_example_configs.py`
- `tests/unit/test_pipeline_runner.py`
- `tests/unit/test_reports_and_stage_events.py`
- `tests/unit/test_seed_source_stage.py`
- `tests/unit/test_cheap_filters.py`
- `tests/unit/test_generation_stages.py`
- `tests/unit/test_labeling_filter_stage.py`
