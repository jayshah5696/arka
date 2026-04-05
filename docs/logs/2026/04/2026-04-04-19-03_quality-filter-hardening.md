# Quality filter hardening

Date: 2026-04-04 19:03

Continued in the direction of hardening the single-judge quality filter path.

## What changed

Implemented first-class dropped-record and reporting support for the labeling quality stage:

- `src/arka/pipeline/filter_stages.py`
  - `LabelingQualityFilterStage` now writes `dropped.parquet` in the stage work dir.
  - Dropped records get a `StageEvent` with:
    - `action: dropped`
    - `reason_code: low_quality_score`
    - detail string with actual vs minimum score
  - Stage now writes `stats.json` with:
    - `count_in`
    - `count_out`
    - `scored_count`
    - `dropped_count`
    - `drop_reasons`
    - `quality_distribution` (`mean/std/min/max`)

- `src/arka/pipeline/output.py`
  - Added `write_dropped_parquet()`.
  - Added dropped-artifact schema including:
    - `drop_stage`
    - `drop_reason`
    - `drop_detail`
  - Kept regular parquet round-trip behavior unchanged for normal stage artifacts.

- `src/arka/pipeline/runner.py`
  - Runner now reads per-stage `stats.json` when building stage stats.
  - `manifest.json` stage stats now include `dropped_count`.
  - `run_report.json` now includes:
    - richer `stage_yields`
    - aggregated `drop_reasons`
    - top-level `quality_distribution`

- `src/arka/pipeline/models.py`
  - Expanded `StageStat` with `dropped_count`, `drop_reasons`, and `quality_distribution`.

- `src/arka/core/paths.py`
  - Added helpers for stage-level artifact paths:
    - `stage_dropped_path()`
    - `stage_stats_path()`

## Tests updated

- `tests/unit/test_labeling_filter_stage.py`
  - Verifies `dropped.parquet` and `stats.json` are produced.
  - Verifies dropped reason is `low_quality_score`.
  - Verifies quality distribution summary.

- `tests/unit/test_reports_and_stage_events.py`
  - Verifies runner surfaces stage stats into `run_report.json`.
  - Verifies aggregated `drop_reasons` and `quality_distribution`.

- `tests/unit/test_pipeline_runner.py`
  - Updated expectations for richer stage stats in manifest output.

## Validation

Passed:

- `uv run pytest -q`
- `uv run ruff check .`
- `uv run ruff format --check .`

## Current status

This completes the first two recommended hardening steps from the direction note:

1. persist dropped records from the label quality stage
2. surface stage-level filter/report stats in `run_report.json`

Not done yet:

- richer label-path error taxonomy (`label_parse_failure`, auth vs retryable vs invalid structured response surfaced to the filter)
- explicit manifest/report role cleanup beyond current improvements
- canary artifact export
- persisted audit/export behavior for non-score failure cases

## Next best step

Implement explicit label-path failure handling so the stage can classify and optionally persist failures with reason codes like:

- `label_parse_failure`
- `label_auth_failure`
- `label_retryable_api_failure`
- `invalid_structured_response`

That would make the filter path more inspectable before moving to multi-judge work.
