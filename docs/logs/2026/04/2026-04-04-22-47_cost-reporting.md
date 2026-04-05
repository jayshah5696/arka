# Cost reporting surfaced into run reports

Implemented provider-reported cost propagation from generator outputs into stage stats and run reports.

## Changes made

### `src/arka/pipeline/models.py`
- Added `cost_usd: float | None = None` to `StageStat`.

### `src/arka/pipeline/generator_stages.py`
- `_write_parse_artifacts()` now accepts `raw_rows`.
- Sums `row.usage.cost_usd` across parsed raw generator rows.
- Writes `cost_usd` into `stats.json` as a rounded total when any provider-reported cost exists.
- Leaves `cost_usd` as `null` when the provider does not send pricing.

### `src/arka/pipeline/runner.py`
- `_build_stage_stat()` now reads `cost_usd` from stage `stats.json`.
- `_serialize_stage_stat()` includes `cost_usd` in serialized stage yields when present.
- `_build_run_report()` now sums `StageStat.cost_usd` across stages and writes top-level `cost_usd`.
- Added `_normalize_cost_usd()` helper.

## Tests added/updated

### Generator tests
- Added coverage that generator stage stats include `cost_usd` when the LLM output usage includes it.

### Runner/report tests
- Added coverage that `run_report.json` includes:
  - top-level `cost_usd` when stage stats provide it
  - per-stage `cost_usd` inside `stage_yields`
  - `cost_usd: null` when no provider-reported pricing exists

## Scope boundary preserved
- No pricing table was introduced.
- Costs are only surfaced when the provider includes them in the response usage payload.
- OpenAI-without-cost stays `null`.

## Verification

Passed:

```bash
uv run pytest -q tests/unit/test_generation_stages.py tests/unit/test_reports_and_stage_events.py tests/unit/test_pipeline_runner.py tests/unit/test_stage_builder.py tests/unit/test_llm_client.py tests/unit/test_checkpoint.py tests/unit/test_cli.py tests/integration/test_smoke_pipeline.py
uv run ruff check src/arka/pipeline/models.py src/arka/pipeline/generator_stages.py src/arka/pipeline/runner.py tests/unit/test_generation_stages.py tests/unit/test_reports_and_stage_events.py tests/unit/test_pipeline_runner.py
```
