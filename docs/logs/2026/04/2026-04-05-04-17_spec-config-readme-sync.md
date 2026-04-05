# Spec, config, and README sync

Date: 2026-04-05 04:17 UTC

## Summary

Closed out the remaining documentation/config follow-up for the completed generator + exact-dedup work.

## What changed

### Config updates
- Updated `config.example.yaml`
- Updated `config.smoke.yaml`
- Updated `config.openrouter.yaml`
- Added `dedup.exact.enabled: false` to canonical configs so the config shape matches implementation

### New example config
- Added `config.examples.dedup-quality.yaml`
- This exercises the current implemented vertical slice:
  - prompt-based generation
  - exact dedup
  - cheap length/language filters
  - single-judge quality filter

### Config catalog
- Updated `docs/config-examples.md`
- Moved `config.examples.dedup-quality.yaml` from future examples into current examples
- Updated descriptions to reflect prompt-based generation now existing in the real pipeline

### SPEC sync
- Updated `docs/SPEC.md` to distinguish:
  - current implementation
  - planned target architecture
- Synced the following areas:
  - current data flow now includes prompt generation and exact dedup
  - artifact layout now reflects real stage directories
  - config example now reflects current config schema (`dedup.exact.enabled`, char-based cheap filters)
  - exact dedup section now documents current implemented behavior
  - `content_hash` comment now notes current payload-based reality vs future normalized-instruction target

### README
- Updated `README.md`
- Kept it under 50 lines per project rule
- Reflected the current implemented path and example configs

## Validation

- `uv run pytest -q` → **103 passed**
- `uv run ruff check .` → clean
- `uv run ruff format --check src tests` → clean

## Files changed

- `README.md`
- `config.example.yaml`
- `config.smoke.yaml`
- `config.openrouter.yaml`
- `config.examples.dedup-quality.yaml` — new
- `docs/config-examples.md`
- `docs/SPEC.md`
- `tests/unit/test_example_configs.py`
