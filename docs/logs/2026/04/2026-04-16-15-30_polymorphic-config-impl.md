# Polymorphic Config Implementation — Complete

**Date:** 2026-04-16  
**Status:** ✅ Done — 259 tests passing, 90% coverage

---

## What Changed

Replaced `enabled: bool` flags on all filter/dedup configs with a **polymorphic list design** using Pydantic v2 discriminated unions.

### Before

```yaml
dedup:
  exact:
    enabled: true
  near:
    enabled: false
    lsh_bands: 16

filters:
  target_count: 10
  length:
    enabled: true
    min_instruction_chars: 40
  language:
    enabled: false
  canary:
    enabled: false
```

### After

```yaml
dedup:
  - type: exact

filters:
  target_count: 10
  stages:
    - type: length
      min_instruction_chars: 40
```

**Presence = enabled. Absence = disabled. Order = execution order.**

## Files Changed

### Config models (`src/arka/config/models.py`)
- Removed `enabled: bool` from all 11 filter/dedup config classes
- Added `type: Literal["..."]` discriminator to each
- Created `FilterStageConfig` discriminated union (10 types)
- Created `DedupStageConfig` discriminated union (2 types)
- `FiltersConfig.stages: list[FilterStageConfig]` replaces 10 individual fields
- `ResolvedConfig.dedup: list[DedupStageConfig]` replaces `DedupConfig` object
- Added `FiltersConfig.get_stage_config()` helper for runtime lookups

### Stage builder (`src/arka/pipeline/stage_builder.py`)
- Replaced if/else chains with `_DEDUP_REGISTRY` dict + `_build_filter_stage()` dispatcher
- Stages instantiated by iterating `config.dedup` and `config.filters.stages` lists

### Pipeline stages (5 files)
- `cheap_filters.py`: Removed 3 `if not cfg.enabled` guards
- `filter_stages.py`: Removed 4 `if not filter_config.enabled` guards
- `dedup_stages.py`: Removed 2 `if not ctx.config.dedup.X.enabled` guards
- `ifd_stage.py`: Removed 1 `if not filter_config.enabled` guard
- `scoring_stages.py`: Removed 4 `if not X_config.enabled` guards
- `runner.py`: Updated 1 labeling_engine config lookup

### Example configs (8 files)
- All 8 example YAML files updated to new list-based syntax
- Minimal configs no longer need any dedup/filter noise

### Tests (16 files)
- 21 new polymorphic config tests in `test_polymorphic_config.py`
- Updated all _base_config helpers across 15 test files
- Removed old `enabled=True/False` patterns
- Integration test updated with list-based dedup construction

### Docs (7 files)
- `docs/configuration.md` — sections 6 & 7 fully rewritten for list-based config
- `docs/SPEC.md` — updated dedup/filter YAML examples
- `docs/rl-data-needed.md` — updated 5 YAML snippets from enabled-flag to list-based
- `docs/validation-matrix.md` — updated language about enabled checks
- `docs/plans/tier-a-next-three-slices-execution-plan.md` — updated 5 references
- `README.md` — updated quickstart YAML snippet
- Historical session logs (`docs/logs/`) left as-is (they document past state)

## Test Results

- **259 passed** (was 238 — added 21 new tests)
- **0 failed**
- **90% coverage** (unchanged)
- **Zero `.enabled` references remain** in src/ or tests/ Python code
