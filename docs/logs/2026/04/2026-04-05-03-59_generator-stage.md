# Generator stage

Date: 2026-04-05 03:59 UTC

## Summary

Implemented Slice 5: a thin prompt-based generator stage wired through `StageBuilder`.

## What changed

### New stage
- Added `src/arka/pipeline/generation_stages.py`
- New `PromptBasedGeneratorStage` (`02_generate`)
- Calls `LLMClient.complete_structured(...)` with a strict `GeneratedConversation` schema
- Expands seed records to `generator.target_count * generator.generation_multiplier`
- Cycles through available seeds when requested generation volume exceeds seed count
- Emits `ConversationRecord`s with:
  - `source.type = "generated"`
  - lineage rooted at the original seed root
  - `parent_ids = [seed.id]`
  - `operator = "prompt_based"`
  - `round = 1`
  - `depth = 1`
- Uses content-stable SHA-256 hashes / ids based on payload + lineage

### Stage assembly
- Updated `src/arka/pipeline/stage_builder.py`
- New ordering:
  - `01_source`
  - `02_normalize`
  - `02_generate`
  - cheap filters
  - labeling filter
- Added `ValueError` for unsupported `generator.type`, mirroring existing `data_source.type` behavior

### Tests
- Added `tests/unit/test_generation_stages.py`
  - generation volume
  - generated source + lineage
  - stable ids/content hashes
  - empty-input behavior
- Updated `tests/unit/test_stage_builder.py`
  - generator stage inclusion
  - stage ordering
  - unsupported generator type
- Updated CLI / smoke tests to stub the generator LLM client
  - `tests/unit/test_cli.py`
  - `tests/integration/test_smoke_pipeline.py`

## Validation

- `uv run pytest -q` → **99 passed**
- `uv run ruff check .` → clean
- `uv run ruff format --check src tests` → clean
- Coverage remains **95%**

## Notes

This is the intended thin slice only:
- no personas
- no Evol-Instruct operators
- no generation artifact stats yet
- no report-level cost aggregation yet

## Files changed

- `src/arka/pipeline/generation_stages.py` — new
- `src/arka/pipeline/stage_builder.py`
- `tests/unit/test_generation_stages.py` — new
- `tests/unit/test_stage_builder.py`
- `tests/unit/test_cli.py`
- `tests/integration/test_smoke_pipeline.py`
