# Generator resume implementation

Implemented prompt-based generator resume/caching with raw response persistence.

## Changed
- Added `src/arka/pipeline/generator_stages.py` with:
  - `compute_prompt_hash(...)`
  - prompt-hash based resume
  - `raw_responses.jsonl` append-mode generation
  - zero-cost reparse from cached raw responses
  - partial-run recovery using `plan_index`
- Kept backward compatibility via `src/arka/pipeline/generation_stages.py` re-export shim.
- Extended `GeneratorConfig` in `src/arka/config/models.py` with safe defaults:
  - `prompt_template`
  - `temperature`
  - `max_tokens`
- Extended `LLMClient.complete(...)` to accept optional `temperature` and `max_tokens`.
- Added generator checkpoint persistence in `src/arka/pipeline/checkpoint.py`:
  - `generator_runs` table
  - `save_generator(...)`
  - `load_generator(...)`
- Updated `src/arka/pipeline/stage_builder.py` to import the new generator module.
- Updated tests to cover:
  - raw response writing
  - same-prompt resume from parquet
  - prompt-change regeneration
  - partial raw-response resume
  - generator checkpoint save/load
  - CLI/integration monkeypatch paths

## Validation
- `just check`
- Result: all checks passed, 105 tests passed.

## Notes
- Partial recovery uses `plan_index`, not only `seed_id`, so repeated seed cycling with `generation_multiplier` resumes correctly.
- If prompt hash changes, stale `raw_responses.jsonl` is discarded and regenerated.
- If parquet is missing but raw responses match the prompt hash, the stage reparses without API calls.
