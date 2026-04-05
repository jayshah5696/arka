# Validation matrix and OSS-style docs update

Completed a current-scope "go all in" validation pass for the implemented Tier A SFT slice.

## What I added

### Automated supported-options matrix
Added `tests/integration/test_supported_options_matrix.py` covering the currently implemented combinations:

- seed input formats
  - JSONL
  - CSV
- executor modes accepted by config
  - `threadpool`
  - `realtime`
  - `provider_batch`
- output formats
  - `jsonl`
  - `chatml`
  - `alpaca`
- dedup combinations
  - off
  - exact only
  - near only
  - exact + near
- quality-gated generation checks
  - prompt-based generation
  - single-judge labeling filter
  - run report artifacts
  - formatter-aware dataset output

### OSS-style documentation
Added `docs/validation-matrix.md` documenting:

- what is actually supported now
- what is covered by automated tests
- the current quality bar for generation runs
- release/checklist workflow
- what remains out of scope

### Developer workflow
Added:

- `just matrix`

Updated docs entry points:

- `README.md`
- `docs/config-examples.md`

## Validation run results
Executed successfully:

- `just matrix`
- `just check`

Result:
- `140 passed`
- overall reported coverage: `92%`

## Important framing
This work defines and verifies the **currently implemented** option matrix, not speculative future spec surface.

That means:
- `labeling_engine.mode: multi` is still not implemented end-to-end
- near dedup is still MinHash-only
- non-`threadpool` executor modes are config-compatible surfaces today, not distinct execution backends
- diversity remains best-effort

## Outcome
The repo now has:
- executable validation for supported config/runtime combinations
- explicit generation quality checks for the current slice
- OSS-style docs that explain what is supported, how to validate it, and what quality means before training
