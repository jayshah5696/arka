## Why

Dataset experiments need explicit model bindings and a clear boundary between experiment changes and execution controls. Add version 2 configuration so engineers can start from a generated preset, edit the complete pipeline, and resume unfinished work with different concurrency or transport settings. Changes to experiment settings or input contents require a new run ID.

## Changes

- Add `arka-init` with SFT and PDF-SFT presets that generate complete editable YAML without reading credentials or calling providers.
- Define named models inside each experiment file, unique stage IDs, typed stage configs, and separate execution controls. Reuse the existing stage registry and context-owned clients.
- Share a dependency-aware plan builder between validation, preview, and execution. Fingerprint configured sources, rubrics, taxonomies, evaluation and training artifacts, and local model files.
- Check resume compatibility before changing run metadata or opening checkpoints. Preserve the original plan and resolved config, record operational settings for each accepted invocation, and reuse completed stages and generator responses.
- Parse YAML before environment substitution, reject duplicate YAML keys and unsupported versions, and regenerate the schema for both formats.
- Add configuration documentation and an installed-CLI verification recipe to the default verification workflow.

## Reviewer notes

Version 1 remains supported with its existing resume behavior. The new contract applies to version 2. Historical runs without dependency plans need new IDs when moving to version 2. Version 2 stages select one named LLM connection; inline connection overrides are rejected. Repair stages that need distinct generation and judge connections can continue using version 1. Decision providers remain in the existing separate `decisions` section.

Presets establish pipeline structure, not dataset quality. Hosted model changes cannot be detected from an unchanged model ID. Run comparison, automatic migration, scheduling, and concurrent-process guarantees are outside this change. No new dependencies were added.

## Verification

- Red/green tests preceded logic changes.
- `just check`: 791 passed, one existing optional native-language test skipped; lint and format checks passed.
- `uv build`: source distribution and wheel built.
- `just verify`: all seven installed-CLI workflows passed, plus example validation and previews. Evidence: `/tmp/arka-verify-m8lyaa8s`.
- The experiment recipe generated and edited a config, verified named-model dispatch, preserved one saved response through interruption, and made exactly two calls for pending records on resume. Completed resume made zero calls. Changed source contents and model settings failed without changing run files or SQLite state.
- Re-ran the experiment recipe after the final empty-reference and YAML error handling fixes. Evidence: `/tmp/arka-experiment-n0713p9x`.
- `git diff --check` passed. No paid provider, optional native/Hugging Face, or representative dataset-quality checks were run.
