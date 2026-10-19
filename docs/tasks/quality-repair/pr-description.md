## Why

Quality filtering currently discards failed answers. Add the targeted answer-repair scope discussed in #173 so users can attempt one replacement while preserving the original instruction and system message.

## Changes

- Add a registered `quality_repair` stage with explicit rubric minimums, eligible dimensions and a required repair limit. Passing originals continue unchanged; replacements must pass a second judgment before downstream checks.
- Support LLM and decision judges. Resolve exact permitted PDF chunks, apply the existing PII policy, clear stale response scores and reasoning, and link replacement records to originals.
- Persist each judgment and generation independently. Validate settings, rubric, evidence and incoming artifacts before resume, including completed stages. Exclude repair judgments from decision-panel prefetch because their evidence differs.
- Report per-source counts, rejection reasons, recorded usage and known incremental repair cost. Add configuration documentation and an example, and regenerate the schema.
- Add 25 focused tests and a local installed-CLI recipe covering seven scenarios. Include that recipe in `just verify` and update the verification skill and feature map.

## Reviewer notes

This implements single-turn targeted repair with one attempt per admitted record. Acceptance counts precede downstream validators and deduplication. A crash before response persistence can repeat a remote call. The CLI fixtures establish behavior and accounting; representative quality gains and production cost savings remain unmeasured.

## Verification

- `just check`: lint and formatting passed; 672 tests passed, one optional native-language check skipped.
- `uv build`: source archive and wheel built successfully.
- `just verify`: all default checks passed, including seven repair scenarios using 27 local HTTP requests. Completed resumes made zero new requests. Interrupted resume reused persisted generation and made one judgment request.
- `git diff --check`: passed.

See `docs/tasks/quality-repair/verification.md` for observations and local evidence paths. No paid provider calls were made.
