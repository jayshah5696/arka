## Why

The targeted repair workflow from #173 ignored privacy rejection metadata on LLM judgments. A passing judgment with blocked PII in its reasoning could accept an answer. A failing judgment could trigger another provider call using feedback that the policy had rejected.

## Changes

- Persist each judgment's privacy drop reason and stop that record after a rejected initial or replacement judgment.
- Retain protected feedback and usage for resume. Exclude rejected scores from quality outcomes and preserve the privacy drop reason.
- Use repair identity version 2. Version 1 repair runs require a new run ID because they did not retain judgment privacy outcomes.
- Add four focused test cases and extend the installed CLI proof with three judgment privacy scenarios and legacy identity rejection. Update the configuration guide and verification feature map.

## Reviewer notes

This fixes the existing targeted answer-repair scope. Source audits, whole-source rebuilding and target refill remain open in #173. Existing run artifacts remain available for inspection. The CLI rejects version 1 repair identities without provider requests or changes to upstream artifacts.

## Verification

- Red/green: all four new test cases failed before the fix. All 29 repair tests pass afterward.
- `just check`: lint and formatting passed; 676 tests passed and one optional native-language test skipped.
- `uv build`: source archive and wheel built successfully.
- `just verify`: all default workflows passed. Repair exercised ten scenarios with 32 local HTTP requests. Judgment privacy scenarios exported no records, preserved protected feedback and usage, and made no new requests on completed resume.
- `git diff --check`: passed.

Local evidence is in `/tmp/arka-verify-iyvpvkzt` and `/tmp/arka-repair-66w7p7nc`. No paid provider calls were made. Optional native/Hugging Face checks and representative model-quality evaluation were not run.
