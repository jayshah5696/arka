# Verification skill and PR

## Progress

I’ll add the shared local verification command, update `verify-arka`, and clarify its use in `AGENTS.md`. I’ll use the show-me skill for a concrete verification example in the PR description, then run the checks and create the PR.

The shared command is in place, and the failure-reporting tests pass. It will run config validation, stage previews, cleanup checks, and decision-model checks. Each run gets its own evidence directory and stops at the first failure while preserving the logs.

I’m updating the skill and feature map now. Native language checks and live provider calls will stay outside the default command.

The full check passed: 545 tests passed and one optional native-language test was skipped. The build and `just verify` also passed. The CLI workflows produced two exports per scenario and made no extra requests on resume.

I’m checking that the evidence survives helper shutdown, then I’ll add the input-to-output example to the PR description. `AGENTS.md` now explains when to use the skill and which checks need separate prerequisites.

## Result

Implemented the default local verification command and refreshed the project skill with five feature files. Updated AGENTS.md with scope, triggers, maintenance guidance, and optional-check boundaries. Applied show-me to the PR description with a Mermaid flow and an observed cleanup example. Full checks, build, skill validation, live CLI fixtures, evidence preservation, and existing-directory protection passed. Native/HF and paid provider recipes remain unverified. See the committed verification report for details.

## Final response

Created [PR #219](https://github.com/jayshah5696/arka/pull/219) and linked it to this thread.

Added `just verify`, refreshed `verify-arka` and its five-feature map, and improved the decision helper's saved evidence.

`AGENTS.md` now states when to use the skill, how to run it, and when to update its feature map. It also distinguishes the default local checks from optional native/HF checks and authorized live provider calls.

The PR description uses the show-me skill with a Mermaid workflow and an observed cleanup example.

Validation passed: lint, formatting, 545 tests, build, skill validation, and the complete local CLI workflow. One optional native-language test was skipped. Native/HF recipes and live provider calls were not run.
