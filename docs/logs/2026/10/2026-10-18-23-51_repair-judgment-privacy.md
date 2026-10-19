# Repair judgment privacy fix

## User request

Work on issue 173, fix gaps and create a PR.

## Progress updates

I'll review the existing repair workflow against issue #173, fix concrete gaps, and create a PR. I'll keep the changes within the approved targeted-repair scope.

I found a privacy-policy gap in repair judgments. The model client marks judge output as rejected when its reasoning contains blocked PII, but `quality_repair` ignores that result. It can then accept the answer or request a repair using that judgment.

I'll add failing tests for rejection in the initial and final judgments, then fix the workflow to stop that record and retain its usage. I'm also applying the Arka verification and PR-description skills.

The four new tests failed as expected before the fix. All 29 repair tests now pass.

The fix stops a record when either judgment is rejected by the privacy policy. It saves the protected judgment and usage, and excludes the rejected scores from quality results. Older repair checkpoints lack the privacy outcome, so they will require a new run ID. I'll verify both behaviors through the CLI next.

The CLI repair recipe passed all ten scenarios with 32 local HTTP requests. Privacy-rejected initial judgments stopped after one request. A rejected replacement judgment produced no export, and completed resumes made no new requests.

`just check` passed with 676 tests and one optional skip. The package build and `just verify` also passed. I'm reviewing the final diff and preparing the PR, including the checkpoint compatibility change.

## Evidence

Checks ran on base revision `ea097afbe6f405eda7d734cd5f2009e45760aba2` with the product and verification edits present.

- `/tmp/arka-173-check.log` contains the full lint, formatting and test results.
- `/tmp/arka-verify-iyvpvkzt` contains the default CLI workflows and results.
- `/tmp/arka-repair-66w7p7nc` contains the standalone repair proof.
- `uv build` produced the source archive and wheel.

The fix preserves the policy rejection in attempt JSON and dropped records. Counts exclude rejected judgments from initial quality passes and failures. Usage includes the rejected calls. Version 1 repair checkpoints require a new run ID. Broader source-audit and refill behavior remains undecided in issue 173.
