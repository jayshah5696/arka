## Why

Add the approved experimental numeric ranking scorer for #165. Users can train a small ridge model from saved rubric judgments, then reuse it to rank records without requesting new judge labels.

Related to #165. This PR does not close it. The fresh holdout's primary quality improvement check failed.

## Changes

- Add the separate `arka-train-scorer` command, deterministic source/lineage/text group splits, and versioned JSON models with SHA-256 pins.
- Add `distilled_ridge` annotation and numeric scores for composite selection. Keep the existing `distilled_quality` probability classifier and source audits intact.
- Support the original combined-text embedding representation and a revised 12-feature representation with training-only standardization.
- Bind models and resume artifacts to embedding settings and privacy policy. Reject unprotected training text before embedding requests and reject stale predictions during selection.
- Add usage documentation, reproducible evaluation scripts, score tables, and an installed-CLI verification recipe.

## Reviewer notes

This branch integrates the previously local prototype with main at `9d21ca4`, including the newer stage, artifact, annotation and privacy contracts. It adds no dependencies. It does not add automatic judging, rejection or routing. Selection remains explicit.

On the fresh 599-record English OpenAssistant holdout, teacher Spearman rose from 0.204 to 0.297 and human Spearman from 0.208 to 0.338. The paired interval for teacher agreement improvement was [-0.021, 0.206], which includes zero. Total cost savings are unproven. See `docs/verification/issue-165-v2/README.md` for coverage, exclusions, costs and limits. These results concern ridge regression, not the existing probability classifier.

## Verification

- `just check`: lint and format passed; 740 tests passed, one optional native-language test skipped.
- `uv build`: source distribution and wheel built successfully.
- `just verify`: readiness, example validation and previews, plus all five installed-CLI workflows passed. Evidence: `/tmp/arka-verify-pu9b60yp`.
- The new CLI recipe trained both artifact versions, scored 30 records, selected three per version, resumed without extra requests, and rejected changed models and private training text before requests.
- Replayed both archived models through the integrated pipeline with real MiniLM embeddings: all 599 predictions per model matched exactly, and both 299-record selected exports were byte-identical. Evidence: `/tmp/arka-165-current-main-replay`.

Local HTTP fixtures establish behavior. The archived public-data evaluation establishes the reported quality estimates and their limits. No new paid judge calls were made for this integration.
