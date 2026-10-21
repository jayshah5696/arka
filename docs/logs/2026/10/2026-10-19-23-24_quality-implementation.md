# Distilled scoring and source audit implementation

## Scope

Implemented the approved first version for #165 and the source-audit extension of #173. The repository had no open PRs when scope was checked. Targeted answer repair from merged PR #220 was excluded from new product scope. Its judging and evidence code was shared with audits and its existing verification recipe was retained.

## Assistant progress messages

I’ll implement the scorer first, then add the source audit stage. I’ll use the verification skills to check both through the CLI, with tests written before each behavior change.

The scorer will train from judged Parquet artifacts in a separate run. This lets users reuse labels without making new judge calls during training. The training split will keep records from the same source or lineage together.

For audits, I’ll reuse the repair stage’s judging and evidence checks. The audit will preserve records and report uncertain or incomplete results without triggering regeneration.

The scorer’s focused tests pass. Training keeps shared sources, lineage roots, and identical text together across the train/holdout split. Inference preserves teacher scores and filters only when a threshold is configured.

Next I’ll add the source audit stage and CLI fixtures. Cost savings and model quality will still need measurement on representative data.

The audit tests pass, including deterministic sampling and interrupted resume. Audits preserve the incoming records and report missing evidence separately from quality failures.

I also checked decision-judge labels for training. The scorer now accepts valid decision labels and rejects low-confidence labels before requesting embeddings. Next I’ll exercise training, filtering, audits, and resume through the installed CLI.

The CLI fixtures passed for training from LLM and decision labels, annotation, filtering before direct judging, and evaluation on a separate source. Completed resumes made no new requests.

The audit fixtures also passed for PDF evidence, missing evidence, uncertain judgments, privacy rejection, and interrupted resume. I’m adding the usage docs and examples, then I’ll run the full checks and default verification workflow.

The full checks and default CLI verification passed. Final review caught one resume detail: an audit must reject reordered upstream records, so it cannot reuse output in the old order. I added a failing test and fixed that check.

I’m running the final checks on that fix now. The docs mark real-data quality, total cost savings, and native embedding behavior as unverified.

## Implementation and evidence

- Added registered training, scoring and source audit stages, typed configuration and a generated schema.
- Training consumes existing labeled Parquet files. It uses a frozen embedding representation and scikit-learn logistic regression, then saves validated numeric coefficients and compatible metadata. Related sources, lineage roots and identical text stay together across partitions.
- Scoring preserves teacher scores and stores predictions separately. Filtering requires an explicit threshold. Offline labeled evaluation reports teacher agreement, useful-record recall, false rejections and overlap with the development population.
- Audits select deterministic per-source samples, preserve records and checkpoint judgments. Missing evidence, uncertainty and privacy rejection are distinct from quality failure. Resume validates settings, external dependencies and input order.
- Wrote failure cases and tests before implementation. Additional red/green fixes covered decision-label confidence, stale content hashes and reordered audit inputs. Existing example checks exposed the source-less training case, which required a small validator update.
- Added examples 11 through 13 and usage guides. Updated the verification skill, feature map and default helper. The new CLI recipe uses only loopback providers and saves request logs, artifacts and command timings.
- Final `just check` passed with 721 passed tests and one optional native language test skipped. `uv build` produced a source distribution and wheel. The build printed the existing uv-build version-range warning.
- Final `just verify` passed all four workflows. Evidence is retained in `/tmp/arka-verify-ngff_aom` at revision `d656604b4efc4688661bb38b64abed7b32c088d6` with the working changes. All child result files report passed. The quality recipe made 49 local requests and retained the model, dataset exports and resume logs. Earlier complete evidence at `/tmp/arka-verify-maubl9d8` passed before the input-order fix.
- Representative factual quality, total cost savings, native Hugging Face behavior and live providers remain unverified. No paid calls, issue changes, commits or PRs were made.

## Final response

Implemented the scorer and source audit stages.

- `distilled_quality_train` trains a reusable linear classifier from judged Parquet artifacts. Training and holdout partitions keep related sources, lineage and identical text together.
- `distilled_quality_score` loads the saved model without retraining. It preserves teacher scores, supports labeled evaluation, and filters only when `min_probability` is configured.
- `source_audit` samples deterministically by source. It preserves records, checks grounding evidence, and reports uncertainty and privacy rejection separately from quality failures. Interrupted runs reuse persisted judgments.

Added configuration schemas, examples, usage docs and CLI verification. The guides are [distilled quality scoring](/home/jshah/.t3/worktrees/arka/t3-4c56972b/docs/distilled-quality.md) and [source audits](/home/jshah/.t3/worktrees/arka/t3-4c56972b/docs/source-audits.md).

Validation passed:

- `just check`: 721 passed, one optional native language test skipped.
- `uv build`: source distribution and wheel built.
- `just verify`: all four CLI workflows passed, including the existing repair workflow. [Verification evidence](/tmp/arka-verify-ngff_aom/result.json) is retained.

The scorer remains experimental. Representative quality and total cost savings still need measurement on real data. Native Hugging Face inference and live providers were not exercised. No commits or PRs were created.
