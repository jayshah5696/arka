## Why

Quality scoring currently requires direct judge calls for each record. Add a reusable classifier trained from saved judge labels and source audits that report sampled answer quality without filtering records.

Related to #165 and #173. This implements the agreed first version and excludes the targeted repair work already merged in #220.

## Changes

- Add separate `distilled_quality_train` and `distilled_quality_score` stages. Training uses frozen embeddings and logistic regression. Connected sources, lineage roots and identical inputs stay together across training and holdout partitions. The saved JSON model contains numeric coefficients and validated metadata.
- Keep predictions separate from teacher scores. Inference never retrains, supports evaluation against saved labels, and filters only with an explicit probability threshold. Reject incompatible models, missing embeddings and invalid label provenance.
- Add `source_audit` with deterministic sampling, permitted source evidence, LLM and decision judges, per-source reports and persisted judgment progress. Preserve records and distinguish quality failures from uncertainty, missing evidence and privacy rejection.
- Share repair's judging and evidence code with audits. Check external artifacts and ordered inputs before resume. Add examples, documentation, generated configuration schema and a default CLI verification recipe.

## Reviewer notes

The classifier is experimental. Representative factual quality, total cost savings, native Hugging Face inference and live providers remain unverified. Existing embedding integration does not report complete monetary usage, so scorer reports explicitly mark costs incomplete. Training requires existing labeled artifacts from independent groups. No automatic source rebuilding, repeated repair rounds or target refill is included.

Adds scikit-learn for training. Inference uses NumPy and saved coefficients, without loading serialized Python objects.

## Verification

- `just check`: 721 tests passed and one optional native language test skipped.
- `uv build`: source distribution and wheel built.
- `just verify`: all four deterministic CLI workflows passed. The new quality recipe exercises training from both judge backends, scoring, pre-filtering before direct judging, evaluation, PDF evidence, missing evidence, privacy rejection, uncertainty, incompatible artifacts and interrupted resume. Completed resumes make no new requests.
- The quality recipe uses synthetic embeddings and loopback providers. These checks establish behavior, not representative model quality or cost savings.
