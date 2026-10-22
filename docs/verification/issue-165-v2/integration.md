# Integration with current main

The ridge scorer was integrated with main `9d21ca4` on 2026-10-21. Main already
contained a probability classifier, so the numeric ridge stage and score now use
`distilled_ridge`. Existing classifier behavior remains covered by its own tests
and installed-CLI recipe.

The archived baseline and improved JSON models loaded unchanged. We reran the
installed pipeline on the same 599 holdout payloads with real MiniLM embeddings.
For each model, every prediction matched the archived prediction exactly. Both
299-record selected payload files were byte-identical to the archived exports.
No labels were collected and no model was refitted during this replay.

Evidence remains under `/tmp/arka-165-current-main-replay`. The archived Parquet
scores use the earlier prototype record schema. Their `scores_json` columns
were read directly for comparison with the current typed records; the original
artifacts were preserved.

`just check` passed lint, formatting and 740 tests, with one optional native
language test skipped. `uv build` produced both distributions. `just verify`
passed all five installed-CLI workflows, example validation and stage previews.
Its evidence is `/tmp/arka-verify-pu9b60yp`. Fixture scores establish behavior,
not quality. The failed primary quality check in the main report still applies.
