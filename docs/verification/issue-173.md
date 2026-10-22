# Issue 173 verification

Verified the local implementation on October 21, 2026, then prepared an isolated
branch from current main at `ae3c146` for merging. The original checkout and its
unrelated changes were preserved.

## Checks

| Command | Result |
| --- | --- |
| `uv run pytest --no-cov -q tests/unit/test_recovery.py tests/integration/test_recovery_cli.py` | 30 passed during focused development. |
| `just check` | On current main plus the recovery changes, lint and formatting passed. 770 tests passed and one existing optional test skipped. |
| `just schema` | Generated both pipeline and recovery schemas. Recovery schema matches its Pydantic configuration model. |
| `uv build` | Built the source distribution and wheel. Wheel contains the recovery package and both console entry points. |
| `uv run python scripts/verify_recovery_cli.py --output /tmp/arka-173-proof-20261021-final` | Both judge backends passed through installed CLI subprocesses and actual loopback HTTP requests. |
| `git diff --check` | Passed. |

The focused tests were written first and failed because the recovery module did
not exist. Existing tests were then refined before corrections for cleaned
conversation payloads and the PDF source stage's text-only content hash.

## Direct CLI evidence

The helper first created PDF and seed parent runs with the ordinary installed
`arka` command. It assembled their saved records into one fixture parent
population, then audited that population. The audit identified both source
groups and judged one record from each.

The helper selected the PDF source for rebuilding. Its first replacement failed
the quality rubric. Its second replacement was persisted, but its assessment
received HTTP 503. The operation stopped. Resume assessed that saved candidate
and accepted it without repeating generation. A further completed resume issued
no provider requests and preserved the dataset bytes.

For each judge backend:

- Nine HTTP requests occurred before the second audit, including the two parent
  generation calls and one deliberately failed assessment call.
- Two replacement candidates were generated.
- Six assessment or replacement requests contained the exact saved PDF chunk.
- The revised dataset contained the accepted PDF replacement and the unchanged
  seed record, in original order.
- The seed record's failed sampled judgment remained visible as one unresolved
  failure from an unselected source.
- `attempts.parquet` contained one `quality_failed` attempt and one `accepted`
  attempt.
- A new audit of the rebuilt run resolved its ancestry and completed.

These fixtures verify application behavior. They do not measure model quality
or production recovery yield.

## Artifacts

- Final check log: `/tmp/arka-173-check-final-20261021.log`.
- Direct CLI results: `/tmp/arka-173-proof-20261021-final/result.json`.
- Each backend's directory retains parent artifacts, recovery configuration,
  request receipts, interrupted and resumed command logs, reports, dataset,
  journal, and attempt artifacts.
- Built distributions: `dist/arka-0.1.0.tar.gz` and
  `dist/arka-0.1.0-py3-none-any.whl`.

The first direct PDF check exposed the chunk-hash mismatch. Its failed evidence
remains in `/tmp/arka-173-proof-20261021`. The final check used a fresh directory
after correcting the implementation and the unit fixture.

## Integration with current main

The isolated branch also preserves task metadata through the current annotation
API. Its recovery CLI fixture verifies that a parent run with an enabled PII
policy is refused before provider calls. The commands do not bypass the existing
protected pipeline stages.

Final integration evidence is saved in `/tmp/arka-173-main-check.log` and
`/tmp/arka-173-main-verify`. The default `just verify` workflow now includes the
recovery helper alongside the existing ridge, cleanup, decisions, repair, and
quality helpers.
