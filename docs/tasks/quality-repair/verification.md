# Quality repair verification

Verified on October 18, 2026, in the working tree based on `dcd3088d93e6e56824699b2448db114dbef35d71`. Product changes were pending when the verification helper recorded that revision.

The verified product files were committed as `e914859`. The verification workflow was committed as `49f69f9`. No product code changed after the final proof. The change is published in [PR #220](https://github.com/jayshah5696/arka/pull/220).

## Commands and results

- The initial `uv run pytest tests/unit/test_quality_repair.py --no-cov -q` failed during collection because the new config and stage did not yet exist. The failure cases and 21 focused checks were written before product code.
- The focused checks passed after implementation. Corrections to the test fixture used the existing PII configuration fields.
- `just check` passed lint, format checks and 668 tests. One optional native-language integration test skipped because its model prerequisite was absent.
- `uv build` produced the source archive and wheel. The build emitted the existing uv-build version warning and completed successfully.
- `just verify` passed CLI readiness, example validation and previews, cleanup, decisions and repair.
- `git diff --check` passed.

The final default verification evidence is `/tmp/arka-verify-pcvuyks7/`. Its root `result.json` reports all six command groups passed. `cleanup/result.json`, `decisions/result.json` and `repair/result.json` remain available with logs and outputs. The evidence directory is local and is not part of the Git repository.

## Repair observations

The repair recipe used 25 chat-completion requests and two decision requests across seven scenarios. Completed resumes made no additional requests and preserved export bytes.

The bounded scenario judged six input records. One passed initially. Three received a repair attempt; two replacements passed the second judgment. The repair stage returned three records, and downstream exact deduplication removed one replacement that duplicated a passing original. Two records reached export. The rejected records had separate exhaustion, excluded-dimension and repair-limit reasons.

The fixture reported a synthetic total cost of 0.12 and incremental repair cost of 0.06 for that scenario. These values prove accounting only. They are not provider charges or an estimate of production savings. The per-source report retained counts for all six inputs and five initial failures.

The PDF scenario supplied the exact permitted source chunk to generation, both judgments and repair. The missing-evidence scenario made only the ordinary generation request, then rejected the record without judging or repairing it. The decision scenario used two decision judgments and one repair call; its earlier panel sent no repair questions.

The privacy scenario rejected generated PII, retained protected attempt artifacts and usage, and made no second judgment. The interrupted scenario persisted generation before a fixture provider failure. Resume made one judgment request and exported the replacement without repeating generation.

Changing rubric contents or incoming stage artifacts caused completed-stage resume to fail before new provider requests. The saved logs include `changed-rubric.log`, `changed-input.log` and `resume-after-interruption.log`.

An earlier repair-helper run failed because its fixture read the decision wire field as `state` instead of the existing `input` field. The fixture was corrected. The fresh final default recipe above passed. Partial evidence remains at `/tmp/arka-repair-goy6mpow/`.

## Limits

No paid providers were called. Optional native-language and Hugging Face recipes were not run. Local fixture judgments do not establish answer quality, representative repair success rates or production cost savings. An independent quality review and comparison with the zero-repair baseline remain future evaluation work.
