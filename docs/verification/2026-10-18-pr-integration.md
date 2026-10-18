# PR integration verification

Combined PRs #217, #218 and #219 in an isolated checkout before merging.

The integration keeps the shared generator engine, labeling evaluator, bound configuration, typed stage results and single artifact writes from #217. It applies #218's privacy guards at those shared boundaries, carries privacy summaries through generator checkpoints, and combines privacy and decision drop counts. The runner retains both resume state and the shared privacy policy. The agent rules include the architecture rules and the local verification workflow from #219.

The new native response cache could persist provider output before the privacy wrapper redacted it. A failing integration test reproduced an email in SQLite. Protected runs now disable that cache. The test covers text and malformed structured responses. Protected generator and completed-stage checkpoints remain available. Interrupted judge or Evol-Instruct stages may repeat requests.

An existing integration assertion required raw checkpoint rows in plan order. The shared engine deliberately flushes completed slots immediately. Updated that assertion to check complete slot coverage, ordered loading, identical resumed IDs and payloads, and no new provider calls.

Observed checks:

- `just check` passed with 641 tests passed and one optional native-language test skipped. Lint and formatting passed. Total coverage was 91%.
- `uv build` built the source package and wheel.
- `just verify --output /tmp/arka-merge-20261018-proof` passed CLI readiness, all nine example validations and previews, PDF cleanup, taxonomy selection, decision scoring, exports and resume.
- Cleanup used 14 loopback provider calls and exported two PDF records and two taxonomy records. Resume made no additional calls.
- Decision scoring used nine loopback provider calls and exported two records. Resume preserved dataset bytes and made no additional calls.
- `uv run python scripts/verify_pii_cli.py --output /tmp/arka-merge-20261018-pii` passed all five regex cases, including PDF, malformed structured output, Evol-Instruct, transform and taxonomy generation. It used seven loopback calls. All resumes made no additional calls.

The CLI evidence records combined revision `4f58ff7e6421b89dd89891beee1d8b0b80330aeb`. The final integration-test correction and cache documentation do not change product behavior. Full logs remain in `/tmp/arka-merge-final-check.log`, `/tmp/arka-merge-build.log`, and the two evidence directories above. No paid provider calls were made. Native GLiNER inference, native language and Hugging Face export checks were not repeated.
