# Verification for PRs #184 through #191

Verified on 2026-10-10. Each PR's implementation head passed `just check` and `uv build` in a detached checkout. [heads.json](heads.json) records exact heads, bases, commands, counts, and exit codes. Full local logs remain under `/tmp/arka-verification-20261010-heads/pr-N/`; compact results here are the durable receipts.

| PR | Tests passed | Skipped | Lint, format, build |
| --- | ---: | ---: | --- |
| #184 | 349 | 0 | Pass |
| #185 | 370 | 0 | Pass |
| #186 | 383 | 0 | Pass |
| #187 | 409 | 0 | Pass |
| #188 | 439 | 0 | Pass |
| #189 | 448 | 0 | Pass |
| #190 | 464 | 0 | Pass |
| #191 | 481 | 0 | Pass |

The combined CLI checks ran on the top implementation plus the new verification helpers. These are distinct from each lower branch's test run.

- [Cleanup result](cleanup/result.json): 14 actual SDK requests to a local HTTP fixture, two PDF exports, two taxonomy exports, and zero additional requests on resume. [PDF output](cleanup/pdf/dataset.jsonl) and [taxonomy output](cleanup/taxonomy/dataset.jsonl) show surviving records. Stage stats record drops and coverage.
- [Combined stack result](stack/result.json): 7 actual SDK requests, 1,200 PDF header bytes removed, one reasoning trace stripped, one benchmark leak dropped, two exports, and zero additional requests on resume. [ChatML](stack/dataset.jsonl), [canonical JSONL](stack/jsonl.jsonl), and [Alpaca](stack/alpaca.jsonl) preserve the retained trace as documented.
- [HF result](stack/hf-result.json): datasets loads all three formats. Qwen3 renders one think block, produces 66 tokens, and preserves trace and answer after token decoding. [Rendered text](stack/hf-rendered.txt) shows the training template.
- [Environment](environment.json): implementation head, local language-model digest, provider boundary, and current full-check excerpt. The top working tree also passed lint, formatting, 481 tests, and build after adding the helpers.

Run the reusable [verify-arka skill](../../../.agents/skills/verify-arka/SKILL.md) for fresh proof. It records commands, prerequisites, feature paths, evidence, and cleanup. Output directories must be new. The fixture server closes when the helper exits, and evidence survives cleanup.

The native language model was supplied locally. Fixture responses do not establish production false-positive rates. The 500-record repetition audit and 1,000-record identity audit remain pending. Qwen3 token preservation does not establish training quality or compatibility with every model template. Earlier synthetic performance timing is not rerun evidence in this report.
