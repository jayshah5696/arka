# PII verification

Verified on October 17, 2026 in the managed PII worktree, based on main at `779040204b0f392465b7e6240bf9d9200816dc5d`. The PR description records the final checked commit.

## Results

| Command | Result |
| --- | --- |
| `just check` | Lint and formatting passed. 614 tests passed and one unrelated native-language integration test skipped. |
| `uv build` | Source distribution and wheel built. |
| `just schema` | Regenerated the PII configuration schema. |
| `just validate-examples` | All existing examples validated. |
| `just dry-run-examples` | All nine examples passed. |
| `uv run python scripts/verify_cleanup_cli.py --output /tmp/arka-pr218-fixes-cleanup-20261017` | Existing PDF, taxonomy, cleanup and completed-run resume paths passed. 14 loopback provider calls and zero extra resume calls. |
| `uv run --no-sync python scripts/verify_pii_cli.py --output /tmp/arka-pr218-fixes-pii-20261017 --model-path /tmp/arka-gliner2-pii-model` | All eight PII cases passed with real offline GLiNER2 inference. 11 loopback provider calls and zero extra calls on resume. |
| `git diff --check` | Passed. |

The native run used gliner2 2.1.0, PyTorch 2.14.1+cpu, Transformers 5.17.0 and huggingface-hub 1.8.0. The optional runtime was installed with uv. The CPU PyTorch wheel came from its official CPU index. The provisioned checkpoint revision was `1cb4166094dc58fa8d836429f060d6c95f62b495`.

The loader emitted upstream model-type and TorchScript deprecation warnings. They did not prevent inference or CLI completion. No paid provider call was made.

## What the fixtures prove

The [committed result](evidence/cli-result.json) covers PDF protection before its decision request, generated raw output, malformed escaped JSON, Evol-Instruct's intermediate request, transform and taxonomy generation, real GLiNER PDF and Evol runs, and a 4,500-character input with repeated names and emails at both ends. The CLI runs inspect outgoing HTTP bodies, JSON artifacts, Parquet, logs and SQLite state. Each completed run resumes with an identical export and no extra provider call. PDF source coordinates continue to refer to original extracted text.

Observed fixture input:

```text
Contact John Smith at generated@example.com. He lives at 123 Main Street, London.
```

Observed output from the [real GLiNER export](evidence/gliner-dataset.jsonl):

```text
Contact [person] at [email]. He lives at [address].
```

The initial feature added 53 focused cases. The review fixes add 18 more, for 71 total. They were written before their corresponding logic changes. Besides the CLI paths, they cover blocked categories, ratio rejection with retained usage, Unicode byte denominators, overlap union, repeated offsets, unsupported coverage, absent local models, invalid spans, safe detector errors, safe upstream exception logs, stable placeholders, role preservation, malformed schemas and incompatible resume policies.

## Review regressions

The review of `c2b12fa` exposed three gaps. The new `tests/integration/test_pii_recovery.py` covers:

- Provider and malformed-response failures through the native client in labeling, double critic and Elo stages. Their existing drop or skip behavior now applies with PII enabled.
- Payload-free client error codes for completion, structured completion and response scoring. Explicit generator rejection mode retains usage without changing the original judge wrapper.
- A source stage that fails after protection, then retries with and without resume. Counts and rows reflect one attempt. A later completed resume leaves both PII aggregation files byte-identical.
- Taxonomy rejection for blocked entities, ratio limits and malformed output, with decision checks both enabled and disabled. Protected drop rows match the reported counts and reasons, and generation costs remain counted.

The red run recorded 15 failures and two passing controls. The additional generator-mode test then failed before its API existed. After the fixes, the focused recovery, boundary and taxonomy set passed all 39 cases. Full local evidence is at `/tmp/arka-pr218-recovery-red-short.log`, `/tmp/arka-pr218-fixes-cleanup-20261017/result.json` and `/tmp/arka-pr218-fixes-pii-20261017/result.json`.

## Limits

These are integration fixtures with known synthetic values. They do not measure production recall, precision, CPU throughput or data utility. Declared multilingual coverage comes from the checkpoint documentation; this run does not validate every category in all seven languages. Representative domain evaluation remains necessary before selecting thresholds for a production dataset.

The original seed files and PDFs are deliberately outside the output inspection contract. User configuration, filenames and arbitrary metadata also remain outside this text policy. Original inputs are retained in the local evidence directory so the fixture can be reproduced.

The full reasoning and Hugging Face export workflow was not repeated for this PII change. The existing native-language test remains skipped because its optional prerequisite is absent. Existing Pydantic and legacy-configuration warnings remain.
