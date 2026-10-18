# Local verification workflow

Run `just verify` from the repository root. The checked source started at `7790402` with the verification changes recorded in the run's working tree status. Evidence remains at `/tmp/arka-verify-z7wqw6zx/`; temporary files are local to this machine.

## Observed behavior

All five checks passed: CLI version, example validation, example stage previews, cleanup, and decisions. These use deterministic loopback fixtures and no paid calls.

The PDF fixture generated six responses. It stripped the opener from `Certainly! A useful answer about the document.`, removed its clean duplicate, and dropped the repetition, identity claim, and placeholder records. The two exports were `A useful answer about the document.` and `ChatGPT was developed by OpenAI.`. Page provenance spans original pages 1 through 3 across empty page 2.

The taxonomy fixture made eight requests and exported two records covering science and arts. Together the cleanup cases made 14 initial requests. Both resumed with identical dataset bytes and zero new requests.

The decision fixture processed three seeds through nine requests and exported two records. Resume made zero additional requests and preserved dataset bytes. Saved CLI and resume logs, request payloads, manifests, stage artifacts, and datasets remained after exit. A socket connection confirmed that the decision fixture port was closed after helper teardown. A repeated invocation against the same evidence path failed without changing the original summary.

## Validation

- `uv sync --frozen` passed.
- `just check` passed lint and formatting checks. Pytest reported 545 passed and one skipped. The skipped native reasoning-language test needs optional fastText prerequisites.
- `uv build` produced a wheel and source distribution. It emitted the existing uv build-version warning.
- The skill creator's `quick_validate.py` accepted `.agents/skills/verify-arka`.
- Failure-evidence tests first failed because the new runner did not exist, then passed after implementation. They check complete output from a failed subprocess and evidence for a missing executable.

Native language checks, Hugging Face token checks, live provider calls, and representative dataset audits were not run. The feature map names these limits explicitly. Product code and dependencies are unchanged.
