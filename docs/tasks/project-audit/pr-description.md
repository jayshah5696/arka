## Why

Project review found six concrete bugs in run paths, privacy filtering,
configuration identity, resume reporting, seed ingestion, and config errors.
Fix these issues and preserve the remaining findings for review.

## Changes

- Share run identifier and path containment checks between the pipeline and recovery.
- Check system text, reasoning, and saved turns for canary phrases.
- Compute config hashes from current values after mutations or copies.
- Preserve original input counts in resumed stage reports.
- Hash CSV and JSONL seed files once per input instead of once per record.
- Return config validation errors for numeric and boolean YAML roots.
- Add tests written before the fixes, update stale project docs, and document 13 remaining review items in `docs/project-review-2026-11-09.md`.

## Reviewer notes

Run IDs now follow the same filename restrictions as recovery commands and reject paths that resolve outside the runs directory. Unchanged configs retain their existing hash algorithm and serialization.

Broader resume compatibility, downstream checkpoint invalidation, export failure state, and embedding failure policy remain review items. This PR does not resolve those contracts. No performance speedup was measured.

## Verification

- `just check`: 787 passed, 1 optional native language test skipped; lint and format checks passed.
- `uv build`: wheel and source distribution built successfully.
- `just verify`: all six local installed CLI workflows passed, including recovery, ridge scoring, cleanup, decisions, repair, and distilled quality.
- Both JSON schemas match their models; `git diff --check` passed.

Local CLI evidence is retained at `/tmp/arka-verify-0rom3li9/result.json` on the audit machine. No paid provider calls were made. Optional native integrations and production model quality remain unverified.
