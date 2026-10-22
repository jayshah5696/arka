# Source audits and explicit rebuilding

Run `uv run python scripts/verify_recovery_cli.py --output /tmp/arka-recovery-proof`
with a fresh output directory, or use `just verify`.

The helper creates PDF and seed parent runs through the installed CLI. It
combines their saved artifacts into one fixture population, audits both sources,
then explicitly rebuilds the PDF source. It exercises both judge backends.

Confirm these results in each backend's logs, `requests.json`, and artifacts:

- A parent marked with a PII policy is refused before provider requests.
- PDF judgments and replacement requests contain the exact persisted chunk.
- The first replacement fails quality. The second is persisted before an
  injected HTTP 503 interrupts its judgment.
- Resume judges the persisted candidate without repeating generation. A
  completed resume makes no requests and preserves dataset bytes.
- The selected source receives the accepted replacement. The unselected seed
  record remains unchanged, with its known audit failure reported as unresolved.
- `attempts.parquet` records `quality_failed` and `accepted` outcomes.
- A further audit of the rebuilt run resolves its recorded source ancestry.

Inspect the root `result.json`, each backend's recovery report, the revised
dataset, and `interrupted.log`, `resumed.log`, and `completed-resume.log`.
The fixture proves operation behavior, not representative answer quality.
