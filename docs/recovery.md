# Audit sources and rebuild failed answers

Use `arka-audit` to inspect conversation records from an existing Arka run.
Use `arka-rebuild` to judge every record in explicitly selected sources and
replace failed answers. Both commands accept a separate recovery configuration.

Copy [the example configuration](examples/recovery.yaml) and set its judge,
rubric, and replacement model. Paths in the configuration are relative to its
directory. The project root defaults to that directory. Use `--project-root` to
select another project containing `runs/`.

```sh
uv run arka-audit --config recovery.yaml --parent-run parent-run --run-id audit-1
```

The audit defaults to the parent's final saved stage. Use `--stage STAGE_NAME`
to inspect an earlier stage's surviving conversation records. The parent
manifest lists stage names. Dataset exports alone are insufficient because
they omit the source metadata and lineage needed for recovery.

The audit takes a repeatable sample of up to `sample_size` records from each
source. It judges every record in smaller sources. The default sample size is
200, a configurable starting value. Sampled failure rates describe those
samples. They do not identify every failed answer in a source.

Select source keys printed by the audit. Seed keys contain the original seed
file hash. PDF keys contain the original PDF file hash, resolved through saved
source records. Documents with the same filename remain separate when their
hashes differ. A seed file is one source, even if its records cover many topics.
There is no required failure-rate threshold and no automatic source selection.

```sh
uv run arka-rebuild --config recovery.yaml --audit-run audit-1 \
  --source seed:FILE_HASH --run-id rebuild-1
```

Repeat `--source` to select several sources. The rebuild uses the audit's input
population and judge configuration. It reuses completed sampled judgments and
judges all other records in the selected sources. It then replaces only failed
answers, preserving their instructions and conversation context.

Each replacement receives the original instruction, source evidence where
available, and the previous answer with its assessment feedback. Decision
judges provide scores rather than written criticism. Each new answer is judged
against the same rubric and pass threshold. Decision score levels are mapped
to the rubric's scale, including scales that start above zero.

For a decision judge, replace the example's `judge` block with this form:

```yaml
judge:
  backend: decision
  rubric_path: rubrics/sft_quality.yaml
  min_overall_score: 3.0
  decision:
    provider: openai
    model: YOUR_DECISION_MODEL
    api_key: ${DECISION_API_KEY}
    base_url: https://YOUR_DECISION_ENDPOINT/v1
    max_state_chars: 24000
```

The endpoint must support Arka's existing decision protocol. The replacement
model remains an LLM configured under `replacement.llm`.

The required positive `replacement.max_attempts` limits attempts per failed
answer. An unchanged or empty answer consumes an attempt. By default,
`replacement.exact_dedup` also rejects instruction and answer pairs identical
to retained records or earlier accepted replacements. Set it to `false` to
disable that comparison. This comparison ignores reasoning and other context.
Near deduplication and the parent's other filters are not rerun by these
commands. Include the required quality criteria in the recovery rubric.

For PDF records, the commands require one exact persisted ancestor chunk.
Both judges and the replacement model receive that chunk. Missing or ambiguous
evidence stops the operation before provider use. The commands read saved
artifacts, not raw input files. A decision state larger than
`judge.decision.max_state_chars` stops the operation instead of truncating its
evidence. Increase that setting and create a new audit if needed. Taxonomy-only
generation without a seed file or PDF source is outside this version's scope.

Standalone recovery currently refuses parent runs with an enabled `pii` policy.
Use the existing `source_audit` and `quality_repair` pipeline stages for those
runs. Recovery checks the saved parent configuration before provider calls,
including configurations of recorded ancestor runs.

## Revised dataset and reports

The rebuild exports `runs/rebuild-1/dataset.jsonl`. It preserves input order,
with these changes:

- Accepted replacements take their original records' positions.
- Answers that still fail after their attempt limit are omitted.
- Passing answers and unselected sources remain unchanged.

Known failed audit samples from unselected sources remain in the dataset.
The report counts them as unresolved. This count covers known sampled failures,
not an estimate of all failures in unselected sources.

Accepted replacements receive new identities, lineage links to their originals,
fresh scores, and new reasoning where the model supplies it. Old answer scores,
including distilled predictions, are cleared. Task taxonomy metadata remains.

Each operation saves these files:

| Artifact | Contents |
| --- | --- |
| `operation.json` | Effective settings, input fingerprint, rubric, and parent references. API keys are excluded. |
| `recovery.sqlite3` | Completed judgments, generated candidates, usage, and attempt outcomes. |
| `report/recovery_report.json` | Sample counts and rates by source, assessment and replacement counts, unresolved failures, tokens, and known costs. |
| `manifest.json` | Operation status, parent references, and exported dataset location. |

A rebuild also saves `stages/rebuild/data.parquet`, `dropped.parquet`, and
`attempts.parquet`. These contain the revised records, exhausted originals,
and generated attempts with their assessment scores and outcomes. The journal
retains completed work even when a later provider call fails.

Reports count completed client calls. HTTP retries, fallback requests within a
client call, and failed calls are not counted. Usage and costs cover persisted
successful responses. `cost_complete` indicates whether all counted calls had
reported costs. The attempt limit is not a prepaid monetary budget.

## Resume

```sh
uv run arka-rebuild --config recovery.yaml --audit-run audit-1 \
  --source seed:FILE_HASH --run-id rebuild-1 --resume
```

Use the same run ID and source selection. Audit supports `--resume` too.
Provider errors stop an operation. They do not become low-quality judgments or
exhausted records. Resume reuses completed judgments and saved candidates,
including a candidate whose assessment was interrupted.

Input artifacts, evidence, rubric, judge configuration, replacement settings,
and source selection must remain compatible. Changed inputs require a new
operation. API keys can rotate because credentials are not part of persisted
identity. Concurrent writers to the same operation are rejected.

A crash after a provider responds but before its response is committed can
repeat the remote call. Completed persisted work is reused. A completed rebuild
can itself be audited while its recorded ancestor artifacts remain available.

The configuration schema is [recovery.schema.json](../recovery.schema.json).
Run `just schema` to regenerate it with the pipeline configuration schema.
