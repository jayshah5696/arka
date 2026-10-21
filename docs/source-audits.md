# Audit answer quality by source

`source_audit` judges a deterministic sample from each source and reports observed failures. It preserves input payloads, existing scores and order. Normal privacy protection still applies. It does not repair answers or rebuild sources. Use the existing [quality repair stage](quality-repair.md) for bounded answer repair.

```yaml
pipeline:
  - type: seed_source
    path: ./seeds.jsonl
  - type: normalize_conversation
  - type: source_audit
    rubric_path: ./rubric.yaml
    min_scores:
      correctness: 3
    sample_size: 20
    sample_seed: 0
```

`sample_size` is required and applies separately to each group. The example value is illustrative. Arka groups records by document ID, then seed file hash. Records without either identifier belong to `ungrouped`. Small groups are sampled in full. A stable hash of the seed, group and record ID selects records independently of input order. Duplicate record IDs stop the audit before judge calls.

Only single-turn conversation records are eligible. Other records and conversations with turns remain in the pipeline and are counted as unsupported. Population and eligible-population counts are reported separately.

## Judgments and source evidence

The stage uses the same rubric validation, evidence resolution and judging code as answer repair. It requires every configured minimum to pass. Set `judge_llm_override` to choose a separate judge, or use `backend: decision` with the run's decision configuration and optional `decision_override`. Decision judgments below `min_confidence`, default 0.5, count as uncertain.

For grounded records, Arka loads `data.parquet` from the configured `evidence_stage`, default `01_source`. The source chunk must match the record's lineage root, document ID and chunk ID. Missing, ambiguous or privacy-rejected evidence is reported as missing evidence. It does not become an answer-quality failure. Imported seed answers without document identifiers are judged without assumed reference evidence.

The audit uses its own judgments. Existing record scores do not prove that the same evidence and effective judge settings were used. Within an audit, resume reuses judgments only when settings, rubric, permitted evidence and input identities still match.

## Read the report

`runs/<run-id>/stages/03_source_audit/stats.json` contains per-source counts for the population, supported records and sample. It separates passed and failed judgments from uncertain judgments, missing evidence and privacy-rejected judgments.

`failure_rate` is failures divided by passed plus failed judgments. It is null when there are no valid judgments. It describes only the observed valid sample. It is not a confidence bound or a validated trigger for replacing an entire source.

`census` means the entire source population was selected. `incomplete` means at least one sampled record has no valid quality outcome. Check both values before interpreting a rate. The stage does not recommend intervention from small samples or rates near a threshold. Sampling policy and intervention thresholds need a separate workload-specific decision.

The report includes historical LLM and decision usage after resume, known costs and a cost-completeness flag. Provider errors and malformed judgments stop the run. They do not become quality failures.

## Resume

`audit.json` stores the selected IDs and each protected judgment. Arka saves a judgment atomically before moving to the next record. A compatible interrupted run reuses completed judgments. A completed audit makes no new requests on resume. Changing the rubric, settings, evidence or incoming records requires a new run ID.

A crash after a provider response but before persistence can repeat a remote request. The audit does not promise exactly-once provider calls. Protected judge reasoning can be retained in the audit artifact under the active privacy policy. That policy's existing limits for filenames and arbitrary metadata still apply.

See [the source audit example](../examples/13-source-audit.yaml). Run `just verify` or `uv run python scripts/verify_quality_cli.py` for deterministic CLI proof of sampling, evidence checks, uncertainty, privacy and interrupted resume.
