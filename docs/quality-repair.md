# Repair failed answers

Add `quality_repair` where you would otherwise judge and drop low-quality answers. It judges every supported input, keeps passing originals, attempts one replacement for eligible failures, and judges each replacement with the same rubric and source evidence. It preserves the instruction and system message. It clears reasoning when replacing an answer.

```yaml
pipeline:
  - type: seed_source
    path: ./seeds.jsonl
  - type: normalize_conversation
  - type: quality_repair
    rubric_path: ./rubric.yaml
    min_scores:
      correctness: 3
      instruction_clarity: 3
    repair_dimensions: [correctness]
    max_repairs: 100
    temperature: 0.2
    max_tokens: 512
  - type: exact
```

Use names and thresholds from your rubric. Arka checks that all threshold names exist and thresholds fit their dimension scales before calling a provider. `repair_dimensions` must be a subset of the threshold names. A record passes only when every configured minimum is met. If any failed dimension is excluded from `repair_dimensions`, Arka rejects the record without attempting repair. Dimensions without configured minimums are scored but do not determine acceptance.

`max_repairs` is required. It limits records admitted to repair, in input order. A value of zero provides a judge-and-drop baseline with the same acceptance rules. It does not limit initial judgments. Each admitted record can require one generation call and one additional judgment. Provider transport retries and calls repeated after an unpersisted response are separate from this logical attempt limit. There is no monetary spending cap or automatic target refill.

The default judge is the run's LLM. Set `judge_llm_override` to change its model or endpoint. Set `llm_override` to change the repair model or endpoint. Repair uses the stage's `temperature` and `max_tokens`. LLM criticism is retained and supplied to repair. For `backend: decision`, configure the run's `decisions` provider and optionally `decision_override`. The repair prompt receives scores and rubric criteria without invented criticism. A judgment below `min_confidence` is rejected as uncertain and does not trigger repair. Repair judgments use additional source evidence, so an earlier decision panel does not prefetch them.

## Grounding and supported records

The first version supports single-turn `ConversationRecord` inputs, including imported seed answers and generated answers. It rejects records with nonempty `turns` rather than rewriting part of a conversation. Other record types pass through without repair.

For a record with document or chunk identifiers, Arka loads the source stage's `data.parquet`. The default stage directory is `01_source`; change `evidence_stage` when needed. A chunk must match the record's lineage root ID and both document and chunk identifiers. Arka uses that chunk in both judgments and the repair request. Missing, ambiguous or privacy-rejected evidence makes the record ineligible. Arka does not reopen the original PDF or reconstruct evidence from page coordinates. For seed answers without document identifiers, no reference evidence is assumed. Their scores do not establish independent factual correctness.

Place cheap rejection filters before repair to avoid spending on unusable records. Place all response-dependent validators after repair. Include deduplication after repair so replacements are compared with passing originals and with each other. Existing downstream stages determine which repaired records reach export. The repair-stage acceptance count precedes those checks.

## Artifacts, usage and resume

The stage directory is `runs/<run-id>/stages/03_quality_repair/`. It contains the normal `data.parquet`, `dropped.parquet` and `stats.json`, plus:

- `repair_identity.json` contains hashes of effective settings, rubric contents, permitted evidence artifacts and input records.
- `attempt-*.json` contains the original record, initial judgment, admission status, protected effective repair messages, generated replacement and second judgment as each step completes. Generation metadata records the model, prompt hash and available usage.

The runner writes stage artifacts once. A replacement has a new record ID and content hash, and its parent ID identifies the original. Arka recomputes quality scores and discards response-dependent scores and annotations. It retains instruction taxonomy assignments and seed domain metadata.

`stats.json` distinguishes initially passing records, attempted repairs and accepted repairs. `by_source` groups inspected and judged populations by document ID or seed file hash, with missing identifiers reported as `ungrouped`. Drop reasons distinguish exhausted repairs, limits, excluded dimensions, missing evidence, unsupported turns, uncertain judgments and PII rejection. `llm_usage` and `decision_usage` retain recorded calls and tokens. `cost_usd` sums known reported costs; `cost_complete` is false when a completed call did not report cost. `repair_cost_usd` excludes initial judgments and includes generation and second judgments. `repair_cost_complete` states whether those calls all reported cost. These are historical totals after resume. Failed requests that return no usable response may incur unreported usage or cost.

Run the same command with `--resume` after interruption. Each completed judgment and generation is saved with an atomic file replacement. A persisted generation is reused if its second judgment fails. A completed stage is also checked for compatible settings, rubric, evidence and input artifacts before the runner skips it. Use a new run ID after changing those inputs. The stage refuses existing attempt artifacts without `--resume`.

A crash between a provider response and its persistence can repeat the remote call. Provider idempotency is not promised. Provider and malformed-output errors stop the run; they do not silently become quality rejections. Normal runs support one `quality_repair` stage.

When `pii` is configured, the existing policy protects source record text, repair request text and generated output before persistence or subsequent calls. Rejected generated text is retained only in its protected form, with available usage, and is not judged again. Model reasoning and effective repair messages are also protected. As with the existing policy, original files, filenames, user configuration and arbitrary metadata remain outside its text boundary.

## Verification

Run `just verify` for deterministic installed-CLI proof, or run `uv run python scripts/verify_repair_cli.py --output /tmp/arka-repair-proof` for the repair recipe alone. The fixtures exercise success, repair limits, exclusions, failed replacements, privacy rejection, downstream deduplication, PDF evidence, decision judgments and interrupted resume. They do not establish model quality on representative data.

For a quality evaluation, compare `max_repairs: 0` with the chosen repair limit on the same inputs. Check exported replacements against the rejected original IDs. Report final retained yield and additional cost per recovered record after downstream checks. Use an independent review to assess quality beyond agreement with the configured judge.
