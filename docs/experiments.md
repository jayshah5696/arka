# Reproducible dataset experiments

Version 2 adds named LLM connections, explicit stage IDs, and separate execution controls. Version 1 files keep their existing execution and resume behavior. The new dependency and operational-resume contract applies to version 2 runs.

## Start from a preset

```sh
uv run arka-init --preset sft --output experiment.yaml \
  --source seeds.jsonl --model YOUR_MODEL \
  --base-url http://localhost:8000/v1 --target-count 100
```

Use `--preset pdf-sft` for a PDF source. Both presets generate complete editable files. There are no preset references to resolve at run time. Generation requires no credentials or provider requests and refuses to overwrite existing files. Set `ARKA_API_KEY` before loading the generated config. Local servers can use a dummy value.

Presets provide generation, exact deduplication, and length filtering. They do not establish dataset quality or guarantee the final count after filtering. Add quality assessment and evaluate a held-out dataset before training.

## Configuration

```yaml
version: "2"
models:
  writer:
    provider: openai
    model: YOUR_MODEL
    api_key: ${ARKA_API_KEY}
    base_url: http://localhost:8000/v1
default_model: writer
execution:
  max_workers: 4
  timeout_seconds: 30
  max_retries: 3
pipeline:
  - id: source
    config:
      type: seed_source
      path: seeds.jsonl
  - id: generate
    model: writer
    config:
      type: prompt_based_generator
      target_count: 100
      generation_multiplier: 2
      temperature: 0.7
      max_tokens: 512
  - id: deduplicate
    config:
      type: exact
output:
  format: jsonl
  path: output/dataset.jsonl
```

Each stage has a unique `id`, an optional named `model`, and a typed `config` using existing stage fields. Omitting `model` selects `default_model`. All named models live in this file. Sampling parameters belong to the stage. Version 2 rejects inline LLM connection overrides, including judge connection overrides. Use separate stages with named models, or version 1 for a repair stage that needs distinct generation and judge connections. Sampling-only LLM overrides remain supported.

Named connections support `provider: openai`, `model`, `api_key`, `base_url`, `supports_json_schema`, and `openai_compatible`. Transport settings belong under `execution`. Existing `decisions`, `embeddings`, `labeling_engine`, and `pii` sections remain available. Execution timeout and retry settings override decision and embedding transport settings as well as named LLM transport settings. Decision requests still use `decisions`; a stage's `model` selects its LLM connection.

Repeated stage types have separate artifact directories and checkpoints. Evol-Instruct expands an ID into names such as `evolve_01` and `evolve_02`. Expanded IDs must also be unique. Set `evidence_stage` to the actual source stage ID when using audits or repair, rather than the version 1 default `01_source`.

Relative paths use the config file's directory. Environment substitution happens after YAML parsing, within string values. Environment contents cannot introduce YAML keys. Missing variables and duplicate YAML keys are errors. The root `schema.json` describes both supported versions, `"1"` and `"2"`.

## Inspect and run

```sh
uv run arka --config experiment.yaml --validate-config
uv run arka --config experiment.yaml --dry-run
uv run arka --config experiment.yaml --run-id baseline
uv run arka --config experiment.yaml --run-id baseline --resume
```

Validation, preview, and execution use the same plan builder. Planning checks configured input files and fingerprints their contents. Dry run displays the fingerprint, input paths, and stage IDs without creating run state or calling providers. Execution checks dependencies again before starting.

Each run saves `plan.json` with normalized experiment settings, expanded stage IDs, dependency hashes, and a fingerprint. It also saves the original `config.resolved.yaml` with API keys omitted. `execution.history.jsonl` records operational settings for each accepted invocation. Existing stage artifacts, manifests, reports, and checkpoints remain available.

## Resume contract

Resume permits changes to concurrency, timeouts, transport retry limits, and credentials. Completed stages and persisted generator responses remain complete. Operational changes apply to unfinished work. They can affect which requests succeed, so they do not promise identical execution outcomes.

Any experiment setting change requires a new run ID. This includes model IDs and endpoints, stages and their order, prompts, sampling settings, quality rules, and output settings. Source, rubric, taxonomy, evaluation, training-artifact, and local model paths are fingerprinted by contents. Local model directories include all regular files. Moving a dependency also changes the plan.

Compatibility checks run before replacing saved configuration or opening the checkpoint database. Incompatible resumes preserve the run and make no provider requests. The original config and plan remain unchanged; operational history records later settings. Reusing a version 2 run ID without `--resume` is an error. Historical runs without a dependency plan need a new ID when moving to version 2.

Hosted models can change without an ID change. Pin provider revisions where available. Run comparison, automatic version 2 migration, and scheduling are outside this change.
