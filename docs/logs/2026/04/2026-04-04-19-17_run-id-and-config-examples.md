# Run id and config examples

Date: 2026-04-04 19:17

Answered questions about `--run-id` and created a follow-up todo to build a full catalog of commented example configs for major use cases.

## What `--run-id` means

`--run-id` is the execution name for a single pipeline run.

Current significance in the codebase:

- it names the run artifact directory:
  - `runs/<run_id>/...`
- it is written into `manifest.json`
- it is written into `run_report.json`
- it is stored in the checkpoint/SQLite run registry
- it is passed into `StageContext.run_id` for every stage
- it is the unit of resume behavior when using `--resume`

So conceptually:

- config = what pipeline should do
- run_id = which specific execution instance did it

## Why it matters

A fixed `run_id` lets you:

- inspect one run's artifacts cleanly
- compare multiple runs without overwriting outputs under `runs/`
- resume the same run after interruption
- refer to a run in notes/debugging/logs by a stable name

## Practical examples

### 1. Repeatable smoke test

```bash
uv run arka --config config.smoke.yaml --run-id smoke-run
```

Artifacts go under:

- `runs/smoke-run/...`

### 2. Resume an interrupted run

```bash
uv run arka --config config.openrouter.yaml --run-id openrouter-smoke
uv run arka --config config.openrouter.yaml --run-id openrouter-smoke --resume
```

The second command reuses artifacts already written for that run id.

### 3. Compare variants

```bash
uv run arka --config config.openrouter.yaml --run-id openrouter-v1
uv run arka --config config.openrouter.yaml --run-id openrouter-v2
```

Now you can compare:

- `runs/openrouter-v1/report/run_report.json`
- `runs/openrouter-v2/report/run_report.json`

## Important caveat in current implementation

The final dataset path is still controlled by config, not by `run_id`.

So today:

- `run_id` isolates run artifacts under `runs/<run_id>/`
- but `output.path` may still overwrite the dataset file if reused across runs

Example:

```yaml
output:
  path: ./output/dataset.jsonl
```

Two different run ids will still write to the same final dataset path unless the config changes.

So right now `run_id` is most important for:

- stage artifacts
- manifests/reports
- resume
- lineage of execution

not for final dataset filename isolation.

## On building a full example config catalog

Agreed. That is the right move.

Created todo:

- `TODO-fb6486aa` — Build commented example config catalog for major Arka use cases

The right direction is to maintain a growing set of commented configs, not just one minimal example.

## Recommended config example set

As we build, we should keep examples like:

1. `config.smoke.yaml`
   - smallest no-surprises local seed pipeline

2. `config.example.yaml`
   - baseline OpenAI-compatible example

3. `config.openrouter.yaml`
   - OpenRouter single-judge quality filtering

4. `config.examples/verify-openrouter.yaml`
   - heavily commented verification/debug config

5. `config.examples/seeds-csv.yaml`
   - CSV seed ingestion example

6. `config.examples/resume-debug.yaml`
   - shows how to run/resume and inspect artifacts

7. future: `config.examples/multi-judge.yaml`
   - when slice exists

8. future: `config.examples/dedup-quality.yaml`
   - when dedup slices exist

9. future: `config.examples/preference-pairs.yaml`
   - when preference generation exists

## Documentation expectation for each example

Each example should include comments for:

- purpose
- when to use it
- required env vars
- provider/model expectations
- input file expectations
- output artifact expectations
- which features are active
- common pitfalls

## Next good step

When we switch back to edits, create a dedicated example-config directory and migrate the verification config into a commented, durable example rather than leaving it only in `scratch/`.
