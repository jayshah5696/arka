# Decision scoring and resume

## Sub-features

Seed gating, shared panel answers, rubric scoring, complexity scoring, reward scoring, inverse critic requests, confidence-weighted selection, usage summaries, redacted resolved config, and checkpoint reuse.

## How to get to it (user POV)

Configure a `decisions` provider, add decision stages, and set `backend: decision` for compatible scoring stages. Run `uv run arka --config PATH --run-id NAME`. Run the same command with `--resume` to reuse checkpoints. See `examples/09-decision-models.yaml`.

## Driving it with CLI

Run `just verify` and inspect its `decisions/` directory, or run `uv run python scripts/verify_decisions_cli.py --output /tmp/arka-decision-proof`.

Three synthetic seeds must produce nine local HTTP requests and two ChatML exports. Three inverse critic questions must use separate requests. The CLI summary and manifest must report nine requests and 90 input tokens. Inspect `runs/decision-cli/stages/03_double_critic/data.parquet` for three scored records. The resolved config must omit the dummy secret.

Resume must make zero additional requests and preserve dataset bytes. Inspect `cli.log`, `resume.log`, `requests.json`, `result.json`, and `dataset.jsonl` after the helper exits.

## Gotchas

Fixture answers prove wiring and request reuse. They do not establish provider availability or scoring quality. Usage on resume retains historical totals; inspect saved request counts to detect new calls. Optional live verification uses an instrumented production pipeline and is a separate recipe, not installed-CLI proof.
