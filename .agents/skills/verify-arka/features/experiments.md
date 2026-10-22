# Generated experiments and safe resume

Run `uv run python scripts/verify_experiment_cli.py --output /tmp/arka-experiment-proof`, or inspect `experiments/` from `just verify`. Use a fresh output directory.

The helper drives installed `arka-init` and `arka` commands against a loopback OpenAI fixture. It generates a config, edits named model bindings, validates and previews it, then interrupts generation after one saved response. The default model points to an unavailable endpoint while the generator selects the fixture model. All requests must use the selected model.

Resume changes transport settings and completes two pending responses. A second resume changes concurrency and makes zero requests. Three examples survive export. The original saved config, plan, and first generator response remain unchanged. Three accepted invocations appear in execution history.

Changed source contents and a changed model ID must fail before any new HTTP calls or writes to run files and the checkpoint database. Keys must be absent from the saved config and plan. Inspect `result.json`, `requests.json`, generated YAML, command logs, stage artifacts, `plan.json`, and `execution.history.jsonl`.

Synthetic responses prove wiring and resume behavior, not dataset quality. The PDF preset uses the existing source and generator implementation; PDF provenance is verified separately by the cleanup recipe. This recipe does not cover every version 2 stage/backend combination or concurrent processes.
