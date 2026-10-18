# Selection and benchmark filtering

## Sub-features

Taxonomy minimum counts, constrained selection, infeasible-pool reporting, benchmark overlap drops or flags, and calibration.

## How to get to it (user POV)

Generate tagged records with `taxonomy_generator`, configure `select.coverage`, and run the CLI. Configure `decontaminate.eval_paths` and threshold to check benchmark overlap before export.

## Driving it with CLI

Run `just verify`. Under `cleanup/taxonomy/`, eight taxonomy-generation requests must export two records, one science and one arts, with saved taxonomy assignments. Inspect `runs/taxonomy/stages/05_composite_select/data.parquet` and stats. Under `decisions/`, confidence-weighted selection must reduce three scored records to two exports.

With the shared skill's native prerequisites, run `uv run --extra language python scripts/verify_stack_cli.py --output /tmp/arka-native-proof` to plant the German Eiffel Tower question and verify one benchmark drop in `02k_decontaminate`.

Run `uv run pytest tests/unit/test_taxonomy_coverage.py tests/unit/test_decontamination.py tests/integration/test_decontamination_pipeline.py` for infeasible pools, overlapping constraints, drop/flag reports, weighting, and calibration-prefix failures. The taxonomy integration's synthetic stage checks the runner; the HTTP helper proves the user CLI generation path.

## Gotchas

Taxonomy coverage does not promise model-quality gains. Bounded search can report unresolved constraints. Exact n-grams can miss paraphrases. Fixture recall and timing are separate from production quality. Calibration must not count an already-contaminated prefix as a newly planted leak.
