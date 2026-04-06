# Examples reorganization + humanizer SFT workflow

Completed the combined task:

## What was implemented

### 1. Reorganized user examples into `examples/`

Created:

- `examples/README.md`
- `examples/01-minimal.yaml`
- `examples/02-openrouter-quickstart.yaml`
- `examples/03-csv-seeds.yaml`
- `examples/04-evol-instruct.yaml`
- `examples/05-pdf-grounded.yaml`
- `examples/06-dedup-quality-filter.yaml`
- `examples/07-resume-debug.yaml`
- `examples/future/README.md`
- `examples/future/multi-judge.yaml`
- `examples/future/preference-pairs.yaml`
- `examples/future/contamination-audit.yaml`
- `examples/seeds/03-python-qa.csv`
- `examples/seeds/04-coding-seeds.csv`
- `examples/seeds/07-humanizer-rewrite.jsonl`
- `examples/pdfs/sample.pdf` (downloaded BCG PDF for the PDF-grounded demo)

Also moved/retired old root-level configs by renaming them to `.bak` and moved smoke config usage to a test fixture:

- `tests/fixtures/smoke.yaml`
- old root configs renamed to `*.bak`

### 2. Added example validation infrastructure

Created:

- `src/arka/examples_validation.py`
- `scripts/validate_examples.py`

Updated:

- `tests/unit/test_example_configs.py`
- `justfile` with `validate-examples`

Validation now checks:

1. all example YAML files parse
2. all 7 required header fields exist
3. COST is one of `free|low|medium|high`
4. OpenRouter configs use `OPENROUTER_API_KEY`
5. `output.path` is relative (`./...`)
6. referenced seed files exist
7. future examples include TODO slice/milestone markers

### 3. Humanizer-style SFT path

Built a concrete seed set for rewriting tasks covering:

- proofreading
- shortening
- formalization
- summarization
- clarity
- humanization / de-AI-ification

The response format for these seeds uses:

- `brief_rationale`
- `final_text`

This matches the requested behavior where outputs include a short explanation plus the rewritten text.

### 4. Docs and references updated

Updated:

- `README.md`
- `docs/config-examples.md` (now points to `examples/README.md`)
- `docs/validation-matrix.md`
- `tests/integration/test_smoke_pipeline.py`
- `tests/unit/test_cli.py`

## Live runs completed

Executed all main examples with real provider calls using OpenRouter + local embeddings where applicable:

- `examples/01-minimal.yaml` → `test-01`
- `examples/02-openrouter-quickstart.yaml` → `test-02`
- `examples/03-csv-seeds.yaml` → `test-03`
- `examples/04-evol-instruct.yaml` → `test-04`
- `examples/05-pdf-grounded.yaml` → `test-05`
- `examples/06-dedup-quality-filter.yaml` → `test-06`
- `examples/07-resume-debug.yaml` → `test-07`
- resume rerun also completed for `test-07 --resume`

### Output counts

- `01-minimal` → 10 records
- `02-openrouter-quickstart` → 20 records
- `03-csv-seeds` → 5 records
- `04-evol-instruct` → 14 records
- `05-pdf-grounded` → 10 records
- `06-dedup-quality-filter` → 19 records
- `07-resume-debug` → 10 records

### Notable confirmations

- `06-dedup-quality-filter` completed with canary status `pass`
- `07-resume-debug` completed with canary status `pass`
- `05-pdf-grounded` completed successfully against the downloaded sample PDF
- `04-evol-instruct` produced evolved lineage records

## Validation passed

- `uv run pytest -q`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run python scripts/validate_examples.py`

Result:
- `224 passed`
- all example validations passed

## Simple explanation of the validation matrix

You asked what the test matrix is.

It is **developer coverage**, not the user workflow.
It exists to verify supported config combinations, such as:

- JSONL vs CSV seeds
- output formats (`jsonl`, `chatml`, `alpaca`)
- executor-mode propagation
- dedup combinations
- quality-filter report artifacts

So it is useful for confidence, but it should sit behind the user-facing example catalog rather than replace it.
