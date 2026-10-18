# Generated text cleanup

## Sub-features

Repetition drops, foreign identity drops with denial exceptions, opener stripping, placeholder drops, cleaned-text deduplication, and immutable lineage.

## How to get to it (user POV)

Place `artifact_filter`, `repetition`, and `identity_claim` after generation in a pipeline config. Export survivors with the CLI.

## Driving it with CLI

Run `just verify` and inspect `cleanup/`, or run `uv run python scripts/verify_cleanup_cli.py --output /tmp/arka-cleanup-proof`. Six PDF-generation HTTP responses include an opener, its clean duplicate, a word loop, a first-person creator claim, a placeholder, and third-person discussion. Expect two exports, one drop from each filter, and one duplicate removed. Inspect dropped Parquet, audit events, and hashes. Both PDF and taxonomy runs must resume with identical bytes and zero extra provider calls.

Use `uv run pytest tests/integration/test_repetition_pipeline.py tests/integration/test_identity_claim_pipeline.py tests/integration/test_artifact_filter_pipeline.py` for focused CLI integration cases. Unit cases cover denial variants, protected code/quotes, and regex boundaries.

## Gotchas

These stages are opt-in. Local SDK responses are deterministic fixtures. They prove wiring and behavior, not prevalence or false-positive rates on generated production data. The issue's 500-record repetition audit and 1,000-record identity audit remain separate requirements.
