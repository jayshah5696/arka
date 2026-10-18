# Reasoning and exports

## Sub-features

Separate canonical reasoning fields, Parquet preservation, native language detection, strip/drop mismatch actions, answer-only records, ChatML/Alpaca think blocks, and HF token preservation.

## How to get to it (user POV)

Supply optional reasoning in seed or generated records. Add `reasoning_language_gate` with a local model path. Choose JSONL, ChatML, or Alpaca export in the config.

## Driving it with CLI

The default `just verify` checks answer-only ChatML datasets and byte-identical resume. It does not prove reasoning-language behavior or HF compatibility.

For that optional coverage, follow the shared skill's native and HF commands. Run the stack helper with `ARKA_TEST_LID_MODEL` set, then the HF helper against that output. A German trace survives; an English trace on German content strips; an absent trace remains absent. Expect two deduplicated exports, one retained trace, and one answer-only record. Inspect canonical fields, Parquet payloads, `hf-rendered.txt`, and `hf-result.json`. HF must load all exports and Qwen3 must render exactly one think block whose token decode contains trace and answer.

Run `uv run --extra language pytest tests/integration/test_reasoning_language_pipeline.py` with the same model variable to prove native seed ingestion and resume separately.

## Gotchas

No model path disables the gate with a warning. A supplied invalid model fails. Missing backend/model skips the optional integration test; report that as unverified. Qwen3 is the checked template, and other model templates need their own check. Token preservation does not establish training quality.
