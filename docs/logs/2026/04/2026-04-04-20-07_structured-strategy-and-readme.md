# Structured output strategy pass and README sync

Continued with the next two requested follow-ups:
1. explicit structured-output strategy flow
2. README sync with current implementation reality

## Structured-output refactor

Refactored `src/arka/llm/client.py` so `complete_structured(...)` now runs through explicit strategies instead of a single hard-coded branch.

Current strategy order:
- `OpenAICompatibleJsonSchemaStrategy`
- `OpenAINativeParseStrategy`
- `PromptParseFallbackStrategy`

## Current behavior

### OpenAI-compatible / OpenRouter path
If the configured base URL points at OpenRouter, Arka now first attempts an OpenAI-compatible strict JSON schema request via:
- `response_format = { type: "json_schema", json_schema: { name, strict, schema } }`

If that request is rejected, Arka falls through to the next strategy.

### Native OpenAI parse path
If the client exposes `beta.chat.completions.parse`, Arka uses the provider-native parse helper with the Pydantic model directly.

### Prompt-parse fallback
If both native/schema-first strategies are unavailable or rejected, Arka falls back to the existing prompt+parse path.

## Capability shape
Updated `CAPABILITIES` to expose a structured-output strategy list conceptually, not just a single boolean. The implementation is still small, but the code now reflects the strategy-based direction called for by the structured-output review note.

## Tests added
Updated `tests/unit/test_llm_client.py` to cover:
- OpenRouter JSON-schema strategy first
- fallback from OpenRouter JSON-schema to native parse
- existing native parse and fallback behavior still working

## README sync
Updated `README.md` so it no longer undersells the current implementation.

Changes include:
- replaced `LabelingEngine single-judge scaffold` framing with the actual implemented single-judge quality filter path
- documented dropped-record persistence and stage stats
- documented the practical provider story: OpenAI-shaped interface, OpenRouter-compatible real path
- documented structured-output behavior at a high level
- added ADR link for boundary modeling

## Validation
Passed:
- `uv run pytest -q`
- `uv run ruff check .`

Note:
- `ruff format README.md` was not used because markdown formatting requires Ruff preview mode.
