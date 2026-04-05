# Structured output best-practices review

Date: 2026-04-04 19:45

Reviewed current official guidance for OpenAI, Anthropic, OpenRouter, and Gemini on structured outputs / JSON schema / Pydantic integration.

## Bottom line

Arka is **not yet using current best practice** for structured responses.

Right now the project mostly does:
- prompt asks for JSON
- raw completion call
- regex / fenced-block extraction
- `json.loads(...)`
- Pydantic validation afterward

That is a workable fallback, but it is **not the preferred 2026 approach** when the provider supports structured outputs.

## Best-practice summary by provider

### OpenAI
Official guidance strongly prefers **Structured Outputs** over plain JSON mode.

Best practice:
- use provider-native structured output
- ideally via SDK helpers with Pydantic
- examples include:
  - `client.responses.parse(..., text_format=MyPydanticModel)`
  - `client.chat.completions.parse(..., response_format=MyPydanticModel)`
- fallback only if model/provider path does not support schema mode:
  - `text.format = {type: "json_object"}`
  - still include explicit JSON instruction

Important points from OpenAI docs:
- Structured Outputs > JSON mode
- JSON mode ensures valid JSON, not schema adherence
- schema-first + Pydantic is recommended to avoid schema/type drift
- refusals and incomplete outputs must be handled explicitly

### Anthropic
Anthropic also has native structured outputs and recommends using schema-driven output rather than prompt-only JSON requests.

Best practice:
- use `output_config.format` / SDK parse helpers with schema
- rely on schema transformation and validation from SDK helpers where possible
- treat refusals and `max_tokens` truncation as explicit edge cases

Important point:
- Anthropic docs position structured outputs as eliminating the need for formatting retries in most cases

### OpenRouter
OpenRouter supports structured outputs for compatible models.

Best practice:
- use `response_format` with `type: "json_schema"`
- enable `strict: true`
- check model support
- if using provider routing, require structured-output support where needed

Important point:
- OpenRouter explicitly recommends strict mode and schema descriptions
- prompt-only JSON is not the preferred path when model/provider support exists

### Gemini
Gemini has strong native structured output support.

Best practice:
- set response MIME type to JSON
- pass JSON schema directly
- Pydantic schemas can be converted with `model_json_schema()`
- validate final output with Pydantic in app code

Important point:
- Gemini docs explicitly support Pydantic-driven schema generation and recommend validation in application code even when schema-constrained generation is used

## What this means for Arka

## Current state in Arka
Arka currently uses a **fallback parsing strategy**:
- tell the model to return JSON
- call normal completion
- regex out a JSON object if necessary
- validate with Pydantic

This is acceptable only as:
- compatibility fallback
- test scaffold
- emergency recovery path

It should **not** remain the primary structured-output implementation for providers that support native schema-constrained output.

## Recommended design direction

### 1. Make provider-native structured output the default path
For `complete_structured(...)`, the preferred order should be:

1. provider-native schema/Pydantic path
2. JSON object mode if schema path unavailable but JSON mode exists
3. prompt+parse fallback only as last resort

### 2. Keep the current regex/fence parsing path as fallback only
Do not delete it.
It is still useful for:
- unsupported models
- OpenAI-compatible providers with imperfect compatibility
- emergency recovery / degraded mode

But it should be clearly labeled as fallback behavior, not the main design.

### 3. Use Pydantic model as the single source of truth
This is aligned with OpenAI and Gemini guidance and also good design generally.

Meaning:
- define the schema once in Pydantic
- derive JSON schema from it when supported
- validate returned content back into the same Pydantic model

### 4. Add capability-aware strategy selection
Arka already has a capability concept. Extend it.

For structured output, choose strategy based on provider/model capability:
- `native_schema`
- `json_object`
- `prompt_parse_fallback`

### 5. Treat refusal / truncation / content-filter as first-class structured-output cases
This is especially important for OpenAI and Anthropic.
Do not collapse these into generic parse failures.

## Suggested implementation strategy

### OpenAI path
Preferred:
- use SDK parse helper or structured output schema support directly
- pass Pydantic model
- handle refusal / incomplete / content filter explicitly

### OpenRouter path
Preferred:
- if model supports structured outputs, send `response_format: { type: "json_schema", ... }`
- enable strict mode
- keep fallback parser because compatibility may vary by upstream model/provider route

### Gemini path
Preferred:
- native Gemini structured outputs if building a direct adapter later
- if accessed through OpenRouter/OpenAI-compat, use the structured-output support exposed by that route when available

### Anthropic path
Preferred:
- native structured outputs via official schema mechanism once Anthropic adapter lands

## Concrete recommendation for Arka

### Do next
Refactor `LLMClient.complete_structured(...)` into a strategy-based flow:

- `OpenAIStructuredStrategy`
- `OpenAIJsonObjectStrategy`
- `PromptParseFallbackStrategy`

For current practical order of work:
1. keep current fallback path
2. add native OpenAI structured-output path first
3. then add OpenRouter structured-output path where supported
4. later add Anthropic native structured-output path

## Final judgment

If the question is:

> are we currently using best practices for JSON / structured responses via Pydantic?

The answer is:

> **No, not yet.**

Arka is currently using a decent fallback parser approach, but official 2026 best practice is:
- provider-native structured outputs first
- Pydantic as schema source of truth
- prompt+parse only as fallback

That should be the direction once the current runtime-safety fixes are handled.
