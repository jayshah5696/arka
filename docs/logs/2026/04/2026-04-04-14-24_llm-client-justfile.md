# Slice 1 Progress — justfile Switch and LLM Client Expansion

Continued Slice 1 with two major changes:

1. switched project task runner from `Makefile` to `justfile`
2. expanded the LLM foundation with a real minimal `LLMClient`

## Changes made

### Task runner
- Added `justfile` with recipes:
  - `setup`
  - `test`
  - `lint`
  - `format`
  - `check`
  - `run`
  - `clean`
- Removed `Makefile`
- Updated `AGENTS.md` to prefer `just`
- Updated `.agents/napkin.md` with the new preference
- Updated `README.md` and `STATUS.md`

### Config model improvements
Extended `LLMConfig` to support OpenAI-compatible routing setups:
- `timeout_seconds`
- `max_retries`
- `openai_compatible.referer`
- `openai_compatible.title`

This keeps OpenAI first while allowing OpenRouter-style configuration later.

### LLM client implementation
Added:
- `src/arka/llm/models.py`
- `src/arka/llm/client.py`

Implemented minimal functionality:
- `complete(messages)`
- `complete_structured(messages, schema)`
- `complete_batch(batch)`
- retry on rate limit and retryable API errors
- fail fast on auth errors
- usage extraction
- latency tracking
- capability registry scaffold

### OpenAI-compatible headers
Updated `OpenAIClientFactory` to pass:
- `HTTP-Referer`
- `X-Title`

when configured. This is useful for OpenRouter-style OpenAI-compatible endpoints.

### CLI improvements
Updated `src/arka/cli.py` to support:
- `--config`
- `--run-id`
- `--resume`

### Pipeline manifest improvements
Added `stage_stats` into `manifest.json` so each run records per-stage counts and resume flags.

## Tests added first (TDD)
Added or expanded tests for:
- LLM client basic completion
- retry on rate limit
- fail-fast on auth error
- structured JSON parse into Pydantic model
- OpenAI-compatible custom base URL support
- CLI `--config` / `--run-id` / `--resume`
- config support for OpenRouter-style OpenAI-compatible settings
- manifest stage stats
- OpenRouter-compatible headers in `OpenAIClientFactory`

## Validation

Commands passing:

```bash
just test
just check
```

Equivalent verified commands:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Final status:
- 17 tests passed

## Current state

The project now has a stronger Slice 1 foundation:
- config-driven run skeleton
- resumable stage runner
- parquet + JSONL outputs
- SQLite run registry
- CLI argument support
- minimal LLM client with retry/error behavior
- OpenAI-compatible future path for OpenRouter routing

## Not done yet

Still incomplete relative to the full spec:
- richer `LLMError` propagation instead of mostly raising `LLMClientError`
- token/cost estimation beyond direct response usage
- threadpool-backed `complete_batch`
- structured output via provider-native schema APIs
- partial-failure artifact handling
- dedicated stage protocols/base classes
- run report artifact beyond manifest

## Future OpenRouter test note

When an OpenRouter key is added later, a good test config shape is:

```yaml
llm:
  provider: openai
  model: google/gemini-3.1-flash-lite-preview
  api_key: ${OPENROUTER_API_KEY}
  base_url: https://openrouter.ai/api/v1
  openai_compatible:
    referer: https://your-site.example
    title: arka
```

This matches the current config model and client setup.
