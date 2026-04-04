# Tenacity-based Retry Update

Updated the LLM retry implementation to use `tenacity` as requested.

## What changed

### Dependency
- Added `tenacity`

### LLM retry behavior
Refactored `src/arka/llm/client.py` to use:

```python
from tenacity import retry, stop_after_attempt, wait_fixed
```

with retry wiring around the request execution.

Current retry setup:
- `stop_after_attempt(self.config.max_retries)`
- `wait_fixed(1)`
- retries only for retryable errors:
  - `RateLimitError`
  - `APIConnectionError`
  - `APITimeoutError`
  - `InternalServerError`
- no retry for non-retryable errors:
  - `AuthenticationError`
  - `BadRequestError`

### Why this is better
- Uses a standard retry library instead of hand-rolled loops
- Keeps retry policy explicit and easier to evolve
- Makes it easier to later switch to exponential/backoff jitter if desired

## Research note
I checked external references before changing it:
- Tenacity docs / API usage patterns
- OpenAI SDK retry/error handling references

The current use is intentionally conservative and easy to reason about.

## Tests added/updated
Expanded TDD coverage for:
- retry on rate limit
- retry on timeout
- fail fast on auth error
- do not retry bad request

## Validation
All passing after the change:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Final status:
- 19 tests passed

## Next likely improvement
If desired, the next refinement would be switching from fixed wait to something like exponential backoff with jitter for real API usage. For now, `wait_fixed(1)` matches the requested direction and keeps tests deterministic.
