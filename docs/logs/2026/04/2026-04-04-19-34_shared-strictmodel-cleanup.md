# Shared StrictModel cleanup

Date: 2026-04-04 19:34

Implemented the first cleanup item from the consistency review: deduplicated repeated `StrictModel` definitions.

## What changed

Created shared base:

- `src/arka/common/models.py`
- `src/arka/common/__init__.py`

Shared model:

```python
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
```

Updated modules to import the shared base instead of defining their own local copy:

- `src/arka/config/models.py`
- `src/arka/records/models.py`
- `src/arka/labeling/rubric.py`
- `src/arka/labeling/models.py`
- `src/arka/labeling/judges.py`

## Test coverage

Added:

- `tests/unit/test_strict_model_base.py`

This verifies that the major boundary models now inherit from the shared base:

- `ResolvedConfig`
- `Record`
- `ConversationPayload`
- `Rubric`
- `LabelResult`
- `JudgeResponse`

Also cleaned one stale typed test seam in:

- `tests/unit/test_pipeline_runner.py`

changing the resumed-stage override from `list[dict]` to `list[Record]`.

## Validation

Passed:

- `uv run pytest -q`
- `uv run ruff check .`
- `uv run ruff format --check src tests`

## Result

This removes one of the clearest consistency smells without changing runtime behavior.

The next nearby cleanup target is still:

- deduplicate OpenAI client construction between `src/arka/llm/client.py` and `src/arka/llm/openai_client.py`
