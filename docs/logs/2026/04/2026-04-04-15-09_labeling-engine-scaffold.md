# LabelingEngine Scaffold (Slice 2 Start)

Started Slice 2 by scaffolding the single-judge LabelingEngine path.

## What was added

### Labeling package
Added:
- `src/arka/labeling/__init__.py`
- `src/arka/labeling/models.py`
- `src/arka/labeling/rubric.py`
- `src/arka/labeling/prompting.py`
- `src/arka/labeling/judges.py`
- `src/arka/labeling/engine.py`

### Rubric support
Implemented:
- `RubricDimension`
- `RubricExample`
- `Rubric`
- `RubricLoader`
- `RubricValidationError`

Behavior:
- loads YAML rubric files
- validates structure with Pydantic
- computes a stable rubric hash
- validates `overall_weights` exactly match dimension names

### Single judge scaffold
Implemented:
- `JudgeResponse`
- `SingleLLMJudge`
- prompt builder helper in `prompting.py`
- weighted overall score calculation
- metadata population into `LabelResult`

### Labeling engine scaffold
Implemented:
- `LabelingEngine`
- `label(...)`
- `label_batch(...)`
- simple canary check using rubric few-shot examples
- warning when known-bad canary scores too high relative to known-good canary

### Config alignment
Updated config models to allow the beginnings of labeling configuration:
- `filters.labeling_engine`
- top-level `labeling_engine`

This keeps config shape closer to the approved spec while still staying thin.

### Rubric example file
Added:
- `rubrics/sft_quality.yaml`

## Tests added first
Added:
- `tests/unit/test_rubric.py`
- `tests/unit/test_single_judge.py`
- `tests/unit/test_labeling_engine.py`
- `tests/unit/test_labeling_config.py`

Covered:
- rubric loading and hash stability
- rubric validation failure on mismatched weight dimensions
- single judge label result metadata
- prompt construction including rubric details and few-shot examples
- batch labeling behavior
- canary warning behavior
- config loading for labeling sections

## Validation

All passing:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Final status:
- 36 tests passed

## Current state

This is a scaffolded Slice 2 start, not a full LabelingEngine implementation yet.
The core shapes now exist and are test-backed.

## Still missing relative to the spec

- no multi-judge mode yet
- no conflict detection yet
- no canary result artifact export yet
- prompt hashing is basic and local to single judge
- no pipeline integration stage using labeling yet
- no provider-native structured output path beyond current LLM abstraction

## Recommended next step

Continue Slice 2 with one of these:
1. integrate LabelingEngine into a filter stage
2. add richer canary reporting artifacts
3. begin multi-judge scaffolding

Best next move: **integrate LabelingEngine into a quality filter stage** so the judging path is exercised inside the pipeline.
