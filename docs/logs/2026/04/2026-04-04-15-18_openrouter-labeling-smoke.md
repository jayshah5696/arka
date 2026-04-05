# OpenRouter Labeling Filter Smoke Run

Integrated the LabelingEngine into a real pipeline filter stage and validated it with a live OpenRouter-backed run.

## Code changes

### New filter stage
Added:
- `src/arka/pipeline/filter_stages.py`

Implemented `LabelingQualityFilterStage`:
- loads rubric from config
- instantiates `LabelingEngine`
- scores `ConversationRecord` items
- writes label metadata into `RecordScores`
- filters out records below `min_overall_score`

### CLI pipeline wiring
Updated:
- `src/arka/cli.py`

Behavior now:
- for seed pipelines, CLI runs:
  1. `SeedSourceStage`
  2. `NormalizeConversationStage`
  3. `LabelingQualityFilterStage` when `filters.labeling_engine.enabled = true`

### Structured parse robustness
Updated:
- `src/arka/llm/client.py`
- `src/arka/labeling/prompting.py`

Improvements:
- prompt now explicitly requests valid JSON-only output
- structured parser now handles:
  - fenced JSON blocks
  - JSON embedded inside labeled text like `Scores: {...}`

### Config example
Updated:
- `config.openrouter.yaml`

Now includes labeling-engine filter configuration.

## Tests added/updated

Added:
- `tests/unit/test_labeling_filter_stage.py`
- `tests/unit/test_prompting.py`

Expanded:
- `tests/unit/test_llm_client.py`
- `tests/unit/test_config_loader.py`
- `tests/unit/test_cli.py`

Covered:
- labeling filter stage behavior
- score propagation into `RecordScores`
- threshold filtering
- prompt enforcing JSON-only response shape
- parsing structured output from code fences and labeled text
- config loading for labeling-enabled OpenRouter shape

## Validation

All passing:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Final status:
- 40 tests passed

## Live OpenRouter smoke run

Used `python-dotenv` via `uv run` to load `.env`, then ran a temporary 10-record seed dataset through the OpenRouter config path with Gemini.

Run result:
- input records: 10
- output kept after labeling filter: 8

Stage yields:
- `01_source`: 0 -> 10
- `02_normalize`: 10 -> 10
- `03_label_quality`: 10 -> 8

Temporary run workdir:
- `/tmp/arka-openrouter10.oAvs1r`

Observed kept records all had:
- `quality = 5.0`
- `instruction_clarity = 5`
- `response_quality = 5`

This confirms the first real LLM-backed end-to-end pipeline path is working.

## Note

Two of the ten simple examples were filtered out by the live judge at the current threshold (`min_overall_score = 3.5`). That is acceptable for this smoke test and demonstrates that the filter is active rather than bypassed.

## Recommended next step

Now that the first LLM-backed filter path works, the best next step is:
1. persist dropped low-quality examples with reasons
2. add quality filter stats into `run_report.json`
3. then move toward multi-judge / conflict detection
