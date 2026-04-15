# All Slices Implementation — TDD Summary

Implemented 9 slices using strict red/green TDD. All 251 tests pass, lint clean.

## Slice 1: TransformGeneratorStage
- **Files**: `src/arka/config/models.py`, `src/arka/pipeline/generator_stages.py`, `src/arka/pipeline/stage_builder.py`
- **Tests**: `tests/unit/test_transform_generator_stage.py`, `tests/unit/test_config_loader.py`, `tests/unit/test_stage_builder.py`
- Generic LLM-backed text transformation stage
- Configurable `input_field`, `output_field`, `prompt_template`, `preserve_original`
- Supports typed field paths: `payload.instruction`, `payload.response`, `payload.system`
- Preserves original text in `payload.system` as JSON metadata
- Updates lineage with `operator="transform"`, parent_ids, depth/round

## Slice 2: Stage-local Model Overrides
- **Files**: `src/arka/config/models.py`, `src/arka/pipeline/generator_stages.py`
- **Tests**: `tests/unit/test_config_loader.py`, `tests/unit/test_transform_generator_stage.py`
- Added `StageLLMOverride` config model (model, base_url, api_key)
- Added `resolve_llm_override()` helper to merge stage override onto base LLM config
- `generator.llm_override` field on `GeneratorConfig`
- `TransformGeneratorStage` uses override when building its client

## Slice 3: LabelingScoreStage (score/filter separation)
- **Files**: `src/arka/pipeline/scoring_stages.py` (new)
- **Tests**: `tests/unit/test_labeling_score_stage.py` (new)
- Scores all records using LabelingEngine rubric judge
- Does NOT drop any records — pure annotation
- Writes quality scores into `RecordScores`
- Downstream filter/select stages consume the scores

## Slice 4: RecordScores Extensions
- **Files**: `src/arka/records/models.py`
- **Tests**: `tests/unit/test_record_models.py`
- Added `humanness: float | None`
- Added `humanness_per_dim: dict[str, float] | None`
- Added `humanness_checklist: dict[str, bool] | None`
- Added `humanness_reasoning: str | None`

## Slice 5: Real IFD Implementation
- **Files**: `src/arka/llm/client.py`
- **Tests**: `tests/unit/test_llm_client.py`, `tests/unit/test_stage_builder.py`
- Enabled `response_scoring=True` for the openai provider
- Implemented `score_response()` using logprobs from OpenAI-compatible APIs
- Handles both v1 format (logprobs.content) and legacy format (logprobs.token_logprobs)
- Fallback path when echo is unsupported
- Computes mean/total logprob from per-token logprobs
- Updated IFD stage builder test since capability is now enabled

## Slice 6: RewardModelScoringStage
- **Files**: `src/arka/pipeline/scoring_stages.py`, `src/arka/config/models.py`
- **Tests**: `tests/unit/test_reward_model_stage.py` (new)
- Calls reward model endpoint for scalar score per record
- Stores in `RecordScores.reward_model`
- Optional `min_score` threshold filtering
- Supports `llm_override` for using a separate reward model endpoint
- Writes dropped.parquet and stats.json artifacts

## Slice 7: PairDeltaFilterStage
- **Files**: `src/arka/pipeline/scoring_stages.py`, `src/arka/config/models.py`
- **Tests**: `tests/unit/test_pair_delta_filter.py` (new)
- Compares configurable score field between child and parent records
- Drops records with insufficient delta (< min_delta)
- Optional `length_ratio_max` check
- Records without a matching parent pass through unchanged
- Resolves parents via `lineage.parent_ids`

## Slice 8: CompositeSelectStage
- **Files**: `src/arka/pipeline/scoring_stages.py`, `src/arka/config/models.py`
- **Tests**: `tests/unit/test_composite_select.py` (new)
- Combines multiple score signals with configurable weights
- Selects top-N by weighted composite score
- Handles missing scores as zero
- strategy: `top_n` (default)
- Writes dropped.parquet and stats.json

## Slice 9: SentenceVarianceFilterStage
- **Files**: `src/arka/pipeline/cheap_filters.py`, `src/arka/config/models.py`
- **Tests**: `tests/unit/test_sentence_variance_filter.py` (new)
- Cheap deterministic pre-filter (zero LLM cost)
- Computes coefficient of variation of sentence word-counts
- Drops records below `min_cv` threshold
- Single sentences pass by convention (CV=1.0)
- Useful for detecting AI-generated uniform sentence patterns

## Final Status
- **251 tests pass**
- **ruff lint clean**
- **91% code coverage**
- All new stages follow existing patterns (artifacts, stage events, drop reasons)
- All new config models are typed Pydantic StrictModel
- Backward compatible with existing configs
