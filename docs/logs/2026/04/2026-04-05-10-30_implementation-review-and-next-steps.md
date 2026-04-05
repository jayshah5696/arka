# Implementation review and next steps

Date: 2026-04-05 10:30

## Current state verification

55 tests pass. Ruff clean. Significant uncommitted work (15 modified/new files, +1018/-179 lines) since the last commit (`5fce440 Harden quality filter artifacts and unify strict models`).

## Verification of top-10 issue list (actual code state now)

| # | Issue | Verified status |
|---|-------|-----------------|
| 1 | `complete_structured` naive | **Fixed.** Strategy-based: `OpenAICompatibleJsonSchemaStrategy` → `OpenAINativeParseStrategy` → `PromptParseFallbackStrategy`. Provider-native preferred. |
| 2 | Retry is fixed wait | **Fixed.** `wait_exponential(multiplier=1, min=1, max=8)` in all three retry sites. |
| 3 | No JSON format instruction | **Fixed.** Prompt includes `Return valid JSON only with this shape: ...` |
| 4 | Duplicated `StrictModel` | **Fixed.** Single definition in `src/arka/common/models.py`. |
| 5 | Dead/duplicated `openai_client.py` | **Fixed.** `build_openai_client()` lives in `openai_client.py`; `client.py` imports and uses it. No duplication. |
| 6 | Anthropic adapter missing | **Still missing.** `provider: Literal["openai"]`. Explicitly deferred in `docs/notes/temporary-defers.md`. |
| 7 | Unstable seed record id | **Fixed.** `_record_id()` uses content-based SHA-256. No `seed-{row_index}` anywhere. |
| 8 | Short content hash | **Fixed.** Full `.hexdigest()` everywhere. No `[:16]` truncation in the codebase. |
| 9 | Uncapped batch workers | **Fixed.** `bounded_worker_count()` in `src/arka/common/concurrency.py` caps at 8 default. Used by both `LLMClient.complete_batch` and `LabelingEngine.label_batch`. |
| 10 | No per-stage failure handling | **Fixed.** Inner try/except around `stage.run()`, failed stage stat recorded, `manifest.json`/`run_report.json` written on failure, SQLite status set to `failed`. |

**Score: 9/10 resolved.** Only Anthropic adapter remains — intentionally deferred.

## Architecture observations (your three points)

### 1. Filter stage pattern — confirmed good ✓

`LabelingQualityFilterStage` writes `dropped.parquet` and `stats.json` to `ctx.work_dir`. Runner reads `stats.json` via `_load_stage_stats()` to aggregate into `run_report`. The contract is clean: each stage owns its artifacts, runner aggregates. This scales to additional filter stages without changing the runner.

### 2. CLI is still hardcoded — confirmed, needs StageBuilder

`cli.py` manually assembles `[SeedSourceStage, NormalizeConversationStage]` and conditionally appends `LabelingQualityFilterStage`. This is already starting to creak — adding a length filter, language filter, or dedup stage will require touching the CLI each time.

**This is the right time to extract a `StageBuilder(config) -> list[Stage]`** before adding more stages.

### 3. `complete_structured` ceiling — partially addressed

The architecture observation about regex JSON extraction failing on nested curly braces is still technically true for the `PromptParseFallbackStrategy`, but the strategy ordering now means:

1. **OpenRouter path** → uses `response_format={"type": "json_schema", ...}` with strict schema — no regex needed
2. **Native OpenAI path** → uses `beta.chat.completions.parse` — no regex needed
3. **Fallback** → still uses regex extraction — this is where the ceiling lives

So the ceiling is real but now only applies when both provider-native paths fail. The current `_extract_json_text` regex (`\{.*\}` with `re.DOTALL`) will indeed match greedily and break on responses with multiple JSON objects. This should be documented as a known limitation.

## Additional findings from this review

### 4. OpenAIClientFactory is unused dead code

`src/arka/llm/openai_client.py` still has `OpenAIClientFactory` dataclass. Nobody uses it — `client.py` uses `build_openai_client` directly. Small cleanup target.

### 5. `_extract_json_text` greedy regex is fragile

```python
json_object_match = re.search(r"\{.*\}", stripped, re.DOTALL)
```

This matches from the first `{` to the last `}` in the entire text. If the model outputs text before/after JSON, or two JSON blocks, this silently grabs wrong content. Fine as a last-resort fallback, but worth adding a comment noting it.

### 6. LabelingEngine canary checks run after every batch

`_run_canary_checks()` fires two extra LLM calls per `label_batch()` invocation. For a 100-record dataset with `max_workers=4`, that's fine. For repeated small batches, costs double. Consider making canary frequency configurable or running once per pipeline run.

### 7. Uncommitted work is substantial and should be committed

1018 lines of changes across 15 files — including the runner failure handling, seed ID fix, structured output strategies, and bounded concurrency. This needs a commit before doing more work.

### 8. `_extract_json_text` has no test

The fallback regex extraction path doesn't have dedicated unit tests for edge cases (nested braces, multiple objects, no JSON present).

## What is next — prioritized

### Immediate (before new features)

1. **Commit the uncommitted work.** 15 files, all tests pass, all lint clean. This is a natural commit boundary: "Fix runtime safety issues: structured output strategies, failure handling, content-stable IDs, exponential backoff, bounded concurrency."

2. **Remove dead `OpenAIClientFactory` class.** 5-line cleanup.

### Next feature work (in priority order)

3. **Extract `StageBuilder(config) -> list[Stage]`.** This is the right structural move before adding any more stages. Design:
   - Input: `ResolvedConfig` + `project_root`
   - Output: `list[Stage]`
   - CLI becomes: `stages = StageBuilder(config, project_root).build()` + `PipelineRunner(...).run(..., stages=stages)`
   - Stages are assembled based on `config.data_source.type`, `config.filters.labeling_engine.enabled`, and future filter/generator config sections.

4. **Add cheap filter stages** (length filter, language filter). These are zero-LLM-cost stages that slot in before the labeling quality filter. They reduce the number of records sent to the expensive judge path. Each writes its own `dropped.parquet` and `stats.json` following the existing stage artifact pattern.

5. **Document `_extract_json_text` limitation** and add edge-case tests for the fallback path.

6. **Make canary check frequency configurable.** Default to once per pipeline run instead of once per `label_batch()`.

### Deferred (after the above)

7. Multi-judge / conflict detection — only after the single-judge path is fully solid
8. Anthropic adapter — after the stage-building pattern is proven with more stage types
9. Exact/near dedup stages (SHA-256 exact, SimHash/MinHash fuzzy)
10. Generator stages (Evol-Instruct, Magpie)

## Summary

The implementation is in good shape. The major runtime-safety issues from the top-10 list are all resolved except the intentionally deferred Anthropic adapter. The three architecture observations are confirmed:
- Filter artifact pattern is clean and scales ✓
- CLI needs StageBuilder extraction — this is the next structural move
- Structured output has a strategy chain now, but the fallback regex has a documented ceiling

**Next action: commit the uncommitted work, then extract StageBuilder.**
