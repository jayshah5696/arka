# Slice 1 Scaffold Alignment — Typed Records and Stage Contracts

Completed the next scaffold-alignment cycle focused on bringing the implementation closer to the approved spec direction.

## Why this cycle happened

The earlier scaffold still used raw `list[dict[str, Any]]` pipeline records, which was behind the approved spec. This cycle moved the project closer to the spec by introducing typed records and explicit stage contracts.

## Changes made

### 1. Typed record models
Added:
- `src/arka/records/models.py`
- `src/arka/records/__init__.py`

Implemented typed Pydantic models for:
- `RecordSource`
- `RecordLineage`
- `RecordScores`
- `StageEvent`
- `Record`
- `ConversationPayload`
- `ConversationRecord`

Also added a small record type registry to support round-tripping typed records from stage artifacts.

### 2. Stage protocol cleanup
Added:
- `src/arka/pipeline/stages.py`

Introduced an abstract `Stage` base class with:
- `name`
- `run(records, ctx) -> list[Record]`

This replaces the previous ad hoc stage shape and aligns better with the intended stage interface.

### 3. Pipeline model cleanup
Updated:
- `src/arka/pipeline/models.py`

Removed the old raw alias:
- `RecordList = list[dict[str, Any]]`

Added a typed `StageStat` model for manifest/status tracking.

### 4. Output serialization cleanup
Updated:
- `src/arka/pipeline/output.py`

Behavior now:
- JSONL export writes each record’s exported payload
- Parquet stage artifacts store typed records through explicit JSON-backed columns
- stage artifact reads reconstruct typed record models instead of plain dicts

This avoids the earlier mismatch between typed nested models and Parquet struct edge cases.

### 5. Runner alignment
Updated:
- `src/arka/pipeline/runner.py`

Changes:
- runner now accepts `list[Stage]`
- in-memory pipeline data is `list[Record]`
- resume path restores typed records from Parquet
- manifest `stage_stats` now includes:
  - `stage`
  - `count_in`
  - `count_out`
  - `status`
  - `resumed`

### 6. Checkpoint metadata alignment
Updated:
- `src/arka/pipeline/checkpoint.py`

Changes:
- `stage_runs` now stores explicit `status`
- added `list_stage_runs(run_id)` for easier inspection

### 7. Project rules/memory updates
Updated:
- `AGENTS.md`
- `.agents/napkin.md`

Added the preference to favor typed Pydantic pipeline models over raw dict flows.

### 8. Explicit defer note
Added:
- `docs/notes/temporary-defers.md`

This records that the Anthropic adapter is intentionally deferred until the typed scaffold and run artifacts are more mature.

## Tests added/updated first

Added or updated tests for:
- typed conversation record behavior
- stage protocol shape
- checkpoint stage status listing
- pipeline runner using typed records
- manifest stage status alignment
- typed Parquet round-trip behavior through `OutputWriter`

## Validation

All passing:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Final status:
- 21 tests passed

## What this fixes conceptually

This closes one of the biggest scaffold gaps:
- pipeline stages no longer pass around raw dict blobs
- the code now has a typed foundation closer to the approved engineering spec

## What is still unfinished

Still incomplete relative to full Slice 1 / spec intent:
- more of the spec record subclasses still need to be added over time
- stage event updates are not yet automatically appended by the runner
- manifest/report is still lighter than the final desired shape
- `LLMOutput.error` is not yet fully leveraged as a first-class non-throwing result path
- Anthropic adapter remains intentionally deferred for now

## Recommended next cycle

Next scaffold-finishing cycle should focus on:
1. automatic stage event emission in runner/output pipeline
2. `run_report.json` artifact skeleton
3. explicit sample config files (`config.example.yaml`, `config.openrouter.yaml`)
4. optional threadpool-backed `complete_batch`
