# Commit readiness and next-step recommendation

Date: 2026-04-04 14:59

## Commit readiness

Yes — this now looks commit-worthy.

The project has crossed the line from scattered bootstrap work into a coherent first foundation snapshot:
- uv-based project scaffold
- justfile task runner
- tests and linting in place
- typed record models introduced
- stage protocol introduced
- resumable runner/checkpoint flow present
- run report artifact added
- docs/spec/scope/plan present in repo
- temporary defers documented explicitly

This is a reasonable first commit boundary for Slice 1 foundation scaffolding.

## What improved since the earlier review

The biggest earlier concern was that the code still revolved around `list[dict]`.
That has improved materially:
- `src/arka/records/models.py` now defines typed record structures
- `src/arka/pipeline/stages.py` adds an explicit stage base class
- runner now works with `Record` objects and emits `StageEvent`s

That is the right correction.

## Remaining small issues (not blockers for commit)

### 1. Report example drift
The approved spec uses richer contamination and reporting language than the current scaffold.
That is fine for now, but current `run_report.json` is still minimal. Commit this as scaffold, not feature-complete reporting.

### 2. OpenRouter attention is still slightly ahead of priority
Not a blocker, just worth watching. Good that `docs/notes/temporary-defers.md` now explicitly records Anthropic defer rationale.

### 3. Some local repo noise exists
Not a blocker for commit, but before or after commit you may want to ensure local-only artifacts stay untracked and clean:
- `.coverage`
- `__pycache__/`
- `.DS_Store`
- `.pytest_cache/`
- `.ruff_cache/`

## Recommendation on next step

Of the two options:
- source stage + smoke pipeline
- Slice 2 LabelingEngine skeleton

**Recommended next: source stage + smoke pipeline**

## Why this should be next

Because the current code is still finishing Slice 1 behavior, not yet ready to jump to Slice 2.
A smokeable vertical path is more valuable right now than a judge skeleton.

What you want next is:
1. a real source stage
2. one tiny pipeline run that starts from actual input
3. stage artifacts written end to end
4. output JSONL written from a real source record
5. smoke config + smoke command

That gives you a working backbone.

Once that exists, Slice 2 LabelingEngine can plug into something real instead of into scaffold abstractions alone.

## Suggested concrete next implementation target

Build a minimal smoke pipeline with:
- `SeedSourceStage` reading a small JSONL/CSV seed file
- one no-op or trivial transform stage
- final JSONL export
- `config.smoke.yaml`
- one integration-style test proving:
  - config loads
  - source stage emits typed records
  - runner writes Parquet stage artifact
  - runner writes output JSONL
  - run report is created

## Suggested commit message shape

A good commit message would be something like:

```text
Bootstrap Slice 1 foundation scaffold
```

or, if you want slightly more detail:

```text
Scaffold Slice 1 foundation with typed records and runner artifacts
```

## Final recommendation

- **Yes, commit these changes**
- Then do **source stage + smoke pipeline** next
- Start LabelingEngine only after the smoke path exists
