# Napkin

## Corrections
| Date | Source | What Went Wrong | What To Do Instead |
|------|--------|----------------|-------------------|
| 2026-04-04 | user | TDD approach was not explicit in project rules | Use red/green TDD for logic changes: write a failing test first, then implement the minimum code to pass |
| 2026-04-04 | user | Added a `Makefile` even though task runner preference was not yet settled | Use `just` for common project tasks instead of a `Makefile` |
| 2026-04-04 | user | Pipeline scaffold used raw `list[dict[str, Any]]` records, which diverged from the approved spec direction | Move the pipeline to typed Pydantic record and stage models early |

## User Preferences
- Use uv for Python project management
- Polars is acceptable for data/Parquet workflows
- OpenAI first, but config should stay flexible enough for OpenRouter-compatible base URLs later
- Prefer red/green TDD
- Use `just` instead of `make` for common project tasks

## Patterns That Work
- Start with a thin vertical slice before implementing real generation logic

## Patterns That Don't Work
- Starting with deep feature logic before the pipeline skeleton exists

## Domain Notes
- Arka is a config-driven synthetic data generation framework with Slice 1 foundation as the current target
