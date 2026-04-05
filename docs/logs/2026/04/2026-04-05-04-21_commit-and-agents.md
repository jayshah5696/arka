# Commit and AGENTS update

Date: 2026-04-05 04:21 UTC

## Summary

Committed the completed generator + exact-dedup work and clarified task-running guidance in `AGENTS.md`.

## Commit

- `76bb98c` — `Implement generator and exact dedup pipeline slices`

## AGENTS update

Added guidance to prefer `just` targets for standard workflows:
- `just test`
- `just lint`
- `just format`
- `just check`
- `just run`

while still requiring those targets to use `uv run` under the hood.

## Why `just` vs raw `uv run`

There is no conflict:
- `just` is the project-level task interface
- `uv run` is the environment/process launcher underneath

So the intended pattern is:
- use `just` when a named project task already exists
- use raw `uv run ...` for ad-hoc commands that are not yet wrapped in the `justfile`

Example:
- preferred: `just test`
- equivalent underlying command: `uv run pytest`

## Validation

- `just check` → passed
