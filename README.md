# arka (अर्क)

**Arka** is a Sanskrit term from Ayurveda for a steam distillate: apply heat to raw material, collect the pure essence.

This repository is the starting point for tools, experiments, and systems built around that idea — turn noisy inputs into clear, useful outputs.

## Status

Bootstrap stage. The project conventions and working rules are being set before the codebase expands.

## Principles

- Config over hidden state
- Inspectable outputs over opaque pipelines
- Small, composable tools over heavy frameworks
- Decisions and responses written down as project artifacts

## Working Conventions

- For Python work, initialize and manage the project with `uv`
- Prefer `uv init`, `uv add`, `uv run`, and `uvx`
- Use `just` for common project tasks
- Save timestamped notes and responses under `docs/logs/YYYY/MM/`
- Keep root docs concise; move longer rationale into `docs/`

## Project Docs

- `AGENTS.md` — focused rules for agents working in this repo
- `justfile` — common developer tasks
- `docs/logs/` — timestamped logs, notes, and response artifacts

## Next

- Continue Slice 1 foundation work
- Expand the LLM client abstraction
- Add CLI/resume polish and stage stats
