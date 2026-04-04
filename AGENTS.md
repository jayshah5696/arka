# arka Agent Rules

- Keep this file short and project-specific. Put longer rationale in `docs/`.
- For Python projects, bootstrap with `uv init` (`--package` if this becomes a library/CLI).
- Use `just` for common project tasks instead of a `Makefile`.
- Manage dependencies with `uv add` / `uv add --dev`.
- Run everything through `uv run`; use `uvx` for one-off tools.
- Do not mix `pip`, manual `venv`, or `requirements.txt` with uv unless asked.
- Prefer a `src/` + `tests/` layout once scaffolding starts.
- Add `pytest` and `ruff` early for Python work.
- Use red/green TDD for logic changes: write or update a failing test first, then implement the minimal code to make it pass.
- Prefer typed Pydantic record/stage models over raw `dict[str, Any]` data flowing through the pipeline.
- Keep edits minimal, clear, and reversible.
- Ask before adding major frameworks, cloud services, or heavy dependencies.
- Avoid destructive commands or force operations unless explicitly requested.
- Preserve user changes; do not rewrite unrelated files.
- Every assistant response must also be saved as Markdown under:
  `docs/logs/YYYY/MM/YYYY-MM-DD-HH-mm_topic.md`
- Use short topic suffixes such as `readme`, `agents`, `plan`, `review`.
- Keep README concise; this file should stay under 50 lines.
- If project shape is unclear, clarify before scaffolding.
