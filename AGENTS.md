# arka Agent Rules

- Keep this file short and project-specific. Put longer rationale in `docs/`.
- For Python projects, bootstrap with `uv init` (`--package` if this becomes a library/CLI).
- Use `just` for common project tasks instead of a `Makefile`.
- Prefer `just` targets for standard workflows (`just test`, `just lint`, `just format`, `just check`, `just run`); use raw `uv run ...` for ad-hoc commands not already covered by the `justfile`.
- Manage dependencies with `uv add` / `uv add --dev`.
- `just` targets should themselves use `uv run`; use `uvx` for one-off tools.
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
**Always prefix commands with `rtk`**. If RTK has a dedicated filter, it uses it. If not, it passes through unchanged. This means RTK is always safe to use.
**Important**: Even in command chains with `&&`, use `rtk`:
```bash
# ❌ Wrong
git add . && git commit -m "msg" && git push
# Correct
rtk git add . && rtk git commit -m "msg" && rtk git push
```
use rtk skill if you want to learn more
