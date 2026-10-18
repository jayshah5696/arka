# arka Agent Rules

- Keep this file under 50 lines; put longer rationale in `docs/`.
- Use `just` targets (`just check`, `just test`, `just lint`); manage deps with `uv add`.
- Use red/green TDD; keep edits minimal, clear, and reversible.
- Ask user for clarification; do not assume; avoid destructive commands; preserve user changes.
- Use `.agents/skills/verify-arka/SKILL.md` after changes to CLI behavior, pipeline stages, checkpoints, or exports, and when asked to verify Arka. It drives the installed CLI with local HTTP fixtures and preserves evidence; it does not replace `just check` or `uv build`.
- Run `just verify` for the default local workflow. Native language and Hugging Face checks are optional recipes with separate prerequisites. Live provider calls require task authorization and are outside the default workflow.
- When a mapped feature or helper changes, update the verification skill and feature map and run the affected recipe. Use `maintain-verification-skill` for a broader drift audit. Keep verification work separate from product fixes, and report unverified paths.
- Every assistant response must also be saved as Markdown under:
  `docs/logs/YYYY/MM/YYYY-MM-DD-HH-mm_topic.md` (e.g. `readme`, `agents`, `plan`, `review`).

## Architecture Seams

- **Single Stage Artifact Seam**: Stages return `StageResult` (or kept `Record` lists) and write per-stage `stats.json` / `dropped.parquet` only once via `StageArtifacts(ctx).write(report=StageReport(...))`. Never write raw `stats.json` or read `stats.json` back inside a stage to patch it.
- **Bound Stage Config**: Stages read their own configuration from `self.config` bound by `StageBuilder`, never by scanning `ctx.config.pipeline` by class type.
- **Context-Owned Clients & Concurrency**: Obtain LLM and decision clients through `ctx.llm_client(override=...)` and `ctx.decision_client(override=...)`, and execute parallel work via `ctx.map_parallel(...)` instead of constructing `LLMClient` or `ThreadPoolExecutor` inside stages.
- **Registry-Driven Dispatch**: Register pipeline stages in `arka.pipeline.registry` (`StageSpec`) and decision evaluators in `arka.pipeline.decision_stages` (`DecisionStageSpec`). Never add central `if stage.type == ...` switchboards.
- **Boundary vs. Internal Types (ADR-0001)**: Use `StrictModel` (Pydantic) at YAML/JSON/Parquet/CLI boundaries and `@dataclass(frozen=True)` for internal execution containers (`StageContext`, `StageResult`, `RunResult`). Store non-numeric stage metadata via `Record.with_annotation(...)`, keeping `RecordScores.quality_per_dim` for numeric dimensions.
- **Iterative Verified Slices**: Deliver multi-seam changes in small, red/green verified slices (`just check` / `uv run pytest` + `uv run ruff check`), never in a single unverified batch.
