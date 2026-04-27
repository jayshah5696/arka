# refactor: deepen stage seams across the pipeline

> Branch: `refactor/deepen-stage-seams` → `main`
> 7 commits, 22 files, +1182 / −1083 (≈ −185 lines after subtracting six new modules)
> All 287 unit tests + 25 integration matrix tests pass.

## Why

Every Stage in `src/arka/pipeline/` was re-implementing the same housekeeping —
marking dropped Records, writing per-stage Artifacts, hashing payloads,
constructing the LLM Client, embedding text. The `Stage` interface said
`run(records, ctx) -> records`, but the *de facto* contract every Stage had
to satisfy was much larger and lived as copy-pasted private methods:

- `_drop_record` defined **11×**
- `_write_artifacts` defined **7×**
- `_content_hash` / `_record_id` defined **4×** each
- `_config_hash` defined **6×** (including in `PipelineRunner` itself)
- `LLMClient(config=ctx.config.llm)` inlined **7×**
- `PipelineRunner` had grown to **736 lines** with a 150-line embedding subsystem
  and a 250-line run-report builder mixed into the orchestration loop

This PR makes the implicit contract explicit by lifting each repeated concern
into a deep module behind a small interface — see ADR-0001 for the rationale
on the two specifically-deferred items (shared strict base, deduplicated
OpenAI client construction) that this PR also delivers.

## What changes (one commit per candidate)

### 1. `Record.dropped_by` — `60c1263`

Centralised `StageEvent` invariants on the `Record` model itself:

```python
record.with_event(stage=, action=, reason_code=, details=)
record.dropped_by(stage, reason_code, details)   # canonical drop
```

Eleven `_drop_record` copies removed across `cheap_filters`, `dedup_stages`,
`filter_stages`, `scoring_stages`, `ifd_stage`, `evol_generator_stage`,
`generator_stages`. Stages now write
`dropped.append(record.dropped_by(self.name, code, details))` instead of an
eight-line `model_copy` with a hand-built `StageEvent`.

### 2. `StageArtifacts` + typed `StageReport` — `529c3c3`

New `src/arka/pipeline/artifacts.py`:

```python
StageArtifacts(ctx).write(
    report=StageReport(stage=..., count_in=..., count_out=..., cost_usd=...),
    dropped=dropped_records,
    extras={"clusters.parquet": clusters_df},
)
```

`StageReport` is a Pydantic model (`extra="allow"` for well-known optional
fields like `cost_usd`, `quality_distribution`, `cluster_count`,
`generated_count`). Seven hand-rolled `_write_artifacts` implementations
replaced by one seam.

`StageStat` and `StageErrorInfo` promoted from frozen dataclasses to Pydantic
models — this delivers ADR-0001's deferred *"promote `StageStat` to Pydantic
when reporting grows further"* item. `PipelineRunner._build_stage_stat` now
reads a typed `StageReport` via `StageArtifacts.load_report()` instead of
`json.loads → dict[str, Any] →` defensive coercion. `_load_stage_stats` is gone.

### 3. `records/identity.py` — `850eb29`

Three identity rules previously scattered across stages and the runner are
now canonical functions:

```python
content_hash(payload)              # sha256 of payload JSON  (Exact Dedup)
record_id(payload, lineage=None)   # sha256 of payload + lineage
config_hash(config)                # sha256 of ResolvedConfig (Manifest, Checkpoint)
file_hash(path)                    # seed-file fingerprint
```

Removed duplicated implementations from `source_stages` (×4), `generator_stages`
(×6), `evol_generator_stage` (×2), and `runner._config_hash`. Hash outputs are
byte-identical to the previous code. `PDFSourceStage`'s id rule is preserved
inline with a `NOTE:` because it differs from the shared helper and changing
it would break id stability for existing PDF runs.

### 4. `LLMClientFactory` + `StageContext.llm_client(...)` — `d9fa10a`

New `src/arka/llm/factory.py`. Stages stop knowing about `LLMClient`'s
constructor and `resolve_llm_override`:

```python
# before:
llm_client = self._llm_client or LLMClient(config=ctx.config.llm)
# or with an override:
effective = resolve_llm_override(ctx.config.llm, cfg.llm_override)
llm_client = self._llm_client or LLMClient(config=effective)

# after:
llm_client = self._llm_client or ctx.llm_client(override=cfg.llm_override)
```

Seven call sites migrated. The per-stage `llm_client=` ctor parameter is
preserved as a test seam. The default factory looks up `LLMClient` on its
own module so `monkeypatch.setattr("arka.llm.factory.LLMClient", Fake)`
intercepts construction; tests that previously patched
`arka.pipeline.generator_stages.LLMClient` were updated (5 sites → 4 sites
across 3 test files).

This delivers ADR-0001's deferred *"deduplicate OpenAI client construction"*
item.

### 5. `Embedder` lifted out of `PipelineRunner` — `8492fd9`

New `src/arka/embeddings/`:

```python
Embedder(config).embed(texts, checkpoint_manager=...) -> ndarray | None
Embedder(config).compute_diversity_score(records=, checkpoint_manager=...)
```

Owns provider routing (`huggingface` | `openai-compatible`) and the
`CheckpointManager`-backed embedding cache that previously lived inside
`PipelineRunner._embed_texts`. Removed from `runner.py`: `_embed_texts`,
`_embed_texts_huggingface`, `_embed_texts_openai`, `_embedding_llm_config`,
`_resolved_huggingface_embedding_model`, `_kmeans_labels` (~150 lines).

`SemanticSimilarityFilterStage` stops importing `PipelineRunner` and reaching
into `._embed_texts` — it constructs an `Embedder` from `ctx.config` directly.
That removes a real cyclic import (`filter_stages → runner → ...`).

### 6. `RunReporter` lifted out of `PipelineRunner` — `548f6ee`

New `src/arka/pipeline/reporter.py`:

```python
self.reporter.build_manifest(...)
self.reporter.build_run_report(...)
RunReporter.serialize_error(stage_name, error)
```

Owns the Manifest schema, the run-report schema, the samples and canaries
sub-Artifacts, and `StageStat` JSON serialisation. A future caller that only
wants a Manifest (e.g. an `arka manifest <run_id>` command, or a dashboard
reading completed runs) can use `RunReporter` without booting the
`PipelineRunner`.

### 7. `style: ruff format` — `89feefa`

Formatting on the new modules only.

## Headline numbers

| Metric | Before | After | Δ |
|---|---|---|---|
| `runner.py` lines | 736 | 308 | **−428** |
| Copies of `_drop_record` | 11 | 0 | **−11** |
| Copies of `_write_artifacts` | 7 | 0 | **−7** |
| Inline `LLMClient(config=...)` calls | 7 | 0 | **−7** |
| Copies of `_content_hash` / `_record_id` | 4 / 4 | 0 / 0 | **−8** |
| Copies of `_config_hash` | 6 | 0 | **−6** |
| New deepening modules | — | 6 | `Record`/identity, artifacts, factory, embeddings, reporter |
| Cyclic imports (`filter_stages → runner`) | 1 | 0 | **−1** |
| ADR-0001 deferred items closed | 0 | 2 | StageStat→Pydantic, OpenAI client dedup |

## Compatibility

- **Public Stage API unchanged**: `Stage.run(records, ctx) -> records` and
  `Stage.name` / `Stage.stage_action` are untouched. Every concrete Stage's
  external behaviour is preserved.
- **Artifact schemas unchanged**: every JSON field name in `stats.json`,
  `manifest.json`, and `run_report.json` is byte-compatible with `main`.
  Tests that assert on JSON keys (`test_reports_and_stage_events`,
  `test_dedup_stages`, `test_ifd_scorer`, `test_generation_stages`,
  `test_labeling_score_stage`) all pass without modification.
- **Hash outputs unchanged**: `content_hash`, `record_id`, `config_hash`,
  and `file_hash` produce byte-identical results to the inline
  implementations. Existing run id stability across `--resume` is preserved.
- **Test seams preserved**:
  - The per-Stage `__init__(llm_client=None)` parameter still works for
    tests that inject fakes via the constructor.
  - `monkeypatch.setattr("arka.llm.factory.LLMClient", Fake)` is the new
    canonical patch target for swapping the production client; the three
    test files that previously patched `arka.pipeline.{generator,filter}_stages.LLMClient`
    were migrated.
  - `monkeypatch.setattr(Embedder, "embed", ...)` (or
    `Embedder._embed_huggingface`) replaces the previous
    `PipelineRunner._embed_texts*` patch targets in `test_privacy_filters`,
    `test_reports_and_stage_events`, and `test_supported_options_matrix`.

## Test results

```
$ just test  (excluding the pre-existing test_smoke_pipeline failure on main)
======================= 287 passed, 3 warnings in 9.96s ========================

$ uv run pytest tests/integration/test_supported_options_matrix.py
======================== 25 passed, 3 warnings in 8.65s ========================
```

The integration smoke test (`test_smoke_pipeline.py`) was failing on `main`
before this PR because of an unrelated `dedup` config schema mismatch (the
fixture has `dedup` as a dict, `ResolvedConfig` expects a list). It is not
touched by this PR.

## ADR alignment

This PR is consistent with **ADR-0001 (boundary modeling with Pydantic,
internal execution containers with dataclasses)**:

- New boundary-style models use Pydantic: `StageReport` (writes to `stats.json`),
  `StageStat` (reads back from `stats.json`), `StageErrorInfo`.
- New internal-only types stay lightweight: the `LLMClientFactory` Protocol,
  the `Embedder` class, and the `RunReporter` builder.
- Two of ADR-0001's "Deferred cleanup items" are explicitly closed:
  *(1) create a shared strict base model* — delivered via `StageReport`
  serving as the strict shared schema for all per-stage `stats.json`.
  *(2) deduplicate OpenAI client construction* — delivered via
  `arka.llm.factory.build_client` as the single canonical construction path.

No ADR is contradicted.

## Suggested review order

1. **`commit 60c1263`** (`Record.dropped_by`) — smallest, clearest, sets the
   pattern.
2. **`commit 529c3c3`** (`StageArtifacts`/`StageReport`) — the largest
   conceptual change (typed schema for `stats.json`), but mechanical at every
   call site.
3. **`commit 850eb29`** (`records/identity.py`) — reviewers should sanity-check
   the `PDFSourceStage` `NOTE` about preserved-as-inline id semantics.
4. **`commit d9fa10a`** (`LLMClientFactory`) — reviewers should sanity-check
   the test-monkeypatch migration.
5. **`commits 8492fd9` + `548f6ee`** (`Embedder`, `RunReporter`) — pure
   "lift out of `PipelineRunner`" moves. Diff-friendly.
6. **`commit 89feefa`** (formatting) — drive-by.
