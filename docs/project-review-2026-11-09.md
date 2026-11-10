# Project review, 9 November 2026

Base revision: `5d353740ae1dd34742841071752814028db1489a`.
This report accompanies the project audit fixes on `t3/audit-fix-project-issues`.

## Scope and limits

This is a repository audit with focused reading of execution, checkpointing,
configuration, source ingestion, filtering, embeddings, reporting, recovery and
identity code. I also inspected the package configuration, example inventory,
verification recipes and documentation. I ran the full test suite and the
default installed CLI verification workflows. This is not a claim that every
line or every possible failure has been reviewed. Historical evidence, optional
native integrations and production model behavior need separate review.

## Fixed

1. Run IDs could escape the runs directory. Main pipeline paths now use the
   identifier and containment checks already used by recovery. Recovery imports
   the shared helpers from `core.paths`. Tests cover traversal, absolute paths,
   empty IDs and symlinks outside the runs directory.
2. The canary filter ignored optional conversation text. It now checks system
   text, reasoning and saved turns as well as instructions and responses.
   Tests reproduce leaks in each optional field.
3. Config hashes remained cached after nested mutation or `model_copy` updates.
   Hashes now use current config values. The unused private cache was removed.
   The serialization and hash algorithm are unchanged for unchanged configs.
4. Resumed reports used checkpoint output counts as input counts. Reports now
   retain the saved input count. The existing resume test checks that a source
   retains its original zero input count.
5. Seed ingestion hashed the complete source file for every record. Both CSV
   and JSONL readers now compute one source hash and pass it to record creation.
   Tests check the number of file hash calls and shared source provenance.
   No runtime speedup was measured.
6. Formatting an error for numeric or boolean YAML roots raised `TypeError`.
   The formatter now checks for a mapping before applying legacy diagnostics.
   Tests cover numeric, boolean, string and list roots.
7. `STATUS.md` still described the initial scaffold. It now describes the
   implemented CLI and points to current verification scope. Obsolete static
   test and coverage badges were removed from the README. The outdated example
   count was removed. The verification skill's workflow count is now six.

All behavioral changes had failing tests before their fixes. The report and
verification map distinguish unit proof from installed CLI fixture proof.

## Review first

### 1. Resume can reuse results from a different configuration

Confirmed with a local reproduction. In `pipeline/runner.py`, `register_run`
replaces the saved config hash before stage reuse. `_should_resume_stage` checks
status and path but not config identity. After changing the configured model,
resume returned the record generated under the original model.

Recommendation: define which config changes are compatible with resume, compare
identity before modifying metadata, and reject incompatible runs. Decide
whether changing only output paths or concurrency should be allowed. A blanket
comparison could prevent currently expected operational changes.

### 2. Resume does not invalidate downstream checkpoints

Source inspection in `pipeline/runner.py`. Each completed checkpoint is reused
independently. If an earlier artifact is missing and gets recomputed, later
completed artifacts can still be loaded. Source stages also inherit a no-op
`validate_resume`, so changing a source file does not itself invalidate a saved
source stage.

Recommendation: track upstream artifact identity or stop reusing downstream
stages after an upstream stage runs again. Decide whether changed source files
must fail or trigger recomputation. Preserve existing paid generation results
until that policy is agreed.

### 3. Export failures and interrupts leave misleading state

Confirmed for an export path that is a directory. The stage succeeds, export
raises `IsADirectoryError`, and the database run remains `running`.
In `pipeline/runner.py`, final export and final reporting are outside the stage
failure handler. `KeyboardInterrupt` also bypasses its `except Exception`.
Checkpoint connections are closed on selected paths rather than in `finally`.

Recommendation: put resource cleanup in `finally` and define failed versus
interrupted terminal states across stage execution, export and reporting. Keep
the original failure if secondary report writing also fails. This requires a
broader failure contract than wrapping one export call.

### 4. Reusing a run ID without resume can mix old and new artifacts

Source inspection in `pipeline/runner.py`, `pipeline/artifacts.py` and
`pipeline/generator_engine.py`. Existing run directories are accepted.
Some optional artifact writes leave old files in place, and generator response
reuse is based on prompt identity independently of the runner's resume flag.

Recommendation: choose explicit semantics for an existing ID. Reject it unless
resuming, or offer a separate replace operation with a concrete preservation
policy. Do not silently delete old work.

### 5. Embedding failures silently disable semantic filtering

Source inspection in `embeddings/embedder.py` and `pipeline/filter_stages.py`.
Provider exceptions become `None`, and semantic filtering then returns every
record. A successful pipeline can therefore omit a configured filter.

Recommendation: allow unavailable embeddings for an optional diversity report,
but raise an actionable error for a required semantic filter. Decide whether
to expose an explicit permissive policy for existing users.

### 6. Embedding cache identity omits the endpoint

Source inspection in `embeddings/embedder.py`. SQLite cache keys contain
provider, model and text. Two OpenAI-compatible endpoints with the same model
name share the cache despite potentially producing different vectors.

Recommendation: include the effective endpoint and embedding options in a
versioned cache identity. Retain old rows without interpreting them as entries
under the new identity. Also deduplicate repeated uncached texts within a batch.

## Other behavior and efficiency decisions

### 7. Concurrency settings do not consistently bound execution

In `pipeline/models.py`, `map_parallel` computes a local worker count, then
ignores it when a shared executor exists. `executor_mode` is stored but does
not select a different execution path there. `LabelingEngine.label_batch` has
the same shared-executor worker-limit issue. `provider_batch` is explicitly
rejected, so it should remain documented as unsupported.

Recommendation: define `realtime` behavior and whether a per-call limit must
constrain the shared pool. Then test actual maximum concurrent requests.

### 8. Seed ingestion changes data without a configurable policy

In `pipeline/source_stages.py`, inputs are truncated to 16,384 characters and
sanitized before storage. Instruction and response values are converted with
`str`, so nulls and objects can become training text. System and turns from
JSONL inputs are not loaded, although `ConversationPayload` supports them.

Recommendation: decide accepted seed formats and whether truncation should
reject, annotate or be opt-in. Validate types before normalization. Avoid
changing ingestion semantics without that decision.

### 9. ChatML export ignores saved turns

In `pipeline/output.py`, `_format_chatml` builds one user and assistant pair
from instruction and response. It ignores `payload.turns`. Raw JSONL retains
turns. Non-conversation records fall back to raw payloads in ChatML and Alpaca.

Recommendation: define supported multi-turn export behavior and whether formats
must reject incompatible record types. Existing exports should not change
silently.

### 10. Semantic similarity materializes every pair

In `pipeline/filter_stages.py`, `sim_matrix = gen_norm @ seed_norm.T` allocates
a generated-by-seed matrix. Large runs can exhaust memory. The current result
needs only each generated record's maximum and nearest seed.

Recommendation: compute those values in bounded batches. Verify equivalent
decisions before considering an approximate nearest-neighbor dependency.
No performance benchmark was run in this audit.

### 11. Near-dedup settings lack validation and determinism

In `config/models.py`, `NearDedupConfig` accepts unbounded hash and band counts.
In `pipeline/dedup_stages.py`, zero bands causes division by zero. More bands
than hashes produces empty shared band keys, which defeats candidate pruning.
Non-divisible counts leave trailing signature values outside band keys.
Candidate clusters are iterated as a set, so a record matching several clusters
can receive a different representative across processes.

Recommendation: define valid band partitions and stable representative order.
Decide whether non-divisible settings are rejected or partitioned unevenly.
The README's O(n) statement needs a typical-case qualification; crowded buckets
still permit quadratic candidate work.

### 12. Compatibility code contradicts the architecture rules

In `pipeline/stages.py`, `get_stage_config` still scans `ctx.config.pipeline`
by class, catches broad validation failures and can use `model_construct` after
validation fails. `config/legacy.py` synthesizes old config namespaces.
`RecordScores.with_annotation` stores nonnumeric values in both annotations and
`quality_per_dim`, despite AGENTS.md reserving the latter for numeric values.

Recommendation: choose a compatibility removal boundary. Bind modern stages
only through StageBuilder and keep any legacy conversion at input boundaries.
Do not remove the compatibility modules as dead code yet. Tests and callers
still use them, and persisted records need a read policy.

### 13. Client factory errors can be mistaken for signature mismatch

In `llm/factory.py`, `default_factory` retries construction on `TypeError`.
`build_client` also catches exceptions around an injected factory call and can
invoke it again. An internal constructor failure may cause repeated side
effects and a different final exception.

Recommendation: inspect supported arguments separately from executing the
factory, then let factory failures propagate once. Confirm the fake-client
compatibility requirements before simplifying the constructor path.

## Documentation and maintenance decisions

- `README.md` says MIT, but there is no license file in the tracked project
  inventory. Confirm the intended license and copyright holder before adding it.
- `CONTEXT.md` still describes `filters.stages` as the main filter structure and
  says exact dedup drops matching instructions or responses. The implementation
  uses the full payload content hash. Decide the intended duplicate definition
  before changing behavior or rewriting the glossary.
- `docs/PLAN.md`, `docs/SCOPE.md` and `docs/notes/temporary-defers.md` mix early
  project plans with current features. Historical verification evidence and PR
  descriptions also occupy the live docs tree. Archive or label these after
  deciding which documents remain authoritative. No historical files were deleted.
- No tracked CI workflow was found. Decide whether checks, generated schema
  parity and package build should run on pushes or PRs.
- Tests emit Pydantic `json_encoders` deprecation warnings from `StrictModel`.
  Replace that setting with supported secret serialization after verifying the
  credential redaction contract. The build also warns about the old uv build
  backend range, but the wheel and source distribution build successfully.
- Performance comments include estimates without supporting measurements,
  including a claimed 20 to 40 percent checkpoint speedup. Treat them as
  unsupported estimates, not evidence. Remove or replace them during cleanup.

## Verification

- Baseline `just check`: 770 passed, 1 skipped.
- Final `just check`: 787 passed, 1 skipped. Ruff lint and format checks passed.
- `uv build`: wheel and source distribution built successfully.
- Both committed JSON schemas matched their models before changes. Private
  cache removal does not change the public schema; final parity is checked too.
- Full default CLI verification covers recovery, ridge scoring, cleanup,
  decisions, repair and distilled quality, plus example validation and previews.
  Final evidence is `/tmp/arka-verify-0rom3li9/result.json`, with status `passed`.
  All six child results passed. Evidence includes 29 datasets and 26 resume logs.
- Reproduction script: `/tmp/arka-audit-reproduce.py`. Original observations:
  `/tmp/arka-audit-reproductions.log`. It was run before the path fix and records
  the path escape, changed-config reuse and export failure status.
- Failing tests were recorded before fixes in `/tmp/arka-audit-red.log`,
  `/tmp/arka-audit-hash-red.log` and `/tmp/arka-audit-path-red.log`.

No paid provider calls were made. Native fastText, Hugging Face export checks,
GLiNER model execution and representative production data quality remain
unverified. The skipped test requires an optional native language model.
