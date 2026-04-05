# Synth Data Playbook — Engineering Spec

**Status:** Approved for implementation (partially implemented; see current implementation notes below)
**Version:** 2.0 (post grill + rebuttal + final verdict)
**Last updated:** April 2026
**Author:** Jay Shah
**Reviewed by:** Senior engineer grill (spec-grill-apr-2026.md)
**Verdict:** spec-final-verdict-apr-2026.md

---

## Overall Goal

Build a config-driven synthetic data generation framework from first principles
using LLM APIs. Input: a YAML config. Output: a training-ready JSONL dataset +
a run report artifact.

The learning objective is equal to the engineering objective. Every technique
is implemented from scratch — Magpie, Evol-Instruct, IFD scoring, MinHash dedup,
LLM-as-judge — so the internals are fully understood. No black-box frameworks.

Starts as a personal research tool. Designed for OSS from day one (typed interfaces,
inspectable artifacts, provenance on everything, clear module boundaries).

**Target scale:** under 50K examples per run for v0.


## Primary Use Cases

What this system builds training data for:

- Instruction tuning (SFT)
- Preference / RL data (DPO, GRPO, RLVR)
- Embedding training data
- Reranker training data
- Evaluation datasets
- Tool use / trajectory data
- PDF / private document grounded generation
- LabelingEngine (rubric-driven multi-LLM scoring — shared with work project)
- Continuous learning flywheel (human corrections → preference pairs)

---

## System Boundary

What lives inside this system vs what is external:

```
┌─────────────────────────────────────────────────────────────────┐
│                        THIS SYSTEM                              │
│                                                                 │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  Config  │  │  Pipeline │  │  Stages  │  │   Artifacts  │  │
│  │  (YAML)  │→ │  Runner   │→ │ (typed)  │→ │  (Parquet /  │  │
│  │ Pydantic │  │ + SQLite  │  │          │  │  JSONL / DB) │  │
│  └──────────┘  └───────────┘  └──────────┘  └──────────────┘  │
│                                    │                            │
│                          ┌─────────▼──────────┐                │
│                          │     LLM Client     │                │
│                          │  (retry / batch /  │                │
│                          │  structured output)│                │
│                          └─────────┬──────────┘                │
└────────────────────────────────────┼────────────────────────── ┘
                                     │ API calls
              ┌──────────────────────▼──────────────────────┐
              │              EXTERNAL                        │
              │  OpenAI / Anthropic / Gemini APIs            │
              │  HuggingFace datasets (optional source)      │
              │  Your PDF / CSV / JSONL files (data source)  │
              │  Eval benchmarks (contamination check only)  │
              └──────────────────────────────────────────────┘
```

**Inside:** config, pipeline runner, all stage logic, LabelingEngine,
dedup, quality filtering, dataset report, artifact storage.

**Outside:** LLM providers (API only), source documents, eval benchmarks
(read-only for contamination), training frameworks (consume the output).

---

## Data Flow

How data moves through the system end to end.

### Current implementation

```
 Seeds (JSONL / CSV)
         │
         ▼
 ┌───────────────┐
 │  DataSource   │  normalized Records with source provenance
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │   Generator   │  prompt-based generation
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │ Exact Dedup   │  payload/content-hash based
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │ Cheap Filters │  length, language
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │ Label Quality │  single-judge LabelingEngine filter
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │    Output     │  dataset.jsonl + run artifacts
 └───────────────┘
```

### Planned target flow

```
 Seeds / PDF / Seedless
         │
         ▼
 ┌───────────────┐
 │  DataSource   │  normalized Records with source provenance
 └───────┬───────┘
         │  ~seeds
         ▼
 ┌───────────────┐
 │   Generator   │  raw (instruction, response) pairs
 │ Prompt-based  │  target: 5–10× final volume
 │ Evol-Instruct │  lineage tracked per record
 └───────┬───────┘
         │  ~50K raw
         ▼
 ┌────────────────────────────────────────────────────────┐
 │      Filter + Dedup  (interleaved by cost)             │
 │                                                        │
 │  1. sanitize        (format / null check, zero cost)   │
 │  2. cheap filters   (length, language, heuristics)     │
 │  3. exact dedup     (SHA-256 hash)                     │
 │  4. near dedup      (SimHash short / MinHash long)     │
 │  5. score           (LabelingEngine — expensive)       │
 │  6. semantic dedup  (embedding cosine, after scoring)  │
 │  7. select          (top-N by composite score)         │
 │                                                        │
 │  dropped.parquet + reason codes at every step          │
 │  clusters.parquet written at steps 3, 4, 6             │
 └───────────────────────┬────────────────────────────────┘
         │  ~10–15K after dedup
         ▼
 ┌───────────────┐
 │ Contamination │  n-gram + semantic check vs eval benchmarks
 │    Check      │  >5% hit rate = pipeline error
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │   Formatter   │  ChatML | Alpaca | DPO | triplet
 └───────┬───────┘
         │
         ▼
 ┌───────────────────────────────────────┐
 │           Output                     │
 │  dataset.jsonl  (final export)       │
 │  run_report.json  (mandatory)        │
 │  samples.jsonl  (20 for human review)│
 │  canaries.json                       │
 │  dropped.parquet  (why things died)  │
 └───────────────────────────────────────┘
```

**LabelingEngine** is called at two points:
- Inside Quality Filter (scoring stage) — rates instruction/response quality
- Inside Generator (optional) — rates evolved instructions before generating responses

**LLMClient** is called by every stage that needs an LLM —
Generator, LabelingEngine, IFD scorer, hard negative selector, query generator.
Single shared client, all usage tracked and summed in run_report.

---

## Settled Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM access | API only | No local inference complexity |
| Orchestration | Custom thin runner | Learn the logic, own every line |
| Off-the-shelf tools | Reference only | Distilabel/curator read, not run |
| Concurrency | Sync + ThreadPoolExecutor | Sufficient at <50K, no async complexity |
| Config format | YAML + Pydantic validation | Fail fast on bad config |
| Internal storage | Parquet per stage | Analytics, resume, auditable |
| Metadata storage | SQLite | Run registry, lineage index, stats |
| Final output | JSONL | Simple, portable, training-compatible |
| Checkpointing | Parquet stage outputs + SQLite index | Resume any stage |
| Smoke test | Dataset report (mandatory artifact) | Canaries + stats + samples |
| Executor modes | config-gated (threadpool default) | Extension point for provider batch |

---

## Scope Tiers

### Tier A — v0 (build first, get right)
- Foundation: LLMClient, config, pipeline runner, typed records, Parquet storage
- LabelingEngine: single judge + multi-judge
- Data sources: seeds, PDF ingestion (text-native only), persona pool
- Instruction generation: prompt-based default, Evol-Instruct
- Quality filter pipeline
- Dedup pipeline
- Dataset report + contamination check
- Full SFT pipeline end-to-end

### Tier B — v1 (extend after Tier A is solid)
- Preference pairs (Best-of-N, rejection sampling, synthetic degradation)
- Embedding triplets + hard negatives
- Reranker candidate list annotation

### Tier C — v2 (research modules)
- Magpie (capability-gated, requires raw completion API)
- RLVR problem datasets
- Tool use / trajectory data
- Continuous learning flywheel

---

## Core Data Types

These are defined before any pipeline logic. Everything flows through these types.

### Record (base)

```python
class RecordSource(BaseModel):
    type: str                    # pdf_chunk | seed | generated | evolved
    doc_id: str | None           # for PDF-grounded
    chunk_id: str | None
    page_start: int | None
    page_end: int | None
    char_start: int | None
    char_end: int | None
    source_hash: str | None      # hash of source file
    seed_file_hash: str | None   # hash of seed file if from seeds

class RecordLineage(BaseModel):
    root_id: str                 # original seed/chunk this derives from
    parent_ids: list[str]        # immediate parents (1 for most, 2 for pairs)
    operator: str | None         # evol operator, generation method, etc.
    round: int | None            # evol round
    depth: int | None            # how many evolution steps from root

class RecordScores(BaseModel):
    quality: float | None        # LabelingEngine overall score
    quality_per_dim: dict        # per-dimension scores
    rubric_hash: str | None      # which rubric produced these scores
    rubric_version: str | None
    judge_model: str | None
    judge_prompt_hash: str | None
    ifd: float | None            # optional, provider-gated
    reward_model: float | None   # optional

class StageEvent(BaseModel):
    stage: str
    action: str                  # kept | dropped | scored | transformed | flagged
    reason_code: str | None      # too_short | duplicate | low_quality | contaminated
    details: str | None
    seq: int                     # sequence index within run

class Record(BaseModel):
    id: str                      # content-stable identity: SHA-256(payload + lineage context)
                                 # same content + same parent = same id (dedup-safe)
                                 # NOT an execution-instance id — use run_id+stage_event.seq for that
    content_hash: str            # current implementation: hash of payload content only
                                 # future exact-dedup target may use normalized instruction text
    source: RecordSource
    lineage: RecordLineage
    payload: dict                # subclassed per record type
    scores: RecordScores
    stage_events: list[StageEvent]
    config_hash: str             # hash of config that produced this record
    created_at: str              # ISO timestamp
```

### Use-case record types

```python
class ConversationPayload(BaseModel):
    instruction: str
    response: str
    system: str | None
    turns: list[dict] | None     # for multi-turn

class ConversationRecord(Record):
    payload: ConversationPayload

class PreferencePayload(BaseModel):
    prompt: str
    chosen: str
    rejected: str
    margin: float                # score(chosen) - score(rejected)

class PreferenceRecord(Record):
    payload: PreferencePayload

class TripletPayload(BaseModel):
    query: str
    positive: str
    negative: str
    task: str
    negative_type: str           # bm25 | dense | llm_hard | synthetic

class TripletRecord(Record):
    payload: TripletPayload

class CandidateListPayload(BaseModel):
    query: str
    candidates: list[dict]       # [{text, relevance_score, rank}]

class CandidateListRecord(Record):
    payload: CandidateListPayload

class TrajectoryPayload(BaseModel):
    task: str
    tools: list[dict]
    steps: list[dict]            # [{tool, args, result, reasoning}]
    final_answer: str
    trace_type: str              # synthetic_unverified | simulated_verified

class TrajectoryRecord(Record):
    payload: TrajectoryPayload

class GroundedChunkPayload(BaseModel):
    text: str                    # cleaned chunk text
    doc_id: str                  # parent document identifier
    chunk_idx: int               # position within document
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    word_count: int
    chunk_strategy: str          # fixed | semantic

class GroundedChunkRecord(Record):
    payload: GroundedChunkPayload
    # source fields (doc_id, page, char range, source_hash) also live in
    # Record.source for pipeline-wide provenance tracing.
```

### LLMOutput

```python
class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None       # estimated from known pricing

class LLMError(BaseModel):
    type: str                    # rate_limit | auth | content_filter | timeout | parse
    message: str
    retryable: bool

class LLMOutput(BaseModel):
    text: str | None
    parsed: BaseModel | None     # populated by complete_structured()
    usage: TokenUsage
    finish_reason: str | None
    model: str
    provider: str
    request_id: str | None       # provider request ID for debugging
    latency_ms: int
    error: LLMError | None
```

### StageContext

```python
class StageContext(BaseModel):
    run_id: str
    stage_name: str
    work_dir: Path               # runs/<run_id>/stages/<stage_name>/
    config: dict
    executor_mode: str           # realtime | threadpool | provider_batch
    max_workers: int
```

---

## Artifact Layout

```
runs/
  <run_id>/
    manifest.json                 # run_id, config_hash, start/end time, stage list
    config.resolved.yaml          # fully resolved config (env vars substituted)
    stages/
      01_source/
        data.parquet
      02_normalize/
        data.parquet
      02_generate/
        data.parquet
      02c_exact_dedup/
        data.parquet
        dropped.parquet
        clusters.parquet
        stats.json
      02a_length_filter/
        data.parquet
        dropped.parquet
        stats.json
      02b_language_filter/
        data.parquet
        dropped.parquet
        stats.json
      03_label_quality/
        data.parquet
        dropped.parquet
        stats.json
    report/
      run_report.json
      contamination.json
state.db                          # SQLite: runs, stages, artifacts, lineage index
dataset.jsonl                     # final export (symlink or copy from format stage)
```

---

## Config Schema (YAML)

```yaml
version: "1"
run_id: null                      # auto-generated if null

llm:
  provider: openai                # openai | anthropic (v0)
                                  # gemini: roadmap v1, config rejects it in v0
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}

executor:
  mode: threadpool                # realtime | threadpool | provider_batch (stub)
  max_workers: 10

data_source:
  type: seeds                     # seeds | pdf | hf_dataset
  # NOTE: seedless is NOT a standalone Tier A source type.
  # Use omit data_source + set generator.use_taxonomy_only: true for
  # prompt-based generation without per-example seeds.
  # Magpie (Tier C) is the only true seedless generator.
  path: ./seeds.jsonl
  # pdf options:
  # chunk_strategy: fixed | semantic
  # chunk_size_tokens: 512
  # chunk_overlap_tokens: 64

generator:
  type: prompt_based              # current implementation: prompt_based only
  target_count: 10000
  generation_multiplier: 5        # generate 5x, filter down

  # planned later:
  # persona_conditioning: true
  # persona_pool_path: ./personas.jsonl
  # topic_taxonomy_path: ./taxonomy.yaml
  # rounds: 4
  # branching_factor: 2
  # operators: [add_constraints, deepen, increase_reasoning_steps, breadth_mutation]

dedup:
  exact:
    enabled: false

filters:
  length:
    enabled: false
    min_instruction_chars: 10
    max_instruction_chars: 4096
    min_response_chars: 10
    max_response_chars: 16384
  language:
    enabled: false
    allowed: [en]
  labeling_engine:
    enabled: true
    rubric_path: ./rubrics/sft_quality.yaml
    min_overall_score: 3.5
  target_count: 10000

contamination:
  enabled: true
  benchmark_paths: []             # paths to eval benchmark JSONL files
  ngram_size: 13
  flag_threshold: 0.01            # >1% hit rate = warning
  error_threshold: 0.05           # >5% hit rate = pipeline error

labeling_engine:
  rubric_path: ./rubrics/sft_quality.yaml
  mode: single                    # single | multi
  # multi options:
  # judges:
  #   - provider: openai
  #     model: gpt-4o
  #     weight: 0.7
  #   - provider: anthropic
  #     model: claude-sonnet-4.5
  #     weight: 0.3
  # conflict_std_threshold: 1.0

output:
  format: chatml                  # chatml | alpaca | dpo | triplet
  path: ./output/dataset.jsonl
```

---

## Slice Specs

---

### SLICE 1 — Foundation

**Tier:** A
**Goal:** The skeleton everything runs on. Config loads, validates, LLM is called
once, output is written, run is checkpointed. No real generation yet.

**What you build:**

**1. `LLMClient`**
- Providers: openai, anthropic (same interface, provider-specific adapters internally)
- `complete(messages) -> LLMOutput`
- `complete_structured(messages, schema: type[BaseModel]) -> LLMOutput`
  - Uses JSON mode (OpenAI) or tool-use coercion (Anthropic)
- `complete_batch(batch, max_workers) -> list[LLMOutput]`
  - ThreadPoolExecutor internally
- Retry: exponential backoff, 3 attempts max
  - rate_limit → retry with backoff
  - auth → fail fast, do not retry
  - content_filter → skip + log, do not retry
- Token counting: tiktoken estimate before call, actual from response after
- Provider capability registry:
  ```python
  CAPABILITIES = {
      "openai": {
          "structured_output": True,
          "logprobs": True,
          "batch_api": True,
      },
      "anthropic": {
          "structured_output": True,
          "logprobs": False,
          "batch_api": True,
      },
  }
  ```

**Provider note:** v0 ships openai and anthropic adapters only. Config validation raises `ProviderNotImplementedError` for any other value. Gemini added in v1 when its adapter is built.

**2. `ConfigLoader`**
- Load YAML → resolve env vars (${VAR_NAME} syntax)
- Validate against Pydantic config schema
- Unknown keys → validation error
- Missing required keys → error with field name + hint
- Output: `ResolvedConfig` + write `config.resolved.yaml` to run dir

**3. `PipelineRunner`**
- Load config → instantiate stages → run in order
- Stage interface: `run(records: Iterable[Record], ctx: StageContext) -> Iterable[Record]`
- After each stage: write Parquet, update SQLite, log count_in/count_out
- On failure: write partial Parquet, log stage + error, exit cleanly
- `--resume <run_id>`: reload last stage Parquet, skip completed stages

**4. `CheckpointManager`**
- SQLite tables: `runs`, `stage_runs`, `artifact_index`
- `save_stage(run_id, stage, path)` — index the Parquet file
- `load_stage(run_id, stage)` — return path if exists
- `list_runs()` — show all runs with status + stage counts

**5. `OutputWriter`**
- Read final stage Parquet → write JSONL (one record per line, payload only)
- Write `manifest.json`

**Acceptance criteria:**
- `python run.py --config config.yaml` runs end to end
- Bad config fails before any LLM call with field-level error
- LLM called once with test prompt, result in JSONL output
- SQLite has run record
- `--resume` skips completed stage, resumes from last checkpoint
- Rate limit triggers retry (test with mock)
- Auth error fails fast (test with bad key)

**Tests:** `test_llm_client.py`, `test_config_loader.py`,
`test_checkpoint.py`, `test_pipeline_runner.py`

**What you learn:** How to build a provider-agnostic LLM interface. How to
design a resumable, debuggable pipeline. How error taxonomies matter.

---

### SLICE 2 — LabelingEngine: Single Judge

**Tier:** A
**Goal:** Given any (instruction, response) pair and a rubric, produce a
structured score with per-dimension ratings and reasoning.

**What you build:**

**1. `Rubric` (Pydantic model)**
```python
class RubricDimension(BaseModel):
    name: str
    description: str
    scale_min: int
    scale_max: int
    criteria: dict[int, str]     # score level → description

class RubricExample(BaseModel):
    instruction: str
    response: str
    scores: dict[str, int]
    reasoning: str

class Rubric(BaseModel):
    version: str
    description: str
    dimensions: list[RubricDimension]
    overall_weights: dict[str, float]
    few_shot: list[RubricExample]

    @property
    def hash(self) -> str:
        return sha256(self.model_dump_json().encode()).hexdigest()[:16]
```

**Rubric YAML format:**
```yaml
version: "1.0"
description: "SFT instruction-response quality rubric"
dimensions:
  - name: instruction_clarity
    description: Is the instruction unambiguous and well-scoped?
    scale_min: 1
    scale_max: 5
    criteria:
      1: Unclear, multiple valid interpretations
      3: Mostly clear, minor ambiguity
      5: Perfectly clear, single interpretation
  - name: response_quality
    description: Accurate, complete, on-topic, appropriately concise?
    scale_min: 1
    scale_max: 5
    criteria:
      1: Wrong, incomplete, or off-topic
      3: Mostly correct, minor gaps
      5: Accurate, complete, well-calibrated length
overall_weights:
  instruction_clarity: 0.4
  response_quality: 0.6
few_shot:
  - instruction: "What is 2+2?"
    response: "4"
    scores: {instruction_clarity: 5, response_quality: 5}
    reasoning: "Clear simple question, correct answer."
  - instruction: "Tell me stuff"
    response: "Here is some stuff."
    scores: {instruction_clarity: 1, response_quality: 1}
    reasoning: "Vague instruction, non-answer response."
```

**2. `LabelResult` (Pydantic model)**
```python
class LabelResult(BaseModel):
    scores: dict[str, int]
    overall: float               # weighted average per rubric weights
    reasoning: str
    rubric_hash: str
    rubric_version: str
    judge_model: str
    judge_prompt_hash: str
    provider: str
    latency_ms: int
```

**3. `SingleLLMJudge`**
- Build prompt: system = rubric (dimensions + criteria + few-shot), user = example
- Call `complete_structured(schema=LabelResult)`
- Prompt template hash stored in every LabelResult
- Calibration: canary set runs every call to `label_batch()` — 5 known-good
  + 5 known-bad included, scores must fall in expected ranges or raised warning

**4. `LabelingEngine` (thin wrapper, extended in Slice 3)**
- `label(instruction, response, rubric) -> LabelResult`
- `label_batch(pairs, rubric, max_workers) -> list[LabelResult]`

**Note on position bias:** Not mitigated via order-swapping for scalar scoring
(that's for pairwise preference tasks). Mitigated via:
- Consistent prompt formatting (always same structure)
- Canary calibration on every batch
- Audit sampling: 5% of examples scored twice at temp=0, variance logged

**Acceptance criteria:**
- Known-good example → high score on both dimensions
- Known-bad example → low score on both dimensions
- Reasoning references rubric criteria by name
- Rubric hash consistent across runs for same rubric
- Canary warning fires when known-bad scores above threshold
- Invalid rubric (missing weight, unknown dimension) → clear error

**Tests:** `test_rubric.py`, `test_single_judge.py` (mock LLM),
`test_labeling_engine.py` (canary logic, batch)

**What you learn:** How LLM-as-judge works. How to design rubrics that produce
calibrated scores. What makes a judge trustworthy (canaries, not just prompts).

---

### SLICE 3 — LabelingEngine: Multi-Judge + Conflict Detection

**Tier:** A
**Goal:** Run the same example through multiple LLMs, aggregate with weighted
scoring, detect disagreements, export conflicts for human review.

**What you build:**

**1. `MultiLLMJudge`**
- Runs SingleLLMJudge for each configured (provider, model, weight) in parallel
- Failure tolerance: if one model errors, continue with remaining, log warning

**2. `VotingAggregator`**
```python
class AggregatedLabelResult(BaseModel):
    per_judge: list[LabelResult]   # raw per-judge results
    aggregate_score: float         # weighted average — soft signal only
    agreement_std: float           # std of overall scores across judges
    is_conflict: bool
    conflict_dimensions: list[str] # dimensions where std > threshold
    confidence: str                # high | medium | low — derived from agreement
```

- `aggregate_score` is a soft signal. Agreement std is surfaced equally.
- Primary use of multi-judge: triage + human review, not "more correct" scores
- Conflict threshold: `agreement_std > config.conflict_std_threshold` (default 1.0)

**3. `ConflictDetector`**
- Flags low-agreement examples
- Exports to `conflicts.jsonl` alongside main output
- Conflict record includes all per-judge scores + reasoning for human review

**4. `LabelingEngine` update**
- `mode: single | multi` in config
- Multi mode → MultiLLMJudge + VotingAggregator
- Conflict cases always exported regardless of mode

**Acceptance criteria:**
- Two judges with identical scores → agreement_std = 0, no conflict
- Two judges diverging by > threshold → conflict flagged, dimensions named
- Weighted aggregate gives higher-weight judge more influence (verify math)
- One judge erroring → falls back to available judges, logs warning
- Conflict JSONL written with all per-judge reasoning

**Tests:** `test_multi_judge.py`, `test_voting_aggregator.py`,
`test_conflict_detector.py`

**What you learn:** Where LLMs agree and where they diverge on quality
assessment. Why agreement is often more informative than the score itself.

---

### SLICE 4 — Data Sources

**Tier:** A
**Goal:** Load data from any source into normalized Records that generators consume.

**What you build:**

**1. `SeedLoader`**
- Loaders: CSV, JSONL, plain text
- All normalize to `ConversationRecord` with source type = `seed`
- Seed file hashed → stored in SQLite → every generated record carries `seed_file_hash`
- Column mapping configurable: which column is instruction, which is response

**2. `PDFIngester` (Tier 1: text-native PDFs only)**
- Extract: PyMuPDF (`fitz`) — better text fidelity than pypdf
- If page text is empty → `ScannedPDFError` (scanned PDFs unsupported in v0)
- Clean: strip headers/footers (line-position heuristic), fix hyphenation,
  normalize whitespace
- Chunk strategies:
  - `fixed`: N tokens per chunk, K token overlap (tokenizer-aware, not char-count)
  - `semantic`: split on sentence boundaries (spacy or simple regex), merge
    until size threshold
- Every chunk carries full provenance: `doc_id`, `page_start`, `page_end`,
  `char_start`, `char_end`, `source_hash`
- Output: list of `GroundedChunkRecord`

**3. `PersonaPool`**
- Load from JSONL (PersonaHub sample or custom)
- Stratified sampling across dimensions (profession, expertise, culture, age)
- If no pool provided: LLM generates N personas from domain description
- `sample(n, strategy: uniform | stratified) -> list[dict]`

**4. `HFDatasetLoader` (optional, Tier A)**
- Load from HuggingFace datasets library by identifier + split
- Normalize to same Record format

**Config:**
```yaml
data_source:
  type: pdf
  path: ./docs/my_doc.pdf
  chunk_strategy: semantic
  chunk_size_tokens: 512
  chunk_overlap_tokens: 64
```

**Acceptance criteria:**
- PDF with 10 pages → N chunks, each ≤ chunk_size tokens
- Each chunk has complete provenance fields (page, char range, source hash)
- Scanned PDF → clear error, not silent empty output
- Seed CSV with 100 rows → 100 Records with seed_file_hash set
- Persona pool stratified sampling: distribution across profession dimension
  within 20% of uniform

**Tests:** `test_pdf_ingester.py`, `test_seed_loader.py`,
`test_persona_pool.py`

**What you learn:** Why chunking strategy matters for downstream generation
quality. Why provenance has to be built in from the start, not retrofitted.

---

### SLICE 5 — Instruction Generation: Prompt-Based + Evol-Instruct

**Tier:** A
**Goal:** Generate (instruction, response) pairs from seeds or PDF chunks.
Two methods: direct prompt-based generation and Evol-Instruct evolution.

**What you build:**

**1. `PromptBasedGenerator`**
- Takes: seed records or chunk records + persona pool + topic taxonomy
- System prompt: persona + topic area + "generate a diverse, specific instruction"
- Generates instruction → then generates response in same conversation
- Persona conditioning: sample persona → inject into system prompt
- Topic enforcement: sample leaf node from taxonomy → include in prompt
- Output: `ConversationRecord` with lineage pointing to seed/chunk

**2. Evolution operators (Evol-Instruct)**

Each operator is a prompt template with a single responsibility:

- `AddConstraints`: "Rewrite this instruction to add 1-2 specific constraints
  that must be satisfied in the answer"
- `Deepen`: "Rewrite this instruction to require more domain expertise"
- `Concretize`: "Replace abstract terms with specific examples or scenarios"
- `IncreaseReasoningSteps`: "Rewrite to require multi-step reasoning"
- `BreadthMutation`: "Create a new instruction on a related but different topic"

**3. `EvolInstructGenerator`**
- `evolve_one(record, operator) -> ConversationRecord | None`
  - Returns None if evolution failed (identical, too short, refusal)
- `evolve_round(records, operators, branching) -> list[ConversationRecord]`
- `evolve_all(seeds, rounds, operators) -> list[ConversationRecord]`
  - Keeps ALL rounds, not just final — mix of depths = better training data
- Lineage: `operator`, `round`, `depth`, `parent_id`, `root_id` all set

**Failed evolution detection:**
- Edit distance < 20 chars from parent → discard
- Contains refusal keywords ("I cannot", "I'm unable", "As an AI") → discard
- Length < min_length tokens → discard
- Identical to parent after normalization → discard

**4. Topic taxonomy YAML**
```yaml
taxonomy:
  Machine Learning:
    Supervised Learning:
      - Classification
      - Regression
    Unsupervised Learning:
      - Clustering
      - Dimensionality Reduction
  Software Engineering:
    System Design:
      - Distributed Systems
      - API Design
```

**Config:**
```yaml
generator:
  type: evol_instruct
  target_count: 10000
  generation_multiplier: 5
  rounds: 4
  branching_factor: 2
  persona_conditioning: true
  persona_pool_path: ./personas.jsonl
  topic_taxonomy_path: ./taxonomy.yaml
  operators:
    depth: [add_constraints, deepen, increase_reasoning_steps]
    breadth: [breadth_mutation]
  filter:
    min_edit_distance: 20
    refusal_keywords: ["I cannot", "I'm unable", "As an AI"]
    min_length_tokens: 20
```

**Acceptance criteria:**
- 10 seeds × 4 rounds × 2 branching = up to 80 evolved records
- Each record has complete lineage (root_id, parent_id, operator, round, depth)
- Failed evolutions filtered: identical output, refusals, too short
- Instructions with persona conditioning cluster differently vs no personas
  (embed 100 each, measure mean pairwise distance — persona set should be higher)
- Mix of rounds: round 1 instructions shorter/simpler than round 4 on average

**Tests:** `test_operators.py`, `test_evol_filter.py`,
`test_evol_lineage.py`, `test_prompt_based_generator.py`

**What you learn:** How complexity is systematically manufactured via operators.
Why mixing rounds is better than keeping only deepest. How personas shift the
instruction distribution.

---

### SLICE 6 — Quality Filter Pipeline

**Tier:** A
**Goal:** Filter generated records from 5-10x volume down to target count.
Cheapest filters run first. Every dropped record has a reason code.

**Stage order (strictly enforced by cost):**

```
sanitize  →  cheap_filter  →  score  →  select
```

**1. Sanitize**
- `FormatValidator`: required fields present, types correct, no null payload fields
- Drop reason: `missing_field` | `null_payload`
- Zero LLM calls, zero cost

**2. Cheap Filter**
- `LengthFilter`: min/max tokens for instruction and response (tokenizer-aware)
  - Drop reason: `too_short` | `too_long`
- `LanguageFilter`: fastText language ID, filter non-target languages
  - Drop reason: `wrong_language`
- `HeuristicFilter`:
  - Alpha ratio < 0.6: drop (reason: `low_alpha_ratio`)
  - Any 4-gram appears > 3 times: drop (reason: `repetitive_ngrams`)
  - Special char density > 0.3: drop (reason: `high_special_chars`)

**3. Score**
- Current implementation note: exact dedup exists as its own stage before filters,
  and currently uses record `content_hash` (payload-based) rather than normalized
  instruction-only hashing.
- Target design: `ExactDeduplicator`: SHA-256 of normalized instruction text
  - Normalization: lowercase, strip whitespace and punctuation
  - Drop reason: `exact_duplicate`
- `IFDScorer` (optional, capability-gated):
  - Enabled only if `provider.capabilities.logprobs == True`
  - IFD = mean(log P(response_token | instruction + prev_tokens)) /
          mean(log P(response_token | prev_tokens))
  - High IFD = instruction meaningfully conditions response = high value
  - Drop reason: `low_ifd`
- `LabelingEngineFilter`:
  - Run LabelingEngine on remaining records
  - Drop records below `min_overall_score`
  - Drop reason: `low_quality_score`

**4. Select**
- Sort by composite score (weighted: quality score + IFD if available)
- Keep top `target_count`
- Log margin: score at cutoff vs score just below cutoff (small margin = tight cutoff)

**5. Drop reason tracking**
- Every dropped record written to `dropped.parquet` with:
  - all record fields
  - `drop_stage`: which stage dropped it
  - `drop_reason`: reason code
  - `drop_detail`: optional string
- This enables post-hoc analysis: why is my pipeline dropping too much?

**Config:** (see main config schema above)

**Acceptance criteria:**
- All known-bad examples (too short, wrong language, repetitive) dropped by
  stage 2 — zero LLM calls wasted on them
- IFD filter only activates when logprobs capability confirmed
- `dropped.parquet` exists and contains reason codes for all dropped records
- Final count ≤ target_count
- Stage yield table logged: count_in, count_out, drop_count per stage

**Tests:** `test_format_validator.py`, `test_length_filter.py`,
`test_language_filter.py`, `test_heuristic_filter.py`,
`test_ifd_scorer.py` (mock logprobs), `test_composite_select.py`

**What you learn:** IFD scoring internals (the log-probability math). Why
filter ordering is a cost discipline problem, not just a preference. How
drop-reason tracking turns a black box into a debuggable system.

---

### SLICE 7 — Deduplication Pipeline

**Tier:** A
**Goal:** Remove near-duplicate and semantically duplicate records. Three passes,
each removing what the previous missed.

**Three passes:**

**1. Exact dedup**

Current implementation:
- stage exists as `02c_exact_dedup`
- enabled via:

```yaml
dedup:
  exact:
    enabled: true
```

- keeps the first record for a given `content_hash`
- drops later duplicates with `reason_code = exact_duplicate`
- writes `dropped.parquet`, `clusters.parquet`, and `stats.json`

Planned refinement:
- switch from current payload-based `content_hash` behavior to normalized
  instruction-text hashing if/when the implementation is brought fully in line
  with the original spec wording

**2. Near-dedup (length-aware)**

For short instructions (< 50 tokens):
- Char n-gram SimHash (n=3, 64-bit hash)
- Hamming distance < 8 bits → near duplicate
- Good for: "What is X?" vs "What's X?" vs "Tell me about X"

For long instructions/responses (≥ 50 tokens):
- Token 5-gram MinHash
- 128 hash functions, LSH banding
- Jaccard threshold: 0.7
- Drop reason: `near_duplicate_minhash` | `near_duplicate_simhash`

**3. Semantic dedup (embedding-based)**

- Embed all instructions with text-embedding-3-small (batch, 512 per call)
- At <50K: brute force cosine similarity (chunked in batches of 1K to avoid
  full NxN in memory at once — NOT a naive dense matrix over all 50K)
- Cosine > 0.85 → near duplicate
- Cluster and keep representative: highest `scores.quality` within cluster
  (if no scores yet, keep longest as proxy)
- Duplicate clusters written to `clusters.parquet`
- Drop reason: `semantic_duplicate`

**Dedup runs on instruction text only** — two different responses to the same
instruction are still duplicates at the training example level.

**Dedup purposes explicitly separated:**
1. Cost-saving dedup (exact + near) — before LabelingEngine scoring
2. Training-diversity dedup (semantic) — after scoring, before select
3. Contamination dedup — separate audit pass (Slice 8)

Note: In the pipeline, near-dedup runs in stage 3 (before expensive scoring).
Semantic dedup runs after scoring so representative selection uses quality scores.

**Acceptance criteria:**
- "What is machine learning?" and "Can you explain machine learning?" →
  near-dup flagged (SimHash or MinHash depending on length)
- Semantically equivalent but lexically different → semantic dup flagged
- `clusters.parquet` contains cluster_id, all member record_ids, representative_id
- Representative is highest-quality in cluster (not arbitrary)
- Dedup stats per pass logged in stage stats

**Tests:** `test_simhash.py`, `test_minhash.py`,
`test_semantic_dedup.py` (mock embeddings with controlled cosine),
`test_dedup_pipeline.py` (full three-pass on synthetic duplicate set)

**What you learn:** How MinHash approximates Jaccard without all-pairs comparison.
Why char n-grams work better for short text. Why semantic dedup removes things
that lexical dedup misses entirely.

---

### SLICE 8 — Full SFT Pipeline + Dataset Report

**Tier:** A
**Goal:** Wire Slices 1-7 into a single end-to-end run. Add contamination check
and mandatory dataset report.

**New components:**

**1. `ContaminationChecker`**
- Two signals run independently — either can flag a record:
  - 13-gram overlap with eval benchmark → `contamination_ngram`
  - Embedding cosine > 0.9 with eval benchmark → `contamination_semantic`
- Contamination levels:
  - `lexical_suspect` — ngram only → flagged in report, kept in dataset
  - `semantic_suspect` — semantic only → flagged in report, kept in dataset
  - `confirmed` — both signals → removed, written to `dropped.parquet`
    with reason `contamination_confirmed`
- Thresholds (based on `confirmed` count only):
  - > 1% → warning in report
  - > 5% → pipeline error, dataset not written

**2. `ChatMLFormatter` and `AlpacaFormatter`**
- ChatML: `{"messages": [{"role": "user", "content": instr}, {"role": "assistant", "content": resp}]}`
- Alpaca: `{"instruction": ..., "input": "", "output": ...}`
- Format selected in config

**3. `DatasetReport` (mandatory — pipeline fails if report cannot be written)**

```json
{
  "run_id": "...",
  "config_hash": "...",
  "timestamp": "...",
  "stage_yields": [
    {"stage": "source", "count_in": 0, "count_out": 5000},
    {"stage": "generate", "count_in": 5000, "count_out": 50000},
    {"stage": "filter", "count_in": 50000, "count_out": 18000, "dropped": 32000},
    {"stage": "dedup", "count_in": 18000, "count_out": 12000},
    {"stage": "select", "count_in": 12000, "count_out": 10000}
  ],
  "quality_distribution": {
    "mean": 4.1, "std": 0.6, "min": 3.5, "max": 5.0,
    "histogram": {"3.5-4.0": 1200, "4.0-4.5": 5100, "4.5-5.0": 3700}
  },
  "length_distribution": {
    "instruction": {"mean": 48, "std": 22, "min": 11, "max": 210},
    "response": {"mean": 312, "std": 180, "min": 24, "max": 1980}
  },
  "diversity_score": 0.87,
  "contamination": {"ngram": 0.002, "semantic": 0.001, "status": "clean"},
  "canaries": {
    "known_good": [{"id": "...", "expected": "high", "actual_score": 4.8}],
    "known_bad": [{"id": "...", "expected": "low", "actual_score": 1.2}],
    "status": "pass"
  },
  "cost": {
    "total_usd": 4.21,
    "by_stage": {"generate": 3.10, "score": 1.11}
  },
  "drop_reasons": {
    "too_short": 8200, "low_quality_score": 15000, "semantic_duplicate": 4800,
    "contamination_ngram": 200
  }
}
```

**Diversity score:**
```python
embeddings = embed(all_instructions)
kmeans = MiniBatchKMeans(n_clusters=50)
labels = kmeans.fit_predict(embeddings)
counts = np.bincount(labels, minlength=50)
probs = counts / counts.sum()
entropy = -np.sum(probs * np.log(probs + 1e-10))
diversity_score = entropy / np.log(50)   # 1.0 = perfectly uniform
# < 0.7 = mode collapse warning in report
```

`samples.jsonl`: 20 random examples for human eyeball review — printed in
report, mandatory human review before training.

**Acceptance criteria:**
- Config in → `dataset.jsonl` + `run_report.json` + `samples.jsonl` out
- Canary results: known-bad scores low, known-good scores high (warnings if not)
- Diversity score calculated and present in report
- Cost tracked by stage (LLM usage summed from LLMOutput.usage)
- Drop reasons present and sum to total dropped count
- `--resume` works: kill after filter stage, resume, same final output
- Contamination rate > 5% → pipeline errors before writing dataset

**What you learn:** How to validate a dataset programmatically before training.
How to catch pipeline failures (mode collapse, broken judge, contamination)
without training a model.

---

### SLICE 9 — Preference / RL Data (Tier B)

**Goal:** Generate (chosen, rejected) preference pairs for DPO training.

**Methods:**

**Best-of-N:**
- For each instruction: generate N responses (4-8) in parallel
- Score all N with LabelingEngine
- Highest score = chosen, lowest score = rejected
- Margin = score(chosen) - score(rejected)
- Filter: margin < threshold (default 1.0) → discard pair (teaches nothing)

**Synthetic rejection:**
- Take a good response
- Prompt LLM: "Rewrite this to be subtly incomplete, slightly wrong, or less
  helpful while remaining plausible"
- Score degraded version → must score lower than original or discard pair
- Drop reason: `rejection_not_degraded` if score doesn't drop

**Output format:**
```python
class PreferencePayload(BaseModel):
    prompt: str
    chosen: str
    rejected: str
    margin: float
    rejection_method: str        # best_of_n | synthetic
```

**Margin distribution** logged in run_report — visualize before training.

---

### SLICE 10 — Embedding Training Data (Tier B)

**Goal:** Generate (query, positive, negative) triplets for embedding model training.
Hard negatives are the bottleneck — invest here.

**Pipeline:**
1. `QueryGenerator`: for each chunk, generate N plausible queries
2. `BM25NegativeMiner`: build BM25 index, retrieve top-K excluding positive
3. `HardNegativeLLMSelector`: LLM picks hardest-looking-but-wrong candidate
4. `FalseNegativeFilter`: LabelingEngine scores (query, candidate) — if high,
   candidate is actually relevant, discard as negative

**Output:**
```python
class TripletPayload(BaseModel):
    query: str
    positive: str
    negative: str
    task: str                    # qa_retrieval | semantic_sim | fact_check
    negative_type: str           # bm25 | llm_hard
```

---

### SLICE 11 — Reranker Training Data (Tier B)

**Goal:** Annotated candidate lists for cross-encoder training.

**Pipeline:**
1. Build candidate list (BM25 + generated)
2. LLM annotates each (query, candidate) on 0-3 scale
3. Construct pairwise pairs where score diff ≥ 1
4. Store full candidate list as canonical artifact, derive pointwise/pairwise/
   listwise training views from same source

---

### SLICE 12 — Evaluation Dataset (Tier B)

**Goal:** Diverse, calibrated, contamination-free evaluation set.

**Pipeline:**
1. `CapabilityTaxonomy`: config-driven dimensions + quota enforcement per leaf
2. `EvalPromptGenerator`: generate prompts per dimension + persona conditioning
3. `DifficultyCalibrator`:
   - Multi-signal: LLM self-rate + agreement across 2 judges + optional pilot
     model pass rate
   - NOT LLM self-rate alone (grill point — too weak as single signal)
4. Contamination check (reused from Slice 8)
5. Optional reference answers

---

### SLICE 13 — Magpie Generator (Tier C, capability-gated)

**Goal:** Generate instructions by feeding chat template prefixes to aligned models.

**Capability requirement:** Provider must support raw text completion semantics,
not just chat messages API. Fails fast with `MagpieBackendNotSupportedError` if
provider doesn't support it.

**Compatible backends (tested):** Together AI, Fireworks, Groq (model-dependent).
**Not compatible:** OpenAI chat API, Anthropic chat API (standard form).

**Template registry:**
```python
TEMPLATES = {
    "llama3": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
    "qwen": "<|im_start|>user\n",
    "chatml": "<|im_start|>user\n",
}
```

Config fails fast if `model_family` not in registry.

---

### SLICE 14 — Tool Use / Trajectory Data (Tier C)

**Goal:** (task, trajectory) pairs where trajectory = tool calls + results + final answer.

**Key design point from final verdict:** LLM-imagined tool results are
`trace_type: synthetic_unverified`. Do not mix with simulated/verified traces
without clear labeling. For v0, all trajectory data is `synthetic_unverified`
and labeled as such. Replayability (deterministic simulation) is Tier C+.

---

### SLICE 15 — Continuous Learning Flywheel (Tier C)

**Goal:** Human corrections → training signal. Three separate products from
one correction event (not collapsed into one):

**1. Judge calibration record**
- (input, llm_score, human_score, delta) → judge accuracy tracking
- SQLite: `judge_calibration` table

**2. Rubric issue record**
- (dimension, llm_score, human_score, notes) → clusters into rubric update suggestions
- SQLite: `rubric_issues` table
- Periodic: cluster by dimension + direction → surface "LLM consistently
  under-scores X, consider adding criterion Y"

**3. Preference pair**
- ONLY produced when two assistant outputs are being compared (pairwise)
- NOT from scalar correction alone
- Feeds into Slice 9 preference pair pipeline

Three SQLite tables. Not one collapsed corrections table.

---

## Implementation Order

| # | Slice | Tier | Depends on |
|---|-------|------|------------|
| 1 | Foundation | A | — |
| 2 | LabelingEngine: single judge | A | 1 |
| 3 | LabelingEngine: multi-judge | A | 2 |
| 4 | Data sources + PDF ingestion | A | 1 |
| 5 | Instruction generation | A | 1, 2, 4 |
| 6 | Quality filter pipeline | A | 1, 2 |
| 7 | Dedup pipeline | A | 1 |
| 8 | Full SFT end-to-end + report | A | 1-7 |
| 9 | Preference pairs | B | 8 |
| 10 | Embedding triplets | B | 8 |
| 11 | Reranker data | B | 10 |
| 12 | Evaluation dataset | B | 8 |
| 13 | Magpie | C | 1 |
| 14 | Trajectory data | C | 1, 4 |
| 15 | Continuous flywheel | C | 2, 3, 9 |

---

## Open Questions (Resolved)

| Question | Answer |
|----------|--------|
| IFD via API? | Capability-gated, OpenAI only, optional flag — not default |
| Semantic dedup at scale? | Brute-force chunked OK at <50K. ANN (hnswlib) when scale grows. |
| BM25? | rank-bm25 for v0, backend pluggable |
| SQLite scope? | Metadata only. Parquet for datasets. |
| Async? | Sync + ThreadPoolExecutor for v0. Executor mode in config as extension point. |
| Rubric versioning? | Hash on every score, mandatory from Slice 2. |
| Definition of done? | Run report passes for each slice. Full SFT pipeline (Slice 8) requires a small training smoke test. |
| Magpie? | Capability-gated Tier C. Not default. |
| Position bias mitigation? | Canary calibration + audit sampling. Not order-swapping for scalar tasks. |
| PDF scanned docs? | Unsupported in v0. Clear error. Tier 2 with OCR backend later. |
