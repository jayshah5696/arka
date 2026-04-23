# Tier A — Next Three Slices Execution Plan

**Date:** 2026-04-05  
**Status:** planning only — no implementation in this document  
**Priority:** execute from this plan  
**Explicit defer:** Anthropic provider adapter is out of scope for this plan

---

## Purpose

This document defines the **next three deep execution slices** for Arka from the current codebase state.

The goal is not a broad backlog review. The goal is to lock the next implementation train tightly enough that work can proceed end-to-end without re-deciding scope every session.

This plan assumes the current baseline already exists and is working:

- seed loading (`seeds` JSONL/CSV) ✔
- prompt-based generation ✔
- exact + near dedup ✔
- cheap filters (length, language) ✔
- single-judge labeling filter ✔
- StageBuilder wiring ✔
- run artifacts / dropped records / stats files ✔

---

## Tier A gaps still open

Out of the approved Tier A target, the major remaining gaps are:

- Evol-Instruct generator
- PDF data source
- Persona pool
- IFD scoring
- Contamination check stage
- Multi-LLM judge
- Anthropic adapter (**ignored here intentionally**)

---

## Decision: next three slices

The next three slices are:

1. **Slice A1 — Evol-Instruct generator**
2. **Slice A2 — IFD scoring filter**
3. **Slice A3 — PDF source stage**

### Why this order

#### 1) Evol-Instruct first
This is the most important remaining generator gap. Prompt-based generation exists, but it does not yet make Arka a canonical reference for synthetic instruction generation. Evol-Instruct is the first truly distinctive recipe missing from Tier A.

#### 2) IFD second
Once generation quality and diversity increase, filtering quality matters more. IFD is the cleanest next quality signal after cheap heuristics and before expensive LLM-judge scoring.

#### 3) PDF third
PDF ingestion is high-value, but it is a source expansion, not the core missing generation recipe. Once prompt-based and Evol-Instruct generation are both established, PDF unlocks grounded use cases cleanly.

### Important execution note

**IFD has one feasibility gate** because the current API stack is chat-completions oriented and does not yet expose a first-class response scoring interface. The plan below includes that gate explicitly. If that gate fails on real provider behavior, implementation should pause at the gate and PDF should move ahead temporarily.

That does **not** change the product priority. It only avoids burning time on an API assumption that may not hold.

---

## End state after these three slices

After these three slices, Arka should support this canonical Tier A path:

```text
Seeds OR PDF
   -> normalize / chunk
   -> prompt_based OR evol_instruct generation
   -> exact dedup
   -> near dedup
   -> cheap filters
   -> IFD filter (capability-gated)
   -> labeling quality filter
   -> final dataset + run report
```

That is the first version that starts to look like a serious reference implementation instead of a scaffold.

---

# Slice A1 — Evol-Instruct generator

## Goal

Add a first-class Evol-Instruct generator path that produces more diverse and harder instructions than prompt-based generation alone, while preserving full lineage and stage artifacts.

## Why now

Current `generator.type = prompt_based` is useful, but it is still the simplest recipe. The biggest Tier A product gap is the lack of a real instruction-evolution pipeline.

## Scope

This slice includes:

- `generator.type = evol_instruct`
- configurable rounds
- configurable branching factor
- four operators only for v1 of this slice:
  - `add_constraints`
  - `deepen`
  - `increase_reasoning_steps`
  - `breadth_mutation`
- full lineage tracking (`root_id`, `parent_ids`, `operator`, `round`, `depth`)
- per-round stage artifacts
- response generation for each evolved instruction
- failed-evolution detection and drops

This slice does **not** include:

- persona conditioning
- taxonomy conditioning
- PDF-grounded Evol-Instruct
- operator scoring/ranking by LabelingEngine before response generation
- multi-turn conversations
- Magpie / seedless generation

## Key design decisions

### 1. Keep prompt-based generation as the baseline path
Do not replace `prompt_based`. Add `evol_instruct` alongside it.

### 2. Each Evol round is a pipeline stage
This matches the intended architecture better than hiding all rounds inside one monolithic stage.

Why:
- resume/checkpointing is cleaner
- per-round yields are inspectable
- failures are localized
- lineage by round is easier to audit

### 3. Preserve all rounds, not just final outputs
The stage should return prior records plus newly generated children so the final dataset contains a mix of difficulty/depth levels.

### 4. Frontier-only evolution
Round `r+1` evolves only records created in round `r`, not the entire accumulated set.

### 5. Output type
Use `ConversationRecord` outputs.

Recommended source typing:
- prompt-based outputs stay `source.type = "generated"`
- evol outputs use `source.type = "evolved"`

This keeps provenance clearer than collapsing everything into `generated`.

## Pipeline shape

For `generator.type = evol_instruct`:

```text
01_source
02_normalize
03_evol_round_01
04_evol_round_02
05_evol_round_03
...
then dedup / filters
```

StageBuilder should create one stage per configured round.

## Config additions

Add the following shape to `GeneratorConfig`:

```yaml
generator:
  type: evol_instruct
  target_count: 10000
  generation_multiplier: 5
  rounds: 4
  branching_factor: 2
  operators:
    - add_constraints
    - deepen
    - increase_reasoning_steps
    - breadth_mutation
  filter:
    min_edit_distance_chars: 20
    min_instruction_chars: 20
    refusal_keywords:
      - "I cannot"
      - "I'm unable"
      - "As an AI"
```
```

### Validation rules

- `rounds >= 1`
- `branching_factor >= 1`
- `operators` non-empty
- unknown operator names fail validation
- unsupported `generator.type` still raises `ValueError` in `StageBuilder`

## Record semantics

Each evolved record should carry:

- `source.type = "evolved"`
- `lineage.root_id = original seed root`
- `lineage.parent_ids = [direct_parent_record.id]`
- `lineage.operator = selected operator`
- `lineage.round = current round number`
- `lineage.depth = current round number`

## Minimal operator contract

Create one operator interface that turns a parent instruction into a new instruction.

```python
class EvolOperator(Protocol):
    name: str
    def build_messages(parent: ConversationRecord) -> list[Message]: ...
```

Each operator should only rewrite the **instruction**. The response is generated in a second sub-step using the existing prompt-based response generation pattern.

## Failure detection rules

Drop a candidate evolution when any of these hold:

- normalized instruction identical to parent
- character edit distance below threshold
- refusal keyword present
- evolved instruction below minimum length
- parse failure from operator output
- parse failure from response generation output

Recommended drop reasons:

- `evol_parse_failure`
- `evol_identical_to_parent`
- `evol_edit_distance_too_small`
- `evol_refusal`
- `evol_instruction_too_short`
- `evol_response_parse_failure`

## Recommended implementation structure

### Files to add / change

- `src/arka/config/models.py`
  - extend `GeneratorConfig`
- `src/arka/pipeline/stage_builder.py`
  - branch on `generator.type == "evol_instruct"`
  - emit one stage per round
- `src/arka/pipeline/generator_stages.py`
  - keep prompt-based stage
  - add shared base helpers if useful
- `src/arka/pipeline/evol_instruct.py` or `src/arka/pipeline/evol_operators.py`
  - operator prompts + helper functions
- `tests/unit/test_generation_stages.py`
  - extend or split into focused files
- `docs/config-examples.md`
  - add `evol_instruct` example

## TDD execution order

### Step 1 — config validation
Write tests that:
- accept valid `evol_instruct` config
- reject unknown operator names
- reject zero rounds / zero branching

### Step 2 — StageBuilder wiring
Write tests that:
- build N evol round stages for `rounds = N`
- preserve downstream stage ordering
- raise on unsupported generator types

### Step 3 — single-round operator success path
Write tests that:
- evolve one seed into one new instruction/response pair
- set lineage correctly
- mark output as `source.type = "evolved"`

### Step 4 — failed evolution filtering
Write tests that:
- drop identical rewrites
- drop refusal text
- write `dropped.parquet` and `stats.json`

### Step 5 — multi-round accumulation
Write tests that:
- round 2 evolves only round 1 outputs
- final record list contains earlier-round records plus children
- stage stats reflect frontier-in and generated-out accurately

### Step 6 — checkpoint/resume
Write tests that:
- resume from a completed earlier round
- do not regenerate already completed raw outputs

## Acceptance criteria

- `generator.type = evol_instruct` works end-to-end from seeds to output
- per-round stage folders are written
- lineage is complete and correct on every evolved record
- failed evolutions are dropped with reason codes
- all rounds are preserved in the output stream before downstream filters
- tests cover parse failures, identical rewrites, refusal filtering, and resume

## Done means

A run with 10 seeds, `rounds=2`, `branching_factor=2` produces a non-empty evolved dataset, writes per-round artifacts, and passes full test + lint.

---

# Slice A2 — IFD scoring filter

## Goal

Add an optional, capability-gated IFD filter stage that scores whether the response meaningfully depends on the instruction, then drops low-IFD examples before expensive judge scoring.

## Why now

Once Evol-Instruct lands, output volume and complexity go up. Cheap heuristics are not enough. IFD is the cleanest next quality signal because it captures whether the instruction is actually doing work.

## Scope

This slice includes:

- IFD filter stage (add `{type: ifd}` to `filters.stages` list)
- per-record `scores.ifd`
- stage artifact writing (`dropped.parquet`, `stats.json`)
- capability gating
- run-time failure if provider doesn't support scoring

This slice does **not** include:

- pricing lookup tables
- semantic quality scoring
- judge aggregation changes
- open-ended fallback approximations if scoring is unsupported

## Critical feasibility gate

Before full implementation, prove the system can score a fixed response sequence with logprobs through the chosen provider path.

### Required outcome

A minimal spike must show that, for one real provider/model combination already used by Arka, we can obtain token-level logprobs for a supplied target response under:

1. conditioned context: `instruction + response prefix`
2. unconditioned context: `response prefix` only

### If this is not possible

Do **not** implement a fake IFD approximation.

Instead:
- stop the slice at the feasibility gate
- document the provider limitation
- move PDF slice ahead temporarily

## Formula

For response tokens `r_1 ... r_n`:

```text
IFD = mean(log P(r_i | instruction, r_<i)) - mean(log P(r_i | r_<i))
```

Equivalent ratio-based variants are acceptable internally, but the stage should surface one stable scalar score in `scores.ifd`.

## Stage placement

Place IFD **after** dedup + cheap filters and **before** label quality.

Recommended order:

```text
generate/evolve
-> exact dedup
-> near dedup
-> length
-> language
-> ifd
-> label_quality
```

## Config additions

Add something like:

```yaml
filters:
  target_count: 10000
  stages:
    - type: ifd
      min_score: 0.2
```

Validation behavior:
- if `ifd` is absent from the stages list, no gating needed
- if `ifd` is present but provider/model cannot support scoring, fail fast

## LLM client/API design

Current `LLMClient` is not designed for teacher-forced sequence scoring.

Add a dedicated scoring API rather than overloading `complete()`:

```python
def score_response(
    self,
    *,
    messages: Sequence[Message],
    target_text: str,
) -> SequenceScore:
    ...
```

Recommended boundary model:

```python
class SequenceScore(BaseModel):
    token_count: int
    mean_logprob: float
    total_logprob: float
    provider: str
    model: str
```
```

Do **not** leak provider-specific logprob payloads into pipeline stages.

## Stage design

Add `IFDFilterStage`:

Inputs:
- `ConversationRecord`s only

Per record:
1. score response with instruction context
2. score response without instruction context
3. compute `ifd`
4. attach `scores.ifd`
5. drop if below threshold

Outputs:
- kept records with `scores.ifd` set
- dropped records with `reason_code = low_ifd`
- stats with score distribution

Recommended additional stats:
- `scored_count`
- `ifd_distribution.mean`
- `ifd_distribution.std`
- `ifd_distribution.min`
- `ifd_distribution.max`

## Recommended implementation structure

### Files to add / change

- `src/arka/config/models.py`
  - add IFD filter config
- `src/arka/llm/models.py`
  - add `SequenceScore` boundary model
- `src/arka/llm/client.py`
  - add scoring API + capability checks
- `src/arka/pipeline/filter_stages.py` or new `src/arka/pipeline/ifd_stage.py`
  - implement `IFDFilterStage`
- `src/arka/pipeline/stage_builder.py`
  - insert IFD stage before label quality
- `tests/unit/test_llm_client.py`
  - scoring API tests
- `tests/unit/test_ifd_scorer.py`
  - IFD formula + gating tests
- `tests/unit/test_stage_builder.py`
  - ordering tests

## TDD execution order

### Step 0 — feasibility test/spike
Before all else, create a minimal test harness or manual verification path for the scoring API assumption.

### Step 1 — config + StageBuilder
Write tests for:
- IFD stage present in `filters.stages` list
- stage insertion ordering
- fail-fast on unsupported capability when IFD is in the list

### Step 2 — LLM client scoring API
Write tests for:
- conditioned score extraction
- unconditioned score extraction
- provider capability rejection

### Step 3 — IFD formula
Write tests that verify:
- higher instruction dependence -> higher IFD
- records get `scores.ifd`

### Step 4 — stage artifact behavior
Write tests for:
- dropped parquet written
- stats include IFD distribution
- drop reason counts match dropped rows

### Step 5 — pipeline integration
Write tests that verify:
- IFD runs before labeling stage
- low-IFD examples never reach judge scoring

## Acceptance criteria

- enabling IFD inserts the stage in the correct place
- unsupported backends fail fast with clear message
- supported backends compute and store `scores.ifd`
- low-IFD records drop with reason code `low_ifd`
- stage stats include IFD distribution
- full pipeline still passes with IFD disabled

## Done means

A run with `{type: ifd}` in `filters.stages` either:
- works end-to-end on a supported backend and writes IFD stats, or
- fails immediately with a precise unsupported-capability error

No silent fallback.

---

# Slice A3 — PDF source stage

## Goal

Support text-native PDF ingestion so Arka can generate grounded synthetic data from private documents, not just seed conversation files.

## Why now

PDF is the highest practical-value source expansion for real-world use. Once generation and filtering improve, grounded data becomes the next major unlock.

## Scope

This slice includes:

- `data_source.type = pdf`
- text-native PDF extraction only
- chunking
- chunk provenance
- StageBuilder support for PDF source
- prompt-based generation from PDF chunks

This slice does **not** include:

- OCR for scanned PDFs
- PDF + Evol-Instruct in the same slice
- table extraction
- image-based PDFs
- persona conditioning

## Key design decisions

### 1. Add a real grounded record type
The current codebase has `ConversationRecord` but not the spec’s grounded chunk record implementation.

Implement:
- `GroundedChunkPayload`
- `GroundedChunkRecord`
- registry wiring in `records/models.py`

### 2. PDF source should feed prompt-based generation first
Do **not** couple PDF support to Evol-Instruct immediately.

Thin-slice rule:
- PDF -> chunk -> prompt_based generation
- PDF -> Evol-Instruct can come later

### 3. StageBuilder should branch cleanly
For PDF source:
- no `NormalizeConversationStage`
- use a dedicated `PDFSourceStage`

### 4. Scanned PDFs fail loudly
If extracted text is empty or effectively empty, raise a clear error. Do not silently emit zero chunks.

## Source stage behavior

`PDFSourceStage` should:

1. read PDF
2. extract text page by page
3. clean text minimally
4. split into chunks
5. emit `GroundedChunkRecord`s with provenance

Required provenance:
- `doc_id`
- `page_start`
- `page_end`
- `char_start`
- `char_end`
- `source_hash`

## Config additions

Extend `DataSourceConfig` to support:

```yaml
data_source:
  type: pdf
  path: ./docs/source.pdf
  chunk_strategy: fixed
  chunk_size_chars: 3000
  chunk_overlap_chars: 300
```
```

If token-aware chunking is easy and already aligned with existing tooling, use tokens. If not, start with character-based chunking as the thin slice and document it clearly.

## Generator compatibility

Current `PromptBasedGeneratorStage` expects `ConversationRecord` seeds.

For this slice, broaden it to accept:
- `ConversationRecord` (existing seed path)
- `GroundedChunkRecord` (new PDF path)

Prompt shape for PDF grounding should be different from seed-pair prompting.

Recommended generator prompt contract:
- given a grounded chunk
- generate one self-contained instruction answerable from the chunk
- generate the response grounded in the chunk
- avoid references like “according to the passage above”

## Recommended implementation structure

### Files to add / change

- `src/arka/config/models.py`
  - extend `DataSourceConfig`
- `src/arka/records/models.py`
  - add grounded chunk payload/record types
- `src/arka/pipeline/source_stages.py`
  - add `PDFSourceStage`
- `src/arka/pipeline/stage_builder.py`
  - branch for `data_source.type == "pdf"`
- `src/arka/pipeline/generator_stages.py`
  - broaden prompt-based generator input handling
- `tests/unit/test_pdf_source_stage.py`
  - new tests
- `tests/unit/test_stage_builder.py`
  - stage composition tests
- `docs/config-examples.md`
  - add PDF example

## TDD execution order

### Step 1 — record model tests
Write tests for:
- `GroundedChunkPayload`
- `GroundedChunkRecord`
- registry round-trip through parquet/json paths if needed

### Step 2 — StageBuilder routing
Write tests that:
- `data_source.type = pdf` builds `PDFSourceStage`
- normalize conversation stage is skipped for PDFs
- unsupported source types still raise clearly

### Step 3 — PDF extraction happy path
Write tests with a small text-native sample PDF that:
- emits non-empty chunks
- sets provenance fields

### Step 4 — scanned/empty failure path
Write tests that:
- empty extracted text raises a clear error

### Step 5 — prompt-based generator compatibility
Write tests that:
- prompt-based generator accepts `GroundedChunkRecord`
- generated records from PDF chunks use grounded lineage/source fields correctly

## Acceptance criteria

- `data_source.type = pdf` works end-to-end with prompt-based generation
- chunks carry provenance
- scanned/empty PDFs fail loudly
- StageBuilder composition remains clean
- prompt-based generator can consume both seed records and grounded chunk records

## Done means

A text-native PDF can be ingested, chunked, turned into generated conversation pairs, then passed through the existing dedup/filter pipeline.

---

# Cross-slice engineering rules

These rules apply to all three slices.

## 1. StageBuilder is the only wiring authority
Do not leak stage assembly back into CLI.

## 2. Preserve typed records
Do not push raw dicts through the pipeline to “move faster.”

## 3. Every stage owns its artifacts
If a stage drops records or computes stats, it writes:
- `dropped.parquet` when relevant
- `stats.json`

## 4. Fail fast on unsupported config combinations
Examples:
- `generator.type = evol_instruct` with no operators
- `{type: ifd}` in `filters.stages` on unsupported backend
- `data_source.type = pdf` with missing path

## 5. Thin slice over completeness
Do not drag persona pool, contamination, or multi-judge into these slices.

---

# What is explicitly not in this execution train

Not part of the next three slices:

- Anthropic adapter
- persona pool
- contamination check stage
- multi-LLM judge
- preference pairs
- embedding triplets
- reranker annotation
- Magpie
- tool-use trajectories

These remain important, but they should not distract the next execution train.

---

# Recommended execution order in practice

## Phase 1
Implement **Slice A1 — Evol-Instruct** fully.

## Phase 2
Start **Slice A2 — IFD** with the feasibility gate first.

- if the scoring backend is viable, continue the slice
- if not, freeze the slice at that gate and move immediately to PDF

## Phase 3
Implement **Slice A3 — PDF source stage**.

---

# Definition of success for this plan

This plan is successful if the next implementation cycle delivers:

1. a real Evol-Instruct path,
2. either a real IFD filter or a precise documented block at the feasibility gate,
3. a working PDF source path feeding prompt-based generation,
4. no scope drift into persona pool / contamination / multi-judge / Anthropic.

That is the right next end-to-end train from the current repository state.
