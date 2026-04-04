# Synth Data Playbook — Overall Plan

Last updated: April 2026

---

## ARCHITECTURE OVERVIEW

```
config.yaml
    │
    ▼
ConfigLoader (Pydantic validation)
    │
    ▼
PipelineRunner
    ├── DataSource        ← PDF / CSV / seeds / Magpie seedless
    ├── Generator         ← Magpie / Evol-Instruct / Backtranslation / Persona
    ├── QualityFilter     ← heuristics → IFD → LabelingEngine → RM score
    ├── Deduplicator      ← exact → MinHash → semantic
    ├── Formatter         ← ChatML / Alpaca / triplet / preference pair
    └── OutputWriter      ← JSONL / HF dataset / CSV
         │
         ▼
      dataset/
```

Each box = one module. Each module = one Python class with a clean interface.
LLM calls go through one shared LLMClient. All stages independently testable.

---

## LAYER 0 — Foundation (build first, everything depends on this)

### 0a. LLM Client Wrapper
- Single class: `LLMClient(provider, model, api_key)`
- Methods: `complete()`, `complete_structured(schema)`, `complete_batch()`
- Provider swappable via config: openai / anthropic / gemini
- Structured output: JSON mode or tool-use depending on provider
- Automatic retry, rate limiting, token counting

### 0b. Config System
- YAML schema for a full pipeline
- Pydantic models for validation — bad config fails before anything runs
- Supports: use_case, data_source, generator, filters, dedup, output
- Composable — mix and match stages

### 0c. Pipeline Runner
- Loads config → instantiates stages → runs in order → writes output
- Each stage: `run(dataset) -> dataset`
- Streaming friendly — processes in batches, not all in memory
- Checkpointing — resume from any stage if interrupted

---

## LAYER 1 — LabelingEngine (build second, used by everything)

### What it does
Given any input + rubric → structured label + confidence + reasoning

### Components
- RubricParser — takes natural language rules + few-shot examples → structured rubric
- MultiLLMJudge — sends same input to N models, collects structured scores
- VotingAggregator — weighted voting, agreement score, confidence interval
- ConflictDetector — flags low-agreement examples for human review

### Techniques implemented
- Single LLM judge with rubric (simplest)
- Multi-LLM weighted voting
- Position bias mitigation (swap order, average)
- Reference-guided scoring (score against a gold example, not absolute)

---

## LAYER 2 — Data Sources

### 2a. Seed Manager
- Load seeds from: CSV, JSONL, HF dataset, plain text, PDF
- PDF ingestion: parse (pypdf/pdfminer) → clean → chunk (fixed / semantic)
- Versioned seed sets — track which seeds generated which examples

### 2b. Persona Pool
- Load from PersonaHub sample or generate from scratch
- Stratified sampling across dimensions (profession, age, expertise, culture)
- Persona → instruction conditioning

---

## LAYER 3 — Generation Techniques (one per use case)

### 3a. Instruction Tuning
- Magpie — feed chat template prefix, model autocompletes user query
- Evol-Instruct — depth operators (constrain, deepen, complicate) + breadth (mutate)
- Backtranslation — document → "what instruction leads to this answer?"
- Persona-conditioned — sample persona → generate instruction in that voice

### 3b. Preference / RL Data
- Best-of-N — generate N responses, LabelingEngine scores, top=chosen bottom=rejected
- Rejection sampling — score all, keep margin > threshold
- Constitutional self-critique — model critiques + revises own output, before/after = pair
- Synthetic rejection — prompt-steered degradation to produce plausibly bad responses

### 3c. Embedding Data
- Task-diverse triplet generation — (task description, query, document)
- Hard negative generation — BM25 candidates → LLM picks hardest
- Cross-encoder denoising — remove false negatives from mined set

### 3d. Reranker Data
- Candidate list construction (BM25 + generated queries)
- LLM relevance annotation (0-3 scale, structured output)
- Pairwise construction from relevance scores

### 3e. Evaluation Data
- Diverse prompt generation across capability dimensions
- Difficulty calibration — LLM self-rates, filter to target difficulty band
- Contamination check against known benchmarks

### 3f. Trajectory / Tool Use Data
- Task definition → agent steps → tool calls → final answer
- LLM generates plausible tool call sequences given a task
- Validation: does trajectory actually solve the task?

### 3g. PDF / Private Doc → Any of the above
- Parse + chunk doc → feed chunks into any generation technique above
- QA pairs, summaries, instructions — all grounded in your content

---

## LAYER 4 — Quality Pipeline

### Stages (in order, cheapest first)
1. Heuristic filters — length, language, format, char ratio
2. Exact dedup — SHA-256 hash
3. Near-dedup — MinHash LSH (5-gram, Jaccard 0.7)
4. IFD scoring — P(response|instruction) / P(response) — keep high IFD
5. LabelingEngine scoring — rubric-based quality score
6. Semantic dedup — embed + cosine similarity > 0.85 → remove
7. Contamination check — n-gram overlap against eval benchmarks
8. Diversity audit — cluster entropy, topic coverage vs taxonomy

---

## LAYER 5 — Continuous Flywheel

- Human corrections captured as structured annotations
- Correct label vs LLM label → (chosen, rejected) preference pair
- Pairs feed back into Layer 3b (preference data pipeline)
- Rule extraction — patterns in corrections → updated rubric for LabelingEngine

---

## EXECUTION ORDER (thin slices)

| Slice | What | Deliverable |
|-------|------|-------------|
| 1 | Layer 0 — LLM client + config + runner skeleton | Config loads, LLM call works, stage interface defined |
| 2 | Layer 1 — LabelingEngine, single judge | Score any (input, output) against a rubric |
| 3 | Layer 1 — Multi-LLM voting + conflict detection | Weighted vote, confidence, flag conflicts |
| 4 | Layer 2 — Seed manager + PDF ingestion | Load seeds, parse PDF, chunk, version |
| 5 | Layer 3a — Magpie instruction generation | Generate 100 instructions from scratch |
| 6 | Layer 4 — Heuristic + IFD + LabelingEngine filter | Filter 100 → quality subset |
| 7 | Layer 4 — Dedup (exact + MinHash) | Dedup the filtered set |
| 8 | Full SFT slice end-to-end | Config in → JSONL dataset out |
| 9 | Layer 3b — Preference pairs (Best-of-N) | Chosen/rejected pairs with margin filter |
| 10 | Layer 3c — Embedding triplets + hard negatives | Training-ready embedding dataset |
| 11 | Layer 3d — Reranker data | Annotated candidate lists |
| 12 | Layer 3e — Evaluation dataset | Diverse, calibrated, decontaminated |
| 13 | Layer 3f — Trajectory data | Tool call sequences |
| 14 | Layer 3g — PDF grounded generation | Any use case from private docs |
| 15 | Layer 5 — Continuous flywheel | Corrections → preference pairs loop |

---

## OPEN QUESTIONS (to resolve before slice 1)

1. Output format — JSONL as primary, with converters to Alpaca / ChatML / HF dataset format?
2. Async vs sync — async from the start (better for API rate limit handling) or sync first then async?
3. Checkpointing — file-based (JSONL append) or SQLite?
4. Evaluation — how do we know a generated dataset is good before training on it? What's the smoke test?
