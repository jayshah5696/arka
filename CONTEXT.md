# Arka — Synthetic SFT Data Generation

Arka (अर्क) is a config-driven framework that takes raw seed material and produces deduplicated, quality-filtered datasets for supervised fine-tuning, preference learning, embeddings, and reranker training. This document defines the ubiquitous language used across configs, code, docs, and conversations about the system.

## Language

### Inputs & artifacts

**Seed**:
A pre-existing example (instruction/response pair, document chunk, persona, etc.) that bootstraps generation.
_Avoid_: Sample, prompt (when referring to source material), input row.

**Data Source**:
The configured origin of seeds for a run — `seeds` (JSONL/CSV) or `pdf` (chunked document).
_Avoid_: Loader, ingestor, dataset (the input side).

**Document Chunk**:
A bounded span of text extracted from a parsed document, identified by `doc_id`, `chunk_idx`, and char/page offsets.
_Avoid_: Passage, fragment, segment.

**Record**:
The canonical typed pipeline unit (`ConversationRecord`, `GroundedChunkRecord`, …) carrying `payload`, `source`, `lineage`, `scores`, and `stage_events`. Every stage consumes and emits Records.
_Avoid_: Row, item, example (when speaking about the in-pipeline value), entry.

**Example**:
A user-facing instruction/response pair as it appears in a final exported **Dataset**. A **Record** becomes an **Example** only after it survives all stages and is serialized.
_Avoid_: Conversation (when used loosely), training pair.

**Dataset**:
The final exported file (JSONL / ChatML / Alpaca) consumed by an SFT trainer.
_Avoid_: Output, corpus, training set.

**Artifact**:
Any file written by a stage to the **Run Directory** — typically `data.parquet`, `dropped.parquet`, `stats.json`, or `clusters.parquet`.
_Avoid_: Output (overloaded), result file.

### Pipeline structure

**Pipeline**:
The full ordered sequence of stages defined by a YAML config and executed for a single run.
_Avoid_: Workflow, graph, DAG, job.

**Stage**:
A single typed transformation in the pipeline (e.g. source, generator, dedup, filter, scorer, output). Every stage is independently testable and produces its own artifacts.
_Avoid_: Step, node, task, phase.

**Run**:
One end-to-end execution of the pipeline, identified by a `run_id` and rooted at a **Run Directory**.
_Avoid_: Job, invocation, batch.

**Run Directory**:
The on-disk root for a single run (`runs/<run_id>/`) holding per-stage artifacts and the checkpoint database.
_Avoid_: Workspace, output dir, work dir.

**Checkpoint**:
A SQLite-backed snapshot of stage state that lets an interrupted run resume from the last completed stage with the same `run_id`.
_Avoid_: Save, snapshot (when ambiguous), state file.

**Manifest**:
The JSON summary of a finished run (stages, counts, drop reasons, costs) — distinct from per-stage `stats.json`.
_Avoid_: Report (overloaded), log.

### Generation

**Generator**:
A stage that synthesizes new **Records** by calling an LLM. Concrete strategies are **Prompt-Based Generation** and **Evol-Instruct**.
_Avoid_: Synthesizer, producer, creator.

**Prompt-Based Generation**:
A generation strategy that fills a single template with a seed and asks the LLM for one new example per call.
_Avoid_: Templated generation, naive generation.

**Evol-Instruct**:
A multi-round generation strategy that applies **Operators** to a seed instruction to increase complexity, depth, or specificity over successive **Rounds**.
_Avoid_: Mutation pipeline, evolution (standalone).

**Operator**:
A named Evol-Instruct mutation (e.g. `add_constraints`, `deepen`, `concretize`) applied to an instruction in a single round.
_Avoid_: Mutator, transformer, evolver.

**Round**:
One iteration of Evol-Instruct in which every active instruction has Operators applied to it.
_Avoid_: Generation, pass, epoch.

**Generation Multiplier**:
The over-generation factor — Arka generates `target_count × generation_multiplier` candidates so that aggressive filtering can still hit the target.
_Avoid_: Oversampling factor, fan-out.

**Target Count**:
The desired number of surviving Records the run aims to produce.
_Avoid_: Goal, quota, sample size.

**Lineage**:
The provenance metadata on a Record (`root_id`, `parent_ids`, `operator`, `round`, `depth`) that traces it back to its originating seed.
_Avoid_: Ancestry, history (overloaded), trace.

### Deduplication

**Exact Dedup**:
A hash-based stage that drops Records whose instruction or response is byte-for-byte identical to one already kept.
_Avoid_: Hash filter, dupe drop.

**Near Dedup**:
A MinHash + LSH-band stage that drops Records whose Jaccard similarity to a kept Record exceeds the configured threshold.
_Avoid_: Fuzzy match, soft dedup, semantic dedup (which means embedding-based).

**LSH Band**:
A bucket of MinHash signature slices used by Near Dedup to find candidate duplicates in O(n) average time without all-pairs comparison.
_Avoid_: Bucket, shard, hash band.

### Quality & filtering

**Filter**:
A stage that drops Records failing a heuristic or scored predicate (length, language, IFD, canary, similarity).
_Avoid_: Validator, gate, check.

**Filter Stack**:
The ordered list of Filters under `filters.stages` that a Record traverses before output.
_Avoid_: Chain, gauntlet, pipeline (overloaded).

**Judge**:
An LLM acting as an evaluator that scores a Record against a **Rubric** and returns structured per-dimension scores.
_Avoid_: Critic, evaluator, grader (when ambiguous), reviewer.

**Labeling Engine**:
The subsystem that orchestrates one or more Judges, applies the Rubric, computes consensus, and attaches scores to Records.
_Avoid_: Scorer (overloaded), annotator, judge runner.

**Rubric**:
A versioned YAML document defining the dimensions, scoring scale, and pass criteria a Judge must apply.
_Avoid_: Criteria, prompt (when referring to the rubric itself), schema.

**IFD (Instruction Following Difficulty)**:
A scalar score estimating how difficult an instruction is to follow; used as a filter threshold to drop trivial or pathological examples.
_Avoid_: Difficulty score (ambiguous), hardness.

**Canary Filter**:
A privacy filter that drops Records containing configured secret phrases that must never leak into training data.
_Avoid_: Secret scan, leak filter.

**Drop Reason**:
A short code (`length_min`, `lang_mismatch`, `judge_below_min`, `canary_hit`, …) attached to a dropped Record explaining which Filter or Stage rejected it.
_Avoid_: Reject reason, error, failure code.

### LLM access

**LLM Client**:
The single Arka-owned wrapper around an OpenAI-compatible API; every Generator, Judge, and IFD call goes through it.
_Avoid_: Provider client, API wrapper, model client.

**Provider**:
The configured backend serving the LLM (`openai`, `openrouter`, …) — distinct from the **Model**.
_Avoid_: Vendor, backend, host.

**Model**:
The specific LLM identifier requested from a Provider (e.g. `google/gemini-3.1-flash-lite-preview`).
_Avoid_: Engine, deployment.

**Structured Output**:
A provider-enforced JSON-schema response from the LLM, guaranteeing parseability for Generators and Judges.
_Avoid_: JSON mode, schema response, function call.

## Relationships

- A **Run** executes one **Pipeline** end-to-end against one **Run Directory**.
- A **Pipeline** is an ordered list of **Stages**; every Stage emits **Artifacts** and updates the **Checkpoint**.
- A **Data Source** produces **Records** from **Seeds** (or from **Document Chunks** when sourcing PDFs).
- A **Generator** consumes seed Records and emits new Records, attaching **Lineage** that points back to the originating **Seed**.
- **Evol-Instruct** applies one or more **Operators** per **Round**; each Operator application bumps `lineage.depth`.
- **Exact Dedup** runs before **Near Dedup**; both produce **Drop Reasons** on dropped Records.
- A **Filter Stack** runs after dedup; the **Labeling Engine** invokes one or more **Judges** with a **Rubric** and writes scores into `RecordScores`.
- A surviving **Record** is serialized into an **Example** within the final **Dataset**.
- The **LLM Client** is the only component that talks to a **Provider**; Generators and Judges request a specific **Model** through it.
- **Generation Multiplier × Target Count** sets how many candidates the Generator produces; the Filter Stack reduces them back toward **Target Count**.

## Example dialogue

> **Dev:** "We're hitting the rate limit during a run. If I bump `max_workers`, can I just resume from the **Checkpoint**?"

> **Domain expert:** "Yes — re-run with the same `run_id` and the **Run** picks up at the last completed **Stage**. The **Checkpoint** is the source of truth, not the **Artifacts**."

> **Dev:** "Okay. The **Generator** emitted 1,000 candidates but only 180 survived. Where do I look?"

> **Domain expert:** "Open `dropped.parquet` for each **Filter** in the **Filter Stack** — every dropped **Record** has a **Drop Reason**. Most of yours probably failed the **Judge** because the **Rubric** raised `min_overall_score` last week."

> **Dev:** "And the few that made it — are those already **Examples**?"

> **Domain expert:** "Not yet. They're still **Records** until the output stage serializes them into the **Dataset**. The **Example** is what the trainer actually sees; everything before that is a **Record** with **Lineage** and **Scores** attached."

> **Dev:** "Got it. One more — the **Evol-Instruct** run produced near-duplicates after Round 3. Is that a **Near Dedup** problem or an **Operator** problem?"

> **Domain expert:** "Both signals matter. **Near Dedup** drops the duplicates, but if a specific **Operator** keeps producing them at high `depth`, that's an Operator problem worth fixing upstream."

## Flagged ambiguities

- **"Output"** was used for (a) the final **Dataset**, (b) per-stage **Artifacts**, and (c) LLM **Structured Output**. Resolved: **Dataset** for the exported file, **Artifact** for any per-stage file, **Structured Output** only for schema-constrained LLM responses.
- **"Sample" / "Example" / "Record"** were used interchangeably. Resolved: **Seed** (input bootstrapping material), **Record** (in-pipeline typed unit), **Example** (a row in the final exported **Dataset**).
- **"Filter"** was used for both individual predicate stages and the whole quality stack. Resolved: **Filter** is one stage; **Filter Stack** is the ordered set under `filters.stages`.
- **"Judge" / "Scorer" / "Labeler"** were overloaded. Resolved: **Judge** is the LLM-as-evaluator role, **Labeling Engine** is the orchestrating subsystem, and `scoring_stages.py` hosts non-LLM scorers (e.g. **IFD**).
- **"Provider"** vs **"Model"** were sometimes conflated (e.g. "openrouter model"). Resolved: **Provider** is the API backend; **Model** is the specific identifier requested through it.
- **"Stage"** vs **"Step"** vs **"Phase"** drifted in docs. Resolved: **Stage** is the canonical word; "phase" is reserved for high-level diagram groupings (Input / Transformation / Output) and never for executable units.
- **"Run"** vs **"Job"** vs **"Pipeline execution"**. Resolved: **Run** is the canonical noun, identified by `run_id`; **Pipeline** is the static config-defined shape that a Run executes.
- **"Dedup"** alone was ambiguous between exact and near. Resolved: always qualify as **Exact Dedup** or **Near Dedup**; reserve **Semantic Dedup** for any future embedding-based variant.
