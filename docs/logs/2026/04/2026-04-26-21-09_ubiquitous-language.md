# Ubiquitous Language extraction — 2026-04-26

Wrote `/Users/jshah/Documents/GitHub/arka/CONTEXT.md`.

## Summary

Built a single-context glossary for arka organized into five clusters:

- **Inputs & artifacts** — Seed, Data Source, Document Chunk, Record, Example, Dataset, Artifact
- **Pipeline structure** — Pipeline, Stage, Run, Run Directory, Checkpoint, Manifest
- **Generation** — Generator, Prompt-Based Generation, Evol-Instruct, Operator, Round, Generation Multiplier, Target Count, Lineage
- **Deduplication** — Exact Dedup, Near Dedup, LSH Band
- **Quality & filtering** — Filter, Filter Stack, Judge, Labeling Engine, Rubric, IFD, Canary Filter, Drop Reason
- **LLM access** — LLM Client, Provider, Model, Structured Output

## Key opinionated calls

- **Record** (in-pipeline) vs **Example** (in-dataset) vs **Seed** (input) — three distinct words for what the codebase often calls "example" or "sample".
- **Filter** is one stage; **Filter Stack** is the whole gauntlet.
- **Judge** = LLM evaluator, **Labeling Engine** = orchestrating subsystem (matches `LabelingEngine` in SCOPE.md without conflating the two).
- Always qualify dedup as **Exact** vs **Near**; reserve **Semantic Dedup** for a future embedding-based variant.
- **Provider** ≠ **Model** (kills "openrouter model" sloppiness).
- **Stage** is canonical for executable units; **Phase** is reserved for high-level diagram groupings only.

## Flagged ambiguities (recorded in CONTEXT.md)

- Output (Dataset vs Artifact vs Structured Output)
- Sample / Example / Record
- Filter (one stage vs the whole stack)
- Judge / Scorer / Labeler
- Provider vs Model
- Stage vs Step vs Phase
- Run vs Job vs Pipeline execution
- Dedup (qualify as Exact / Near / Semantic)

## Sources scanned

- `README.md`
- `docs/features.md`
- `docs/SCOPE.md`
- `src/arka/records/models.py`
- `src/arka/pipeline/models.py`
- Directory layout under `src/arka/{pipeline,records,labeling,llm,...}`
