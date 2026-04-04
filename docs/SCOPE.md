# Synth Data Playbook — Project Scope

Last updated: April 2026

---

## GOAL

Build a config-driven synthetic data generation framework from first principles.
Primary objective is **learning** — understanding how each technique works by implementing it,
not wrapping existing tools. Everything is built with LLM APIs, plain Python, own pipeline stages.

---

## HARD CONSTRAINTS

- LLM via API only — OpenAI, Anthropic, etc. Swappable via config. No vLLM, no local inference.
- No off-the-shelf orchestration — no distilabel, no curator, no LangChain. Read them as reference. Build the logic yourself.
- Config-driven — YAML defines the pipeline. Pydantic validates it. Code executes it.
- Test-driven — every stage is independently testable before wiring together.
- Thin-sliced — smallest working end-to-end slice per use case, then extend.

---

## IN SCOPE

### Use Cases (data types to generate)
- Instruction tuning data (SFT)
- RL / preference data (DPO, GRPO, verifiable rewards / RLVR)
- Embedding training data
- Reranker training data
- Evaluation dataset creation
- Tool use / agentic trajectory data (input + tool call sequence + output)

### Data Sources
- LLM-generated (seed-based, seedless Magpie-style, persona-conditioned)
- Private documents / PDFs (ingestion → chunking → generation pipeline)

### Core Modules (built from scratch)
- LLM client wrapper — single interface, swappable provider, structured output support
- Pipeline runner — config loader → stage executor → output writer
- LabelingEngine — rubric-driven, multi-LLM voting, confidence scoring, structured output
- Generation techniques — Magpie, Evol-Instruct, Backtranslation, Persona-conditioned, etc.
- Quality filtering — heuristic, IFD scoring, reward model scoring, LLM-as-judge
- Deduplication — exact hash, MinHash LSH, semantic (embedding-based)
- Continuous learning flywheel — human corrections → preference pairs → feeds back into RL pipeline

---

## NOT IN SCOPE

- Pre-training / web crawl curation (FineWeb, C4 style)
- Multimodal data (image-text, audio)
- Web UI, job manager, Bedrock APIs (separate work project)
- Verifier agent with web search
- Kappa stats / inter-annotator agreement tooling
- Continual learning / catastrophic forgetting mitigations
- Local inference / quantization / vLLM serving

---

## DESIGN PRINCIPLES

- Own the logic — implement techniques from first principles, use research as reference
- Config-driven — swap models, thresholds, techniques via YAML without code changes
- Thin-sliced — each use case is smallest possible end-to-end slice first
- Quality over quantity — generate 5-10x target, filter aggressively down
- Everything testable — each stage is a pure function or testable class

---

## NOTES

### LabelingEngine / Work Project Overlap
Built as a standalone module here, it becomes the core labeler agent for the work project.
The continuous learning flywheel (human corrections → DPO preference pairs) is the connective tissue.

### Off-the-shelf tools as reference only
- Distilabel: reference for how Magpie, Evol-Instruct, UltraFeedback are implemented
- Curator: reference for async batch generation patterns
- DataTrove: reference for dedup pipeline design
