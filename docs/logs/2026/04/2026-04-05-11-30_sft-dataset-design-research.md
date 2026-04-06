# Research: SFT Dataset Design for Language Rewriting Tasks

## Summary

Building a high-quality SFT dataset for rewriting tasks (proofread, rewrite, concise, formalize, summarize) requires: (1) an explicit task taxonomy with clear per-task instructions, (2) structured output format that pairs brief rationale with the rewritten text, (3) ruthless filtering of AI-generated artifacts from training data, and (4) a quality-over-quantity philosophy — 1K curated examples reliably outperform 50K noisy ones. The key insight from recent research (LIMA, Dr Genré, Re-Critic, ParaRev) is that SFT teaches *how to respond*, not new knowledge — so every training example must model the exact response behavior you want at inference.

## Findings

### 1. Quality Beats Quantity — Asymmetrically
LIMA (Meta, NeurIPS 2023) demonstrated that 1,000 carefully curated SFT examples matched or exceeded Alpaca's 52,000 noisy examples. More low-quality data *actively hurts* — the relationship is asymmetric. For a rewriting-specific SFT dataset, start with 1K–2K curated examples per rewriting task, targeting 5K–10K total across all task types. [LIMA, arXiv:2305.11206](https://arxiv.org/abs/2305.11206) · [machinelearningplus guide](https://machinelearningplus.com/machine-learning/custom-instruction-dataset-fine-tuning/)

### 2. Define an Explicit Rewriting Task Taxonomy
Dr Genré (Google DeepMind, 2025) decomposes text rewriting into three orthogonal task families — **factuality** (correct errors), **stylistic** (formalize, paraphrase, summarize, concise), and **conversational** (tone/email editing). ParaRev (ACL 2025) proposes a complementary paragraph-level taxonomy: *Rewriting (light/medium/heavy), Concision, Development, Content (add/substitute/delete)*. For a proofread/rewrite/concise/formalize/summarize task set, each instruction must unambiguously specify the transformation type and intensity. Vague instructions like "improve this" produce inconsistent outputs; "Rewrite this paragraph to be more concise, keeping all key claims" is specific. [Dr Genré, arXiv:2503.06781](https://arxiv.org/abs/2503.06781) · [ParaRev, arXiv:2501.05222](https://arxiv.org/html/2501.05222v1)

### 3. Use Rationale-Augmented Output Format
Re-Critic (EMNLP 2025) demonstrates that inserting rationale/chain-of-thought *before* the final answer in training data significantly improves reasoning quality — yielding 31.8% improvement even with limited data. For rewriting tasks, this maps to a `{rationale → rewritten_text}` output structure: the model first explains *what* it changed and *why* (2–3 sentences), then produces the final rewritten text. This structure teaches the model to reason about edits rather than pattern-match. If you want the model to produce rationale at inference, train with rationale. If you want direct outputs, train with direct outputs — match training to inference format exactly. [Re-Critic, arXiv:2505.07172](https://arxiv.org/html/2505.07172v1)

### 4. Enforce Output Consistency Across All Examples
Every training example in your dataset should follow identical structural conventions: same JSON schema, same section ordering (rationale then text), same punctuation rules, same casing. If half your examples end with a period and half don't, the model learns randomness. Decide conventions and apply them uniformly. Output length must be consistent within each task type — a "concise" task should always produce outputs 30–60% shorter than input; a "proofread" task should have minimal edit ratios. Dr Genré specifically measures **edit ratio** (word-level relative edit distance) as a quality dimension, confirming that conciseness (minimal unnecessary edits) is a trainable objective. [Dr Genré](https://arxiv.org/abs/2503.06781) · [machinelearningplus](https://machinelearningplus.com/machine-learning/custom-instruction-dataset-fine-tuning/)

### 5. Actively Filter AI Artifacts from Training Data
AI-generated training data inherits model-specific stylistic artifacts that propagate through SFT. The most documented artifacts include:

**Punctuation/Structure:**
- Em-dashes (—) used at 10x+ the rate of human prose, likely originating from digitized 19th-century training corpora
- Tricolon abuse ("X, Y, and Z" lists of exactly three items)
- Predictable sentence-level parallelism

**Vocabulary (overused words to flag/filter):**
- Significance inflation: *delve, pivotal, crucial, transformative, revolutionary, game-changing, cutting-edge*
- Magic adverbs: *arguably, fundamentally, notably, remarkably, interestingly*
- Hedging/softening: *generally speaking, tends to, to some extent, broadly speaking*
- Academic filler: *shed light on, facilitate, bolster, streamline, harness, illuminate*
- Transitions: *that being said, at its core, from a broader perspective, a key takeaway is*

**Mitigation strategy for your dataset pipeline:**
1. Build a blocklist of 30–50 AI-tell words/phrases
2. Score each synthetic output for blocklist hits per 100 words
3. Set a threshold (e.g., >2 hits per 100 words → reject or regenerate)
4. Manually review a random 10% sample for cadence/rhythm issues
5. When using LLM-generated examples, add negative style instructions: "Do not use em-dashes. Avoid words: delve, crucial, pivotal, transformative, harness."

[Sean Goedecke em-dash analysis](https://www.seangoedecke.com/em-dashes/) · [Grammarly AI words list](https://www.grammarly.com/blog/ai/common-ai-words/)

### 6. Use Detailed Per-Example Instructions, Not Generic Labels
ParaRev's experiments show that **detailed, personalized revision instructions** significantly outperform generic category labels ("make it concise") across all models tested (Mistral, Llama3, GPT-4o), with p < 0.05. For your dataset, each example should include a specific instruction like:

- ❌ Generic: `"Proofread this text"`
- ✅ Specific: `"Fix the subject-verb agreement error in sentence 2 and correct the misused semicolon in sentence 4"`
- ❌ Generic: `"Make this more concise"`
- ✅ Specific: `"Reduce this 4-sentence paragraph to 2 sentences by removing the redundant qualification in sentence 1 and merging the examples in sentences 3-4"`

This specificity in the *instruction* is what teaches the model to produce specific rationale in its *output*. [ParaRev, arXiv:2501.05222](https://arxiv.org/html/2501.05222v1)

### 7. Decouple Multi-Objective Quality Signals
Dr Genré introduces three decoupled reward dimensions for rewriting quality that should be tracked during dataset construction:

| Dimension | Definition | Measurement |
|-----------|-----------|-------------|
| **Agreement** | Does the output follow the rewrite instruction? | LLM-judge or human check |
| **Coherence** | Is the output internally consistent and fluent? | LLM-judge or human check |
| **Conciseness** | Are edits minimal and necessary? (No gratuitous rewording) | Edit ratio = relative edit distance / source length |

For each training example, verify all three dimensions. A common failure mode: the output follows the instruction (high agreement) but introduces unnecessary rewording throughout the passage (low conciseness) or creates logical contradictions (low coherence). Each task type should have different weight profiles: proofreading demands very low edit ratio; summarization allows high edit ratio but requires semantic preservation. [Dr Genré](https://arxiv.org/abs/2503.06781)

### 8. Source Real-World Inputs, Generate Synthetic Outputs Carefully
The best pipeline is: **real human text as inputs** + **carefully constrained LLM-generated outputs** + **human review of a random sample**. CRAFT (2024) demonstrates that using real corpus documents (retrieved via semantic similarity from C4, Wikipedia, StackExchange) as the *source texts* for your instruction pairs, then augmenting with LLM-generated rewrites, consistently outperforms fully synthetic approaches. Key constraints for synthetic output generation:

- Generate in batches of 5–10, not hundreds (less repetition, more diversity)
- Include 2–3 gold examples as few-shot demonstrations in the generation prompt
- Apply echo detection: reject outputs where output ≈ input (especially for "proofread" where the lazy generation is copy-paste)
- Apply deduplication: hash instruction+input, keep first occurrence, reject same-question-different-answer contradictions
- Deduplicate *before* train/test split to prevent leakage

[CRAFT, arXiv:2409.02098](https://arxiv.org/pdf/2409.02098v1) · [machinelearningplus](https://machinelearningplus.com/machine-learning/custom-instruction-dataset-fine-tuning/)

---

## Actionable Design Rules (Summary)

| # | Rule | Why |
|---|------|-----|
| 1 | **1K curated > 50K noisy** | SFT teaches format, not knowledge. Quality is asymmetrically more important. |
| 2 | **Explicit task taxonomy** | Each of {proofread, rewrite, concise, formalize, summarize} needs distinct instruction templates with specified intensity. |
| 3 | **Rationale-then-output format** | Train with `{brief_rationale, rewritten_text}` structure to teach reasoning about edits. |
| 4 | **Enforce structural consistency** | Same JSON schema, section ordering, punctuation rules, and expected length ranges per task. |
| 5 | **Filter AI artifacts actively** | Maintain a blocklist of 30–50 AI-tell words/phrases; score and reject high-artifact outputs. |
| 6 | **Specific instructions per example** | "Fix the dangling modifier in sentence 3" >> "proofread this". |
| 7 | **Track 3 quality dimensions** | Agreement (follows instruction), Coherence (internally consistent), Conciseness (minimal edit ratio). |
| 8 | **Real inputs + constrained synthetic outputs** | Source human-written text for inputs; generate outputs with style-negative prompts, echo detection, and dedup. |

---

## Sources

### Kept
- **LIMA** (arXiv:2305.11206) — Foundational evidence for quality-over-quantity in SFT datasets
- **Dr Genré** (arXiv:2503.06781, Google DeepMind 2025) — Multi-task rewriting framework with decoupled objectives: agreement, coherence, conciseness. Directly addresses rewriting task taxonomy and quality measurement.
- **Re-Critic** (arXiv:2505.07172, EMNLP 2025) — Evidence that rationale-augmented training data improves reasoning; 31.8% gain with limited data.
- **ParaRev** (arXiv:2501.05222, ACL 2025) — Paragraph-level revision taxonomy and evidence that detailed instructions outperform generic labels (p<0.05).
- **CRAFT** (arXiv:2409.02098, 2024) — Corpus retrieval and augmentation pipeline for SFT dataset construction from real documents.
- **machinelearningplus guide** (2025) — Practical end-to-end dataset construction guide with concrete code and thresholds.
- **Sean Goedecke em-dash analysis** (2025) — Root cause analysis of em-dash overuse tracing to training data composition (digitized 19th-century books).
- **Grammarly AI words analysis** (2026) — Comprehensive categorized list of AI writing artifacts (vocabulary, transitions, hedging, buzzwords).
- **Zhang et al., "The Best Instruction-Tuning Data are Those That Fit"** (arXiv:2502.04194, 2025) — Evidence that SFT data matching downstream distribution outperforms random high-quality data.

### Dropped
- JMIR healthcare tutorial — domain-specific (medical), not transferable to general rewriting
- Taskmonk/Sapien/CVAT labeling guides — too generic (image/annotation focused), not text-rewriting specific
- Alibaba AI writing articles — SEO content, no original research or evidence
- Medium em-dash opinion piece — anecdotal, superseded by Goedecke's data-driven analysis
- Comprehensive LLM Finetuning Guide (TowardsAI) — too broad, no rewriting-specific insights
- Philschmid HuggingFace guide — infrastructure-focused (how to run SFT), not dataset design

## Gaps

1. **No direct benchmarks for rewriting-specific SFT dataset size vs. quality curves** — LIMA is general-purpose; no equivalent study exists for text editing tasks specifically. Would need to run ablation experiments.
2. **Rationale format trade-offs** — Re-Critic demonstrates rationale-augmented training but in a vision-language setting. No paper directly studies rationale-augmented SFT for text rewriting (proofread/formalize/etc.). The analogy is strong but unvalidated in this exact domain.
3. **AI artifact filtering thresholds** — The blocklist/scoring approach is practitioner consensus, not rigorously validated. No paper provides evidence-backed thresholds for "how many AI-tell words per 100 is too many."
4. **DPO follow-up** — After SFT, a DPO stage with preference pairs (good rewrite vs. bad rewrite for the same input) would further align the model. Dr Genré shows this works but requires separate preference data collection.

### Suggested Next Steps
- Build a pilot dataset of 200 examples across 5 task types (40 each)
- Run human evaluation on a random 50 to validate consistency and artifact scores
- Iterate on instruction templates and output format before scaling to 1K+
