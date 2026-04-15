# Arka Enhancements for Humanize-RL (and Beyond)

> **Status:** Requirements — Arka-first, project-second
> **Date:** 2026-04-13
> **Motivation:** humanize-rl needs paired rewriting data. But the gaps it exposes are generic gaps in Arka that benefit all pipelines.

---

## Philosophy

Do NOT build humanize-rl-specific stages in Arka. Instead, identify what's missing from Arka as a **general-purpose synth data framework** and build that. The humanize-rl pipeline then becomes a YAML config + rubric + prompt template that runs on a better Arka.

Every enhancement below is justified by what SOTA frameworks (distilabel, Bespoke Curator, NVIDIA NeMo SDG, SyGra, CROWDSELECT) already have that Arka doesn't yet.

---

## Gap 1: Reward-Model Scoring Stage (P0)

### What SOTA does
NVIDIA's rewards-guided SDG pipeline (SciPy 2025) integrates a reward model directly into the generation loop: generate → score with reward model → filter by threshold → keep. CROWDSELECT (EACL 2026) uses multi-LLM wisdom with reward model assessment as one of three foundational metrics. OptimSyn (ICLR 2026) uses influence-guided rubrics as reward signals.

### What Arka has
`LabelingEngine` does rubric-based LLM-as-judge scoring. It scores (instruction, response) pairs against a YAML rubric with per-dimension ratings. `LabelingQualityFilterStage` filters by `min_overall_score`.

### What's missing
Arka can score with an LLM judge but has no way to score with a **reward model** (scalar score endpoint, not chat completion). The NVIDIA pipeline calls `client.query_reward_model(messages, model)` which returns a single float. This is different from LLM-as-judge — it's a dedicated scoring endpoint.

### What to build
A `RewardModelScoringStage` that:
- Calls any OpenAI-compatible reward model endpoint
- Gets a scalar score per record
- Stores in `RecordScores.reward_model`
- Filters by configurable threshold
- Supports score normalization (percentile-based, not absolute — makes it model-agnostic)

```yaml
filters:
  reward_model:
    enabled: true
    model: "nvidia/Llama-3.1-Nemotron-70B-Reward"  # or any reward model
    base_url: "https://integrate.api.nvidia.com/v1"
    min_score: 0.0          # normalized threshold
    normalize: percentile   # percentile | minmax | none
```

**Why this is generic:** Any SFT/DPO pipeline benefits from reward model scoring. It's not humanize-rl-specific.

**How humanize-rl uses it:** Score humanized outputs with ArmoRM or Nemotron Reward to ensure quality didn't degrade during rewriting.

---

## Gap 2: Transform/Rewrite Stage (P0)

### What SOTA does
Distilabel has `Task` — an LLM-powered `Step` that takes input text, transforms it with a prompt, and outputs transformed text. SyGra models the pipeline as a directed graph where any node can be a transformation. The NVIDIA pipeline does paraphrasing and question generation as transformation steps.

### What Arka has
- `PromptBasedGeneratorStage`: takes seeds, generates NEW (instruction, response) pairs
- `EvolInstructRoundStage`: evolves instructions through operators

Both **generate new records**. Neither **transforms existing text in-place** (take a text field, rewrite it, store the result alongside the original).

### What's missing
A generic `TransformStage` that:
- Takes a record's text field (configurable: `payload.response`, `payload.instruction`, etc.)
- Passes it through an LLM with a configurable prompt template
- Stores the result in a configurable output field
- Preserves the original text for comparison
- Tracks lineage (operator="transform", parent_id set)

This is the primitive that makes AIify and Humanize expressible as YAML config — no custom stages needed.

```yaml
stages:
  - type: transform
    name: aiify
    input_field: payload.instruction    # field to transform
    output_field: payload.response      # where to store result
    preserve_original: true             # keep original in payload.system (or metadata)
    prompt_template: |
      Rewrite this text to add typical AI assistant writing patterns.
      Add 4-6 patterns prominently while keeping the core content.
      
      Text to rewrite:
      {input_text}
    model: qwen/qwen3.5-9b
    temperature: 0.7

  - type: transform
    name: humanize
    input_field: payload.response       # transform the AI-ified text
    output_field: payload.response      # overwrite with humanized version
    preserve_original: true
    prompt_template: |
      Rewrite the following text to sound more natural and human.
      Remove AI writing patterns while preserving meaning.
      
      Text to rewrite:
      {input_text}
    model: google/gemini-3.1-pro
    temperature: 0.3
```

**Why this is generic:** Any pipeline that needs text rewriting, paraphrasing, style transfer, translation, simplification, etc. Distilabel has this. Arka should too.

**How humanize-rl uses it:** AIify = transform with "add AI patterns" prompt. Humanize = transform with "remove AI patterns" prompt. No custom stages.

---

## Gap 3: Multi-Signal Quality Selection (P1)

### What SOTA does
CROWDSELECT (EACL 2026) is the key paper. It proposes three foundational metrics:
1. **Diverse LLM responses** — score from multiple models, not just one
2. **Reward model assessment** — scalar reward score
3. **IFD (Instruction-Following Difficulty)** — measures how much the instruction conditions the output

Then it combines them with a clustering-based approach for final selection. Models fine-tuned on CROWDSELECT-filtered data improve 4.81% on Arena-Hard and 11.1% on MT-bench.

### What Arka has
- `LabelingEngine` (LLM judge, single or multi)
- `IFDFilterStage` (exists but scoring logic is a stub)
- No reward model scoring (see Gap 1)
- No composite selection that combines multiple signals

### What's missing
1. **Complete IFD implementation** — the log-prob math is stubbed. Needs:
   - Log-prob of output tokens conditioned on instruction
   - Log-prob of output tokens unconditionally
   - IFD = ratio of the two
   - Capability-gated (only when provider supports logprobs)

2. **Composite score selection** — a final `SelectStage` that:
   - Takes multiple score signals (quality, reward_model, ifd)
   - Computes a weighted composite
   - Selects top-N by composite
   - Supports CROWDSELECT-style clustering-then-select

```yaml
filters:
  select:
    enabled: true
    target_count: 5000
    composite_weights:
      quality: 0.4           # from LabelingEngine
      reward_model: 0.3      # from RewardModelScoringStage
      ifd: 0.3               # from IFD scorer
    strategy: top_n           # top_n | clustered_top_n
```

**Why this is generic:** Every SFT pipeline benefits from multi-signal selection. Single-signal (just LLM judge, or just reward model) is known to be suboptimal (CROWDSELECT paper).

---

## Gap 4: Paired/Delta Filtering (P1)

### What SOTA does
DPO pipelines generate (chosen, rejected) pairs and filter by margin. The NVIDIA pipeline generates variants and compares reward scores. Feedback-driven text rewriting pipelines (Emergent Mind survey, Jan 2026) pair original with rewritten text and measure delta.

### What Arka has
`PreferencePayload` with `chosen`, `rejected`, `margin` fields. But no stage that:
- Compares scores between two versions of the same record
- Filters by delta (score improvement must be > threshold)
- Computes semantic similarity between versions

### What's missing
A `PairFilterStage` (or extend existing filter infrastructure) that:
- Compares `RecordScores` between a record and its parent (via lineage)
- Filters by minimum delta on configurable score field
- Optionally computes semantic similarity (BERTScore or embedding cosine)
- Filters by minimum similarity (meaning preserved)

```yaml
filters:
  pair_delta:
    enabled: true
    score_field: quality              # compare this score
    min_delta: 0.30                   # before→after improvement
    semantic_similarity:
      enabled: true
      min_score: 0.80
      method: bertscore               # bertscore | cosine
    length_ratio:
      max: 1.30                       # max 30% length change
```

**Why this is generic:** Any rewriting or preference-pair pipeline needs delta filtering. DPO pair generation, text simplification, style transfer — all need "did the transformation actually improve things?"

**How humanize-rl uses it:** Filter pairs where humanization didn't improve the rubric score enough, or where meaning wasn't preserved.

---

## Gap 5: Deterministic Pre-Filters (P2)

### What SOTA does
Every pipeline runs cheap heuristic filters before expensive LLM calls. Arka already has `LengthFilterStage`, `LanguageFilterStage`, and `HeuristicFilter` (alpha ratio, ngram repetition, special chars). But text quality heuristics go further.

### What Arka already has (good)
- Length filter
- Language filter  
- Alpha ratio, repetitive ngrams, special char density

### What could be added (generic, useful for all pipelines)
- **Sentence variance** — stdev of sentence lengths (rival.tips: #1 discriminator between models, CV 2.78). Useful as a diversity signal for any generated text.
- **Repetition detector** — detect repeated phrases/sentences within a single document (not just across documents like dedup). AI-generated text often repeats ideas in different words.
- **Formatting heuristics** — bullet ratio, heading density, em-dash rate. Useful signals for any pipeline that wants to control output format.

These are cheap, deterministic, and universally useful. NOT humanize-rl-specific.

```yaml
filters:
  heuristic:
    enabled: true
    min_alpha_ratio: 0.6
    max_repeated_ngrams: 3
    sentence_variance:
      enabled: true
      min_cv: 0.15                    # coefficient of variation
    formatting:
      max_bullet_ratio: 0.25         # max % of lines that are bullets
      max_em_dash_rate: 3.0          # per 500 words
```

---

## What Does NOT Go in Arka

| Thing | Where it goes | Why |
|-------|---------------|-----|
| Humanness rubric YAML | `humanize-rl/rubrics/` | Domain-specific rubric, not a framework concern |
| AI-ify prompt template | `humanize-rl/configs/` | Just a YAML config value for the TransformStage |
| Humanize prompt template | `humanize-rl/configs/` | Same — just a prompt |
| Layer 1 deterministic scorer (claudiness) | `humanize-rl/src/` | Project-specific heuristics. The sentence_variance and formatting parts could be Arka heuristic filters, but claudiness regex patterns are humanize-rl-specific |
| RL environment (verifiers) | `humanize-rl/src/` | Training infrastructure, not data pipeline |
| Fine-tuning scripts | `humanize-rl/src/` | Downstream consumer of the data |

---

## How Humanize-RL Runs on Enhanced Arka

With Gaps 1-4 built, the humanize-rl pipeline is ONE Arka config file:

```yaml
version: "1"
run_id: null

llm:
  provider: openai
  model: google/gemini-3.1-pro
  api_key: ${OPENROUTER_API_KEY}
  base_url: https://openrouter.ai/api/v1

data_source:
  type: seeds
  path: ./seeds/human-texts.jsonl     # curated human-written texts

# Step 1: Generate AI-ified version (TransformStage)
generator:
  type: transform
  input_field: payload.instruction
  output_field: payload.response
  preserve_original: true
  prompt_template: |
    Rewrite this to add typical AI assistant writing patterns (4-6 patterns).
    Keep the same information and length.
    {input_text}
  model: qwen/qwen3.5-9b
  temperature: 0.7

# Step 2: Score AI-ified version with humanness rubric
labeling_engine:
  rubric_path: ./rubrics/humanness_v01.yaml
  mode: single

# Step 3: Filters (cheap first, expensive last)
dedup:
  exact:
    enabled: true
  near:
    enabled: true

filters:
  target_count: 5000
  length:
    enabled: true
    min_instruction_chars: 200
    max_instruction_chars: 2400
  language:
    enabled: true
    allowed: [en]
  heuristic:
    enabled: true
    sentence_variance:
      enabled: true
      min_cv: 0.15
  labeling_engine:
    enabled: true
    rubric_path: ./rubrics/humanness_v01.yaml
    max_overall_score: 0.40           # AI-ified text MUST score low (clearly AI)

output:
  format: jsonl
  path: ./output/aiified.jsonl
```

Then a second config for the humanize step. Then humanize-rl glue code pairs them and runs delta filtering.

**OR** if Arka supports multi-step pipelines (transform → score → transform → score → pair_filter), it's a single config. That's the stretch goal.

---

## Priority for Arka

| # | Gap | Effort | Impact | Priority |
|---|-----|--------|--------|----------|
| 1 | TransformStage | Medium | High — enables rewriting, paraphrasing, style transfer, translation for ANY pipeline | P0 |
| 2 | IFD implementation | Small | Medium — completes existing stub, generic quality signal | P0 |
| 3 | RewardModelScoringStage | Medium | High — reward-guided generation is SOTA (NVIDIA, CROWDSELECT) | P1 |
| 4 | PairFilterStage (delta + similarity) | Medium | Medium — enables DPO pair generation, rewriting pipelines | P1 |
| 5 | Composite score selection | Small | Medium — CROWDSELECT shows multi-signal > single-signal | P1 |
| 6 | Deterministic heuristics (sentence variance, formatting) | Small | Low-Medium — cheap pre-filters, nice to have | P2 |

---

## Research References That Inform These Gaps

| Paper/Tool | What Arka should learn from it |
|------------|-------------------------------|
| NVIDIA Rewards-Guided SDG (SciPy 2025) | Reward model scoring as a pipeline stage, generate→score→filter loop |
| CROWDSELECT (EACL 2026) | Multi-signal quality selection (LLM diversity + reward model + IFD), clustering-based final selection |
| OptimSyn (ICLR 2026) | Influence-guided rubric optimization — rubrics as reward signals, iterative rubric refinement |
| RRD (arxiv 2602.05125) | Recursive rubric refinement — decompose/filter/weight rubric criteria for better reward signals |
| Distilabel (argilla-io) | Task = LLM-powered Step for text transformation, DAG pipeline architecture |
| Bespoke Curator | Structured data extraction + curation, Pydantic-based schema validation |
| SyGra (ServiceNow, ICLR 2026) | Graph-based pipeline, dual-stage quality tagging (generation quality + response quality) |
| PiKa (arxiv 2510.06670) | Expert-level alignment datasets, 30K high-difficulty pairs outperform 500K unfiltered |

---

## Open Questions

- [ ] Should TransformStage be a new stage type, or an extension of `PromptBasedGeneratorStage` with a `mode: transform` flag?
- [ ] Multi-step pipeline: can the current `StageBuilder` support transform → score → transform → score → pair_filter in one run? Or is two runs + glue the right pattern?
- [ ] IFD: which open models support logprobs via OpenRouter? (Qwen3.5, DeepSeek, Llama 4 — need to check)
- [ ] Reward model API: is there a standard interface, or does each provider differ? The NVIDIA approach uses `query_reward_model()` — is this an OpenAI-compatible endpoint?
- [ ] `LabelingEngine` currently scores (instruction, response) pairs. Can it score standalone text (just response)? The humanness rubric needs to score text alone, not instruction-response pairs.
