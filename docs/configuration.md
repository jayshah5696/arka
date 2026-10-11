# Configuration Reference

Arka is completely configuration-driven. The entirety of a generation run—from LLM credentials to sequential pipeline stages—is defined in a single YAML file. This document details the schema of the Arka configuration, corresponding to the `ResolvedConfig` Pydantic model.

---

## High-Level Anatomy

A complete configuration file is composed of logical configuration blocks and a sequential list of stages executed in order.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'edgeLabelBackground':'#ffffff', 'tertiaryColor': '#f4f4f4'}}}%%
classDiagram
    class ResolvedConfig {
        version: str
        run_id: str
    }
    ResolvedConfig *-- LLMConfig : llm
    ResolvedConfig *-- ExecutorConfig : executor
    ResolvedConfig *-- "list[PipelineStageConfig]" : pipeline
    ResolvedConfig *-- EmbeddingsConfig : embeddings
    ResolvedConfig *-- LabelingEngineConfig : labeling_engine
    ResolvedConfig *-- OutputConfig : output
```

Here is an outline of the top-level keys in YAML:

```yaml
version: "1"
run_id: "optional-explicit-id"
llm: {...}
executor: {...}
pipeline:
  - type: seed_source
    path: ./seeds.jsonl
  - type: normalize_conversation
  - type: prompt_based_generator
    target_count: 100
  - type: exact
  - type: near
    lsh_bands: 16
  - type: length
    min_response_chars: 50
  - type: labeling_score
    rubric_path: ./rubrics/quality.yaml
    min_overall_score: 3.5
embeddings: {...}
labeling_engine: {...}
output: {...}
```

---

## 1. Top-Level Metadata

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `version` | `str` | **Required** | The version of the configuration schema. Currently must be `"1"`. |
| `run_id` | `str` | `None` | A unique string identifying the run. Used for checkpointing, resuming, and artifact folder naming. If omitted, the CLI `--run-id` flag is used, or one is auto-generated. |

---

## 2. LLM (`llm`)

Defines the connection to the Language Model used for generation and evaluation.

```yaml
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
  base_url: https://api.openai.com/v1
  timeout_seconds: 60.0
  max_retries: 5
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `provider` | `Literal["openai"]` | **Required** | The provider protocol. Currently `"openai"` is supported. |
| `model` | `str` | **Required** | The exact model ID expected by the provider. |
| `api_key` | `str` | **Required** | The authentication token. **Best Practice:** Use environment variable substitution (e.g. `${OPENAI_API_KEY}`). |
| `base_url` | `HttpUrl` | **Required** | The base URL for the API endpoints. |
| `timeout_seconds` | `float` | `30.0` | Connection timeout in seconds. |
| `max_retries` | `int` | `3` | Number of times to retry on transient errors. |
| `supports_json_schema`| `bool` | `None` | Override auto-detection for whether the provider natively supports JSON Schema. |
| `openai_compatible` | `Object` | `None` | Advanced settings for third-party endpoints (e.g. OpenRouter). |

### `openai_compatible` Object
Used to send specific headers required by aggregators like OpenRouter.
* `referer` (`HttpUrl`): Identifies your application URL.
* `title` (`str`): Identifies your application name.

---

## 3. Executor (`executor`)

Controls concurrency and throughput.

```yaml
executor:
  mode: threadpool
  max_workers: 10
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `mode` | `Literal["threadpool"]` | `"threadpool"`| The execution strategy. |
| `max_workers` | `int` | `4` | Maximum number of concurrent tasks (API requests) to run in parallel. |

---

## 4. Pipeline Stages (`pipeline`)

The `pipeline` is an ordered list of stages that run sequentially to ingest, normalize, generate, deduplicate, and filter the dataset. Each stage is defined as an object in the list with a identifying `type` field.

### Data Source Stages

#### `seed_source`
Loads seed data from a structured dataset.
```yaml
- type: seed_source
  path: ./seeds.jsonl
```
| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | `str` | **Required** | Path to the seeds file. Supports CSV and JSONL formats. |

#### `pdf_source`
Ingests raw text by chunking a PDF file.
```yaml
- type: pdf_source
  path: ./documents/manual.pdf
  chunk_strategy: fixed
  chunk_size_chars: 2000
  chunk_overlap_chars: 200
```
| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | `str` | **Required** | Path to the PDF file. |
| `chunk_strategy` | `str` | `"fixed"` | Strategy for breaking down text (e.g., `"fixed"`). |
| `chunk_size_chars` | `int` | `3000` | Number of characters per chunk. |
| `chunk_overlap_chars`| `int` | `300` | Number of overlapping characters between chunks. |

Each chunk reports the first and last pages whose extracted text overlaps it.
Page numbers start at 1 and include empty pages in their numbering.
`char_start` and `char_end` are half-open character offsets into the normalized
document text. Arka strips each page's outer whitespace, joins non-empty pages
with two newlines, and trims each chunk. Slicing the normalized document at
these offsets returns the chunk text exactly. The payload and source contain
the same provenance.

These fields describe extracted text, not coordinates in the PDF. Extraction
quality and reading order depend on the PDF; OCR is not supported. Corrected
provenance changes affected record IDs because Arka hashes the full payload.
The document hash, chunk numbering, and fixed window positions remain unchanged.

---

### Normalization Stages

#### `normalize_conversation`
Normalizes incoming records into standard Conversation records.
```yaml
- type: normalize_conversation
```

---

### Generator Stages

#### `prompt_based_generator`
Generates SFT pairs based on seed instructions using a customizable LLM prompt template.
```yaml
- type: prompt_based_generator
  target_count: 1000
  generation_multiplier: 2
  prompt_template: >
    Generate a new instruction-response pair similar to the following seed.
    Seed: {seed_instruction}
    Response: {seed_response}
  temperature: 0.8
```
| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `target_count` | `int` | **Required** | The desired number of items to generate. |
| `generation_multiplier`| `int` | **Required** | Oversampling factor (e.g. `3` generates `3 * target_count`). |
| `prompt_template` | `str` | *(Default)* | Jinja-style template with `{seed_instruction}` and `{seed_response}`. |
| `temperature` | `float` | `0.7` | LLM sampling temperature. |
| `max_tokens` | `int` | `512` | Maximum length of the generated response. |

#### `evol_instruct_generator`
Performs multi-round Evol-Instruct mutations on instructions and generates responses.
```yaml
- type: evol_instruct_generator
  target_count: 500
  generation_multiplier: 3
  rounds: 2
  branching_factor: 2
  operators: ["add_constraints", "deepen"]
```
| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `target_count` | `int` | **Required** | Desired target size. |
| `generation_multiplier`| `int` | **Required** | Oversampling factor. |
| `rounds` | `int` | `2` | Number of evolution rounds to perform. |
| `branching_factor` | `int` | `1` | Number of variations to create per seed, per round. |
| `operators` | `list[str]` | `[]` | Allowed mutation operators (e.g., `"deepen"`, `"add_constraints"`, `"concretizing"`, `"breadth_mutation"`). |

---

### Deduplication Stages

#### `exact`
Fast content-hash exact deduplication.
```yaml
- type: exact
```

#### `near`
MinHash/LSH fuzzy deduplication for identifying near-duplicate instructions.
```yaml
- type: near
  shingle_size: 5
  num_hashes: 128
  lsh_bands: 16
  jaccard_threshold: 0.70
```
| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `shingle_size` | `int` | `5` | Word/char n-gram size for hashing. |
| `num_hashes` | `int` | `128` | MinHash signature size. |
| `lsh_bands` | `int` | `16` | Number of LSH bands for bucketing. |
| `jaccard_threshold` | `float` | `0.7` | Jaccard similarity threshold for filtering. |

---

### Filter & Scoring Stages

#### `length`
Filters out records whose instructions or responses are too short or too long.
```yaml
- type: length
  min_instruction_chars: 10
  max_instruction_chars: 4096
  min_response_chars: 10
  max_response_chars: 16384
```

#### `language`
Filters out records that do not match the allowed language.
```yaml
- type: language
  allowed: ["en"]
```

#### `repetition`

Drops conversation records with long character runs or repeated word sequences.
Place this stage after generation and length filtering, before LLM judging.

```yaml
- type: repetition
  check_fields: [response]
  min_text_chars: 200
  top_ngram_thresholds: {2: 0.20, 3: 0.18, 4: 0.16}
  dup_ngram_thresholds: {5: 0.15, 6: 0.14, 7: 0.13, 8: 0.12, 9: 0.11, 10: 0.10}
  max_char_run: 50
```

`check_fields` accepts `instruction`, `response`, and `system`. By default, only
the response is checked. Character runs longer than `max_char_run` are dropped
even in short text. `min_text_chars` applies only to word-sequence checks.

Arka splits words on whitespace and preserves case and punctuation. For each
configured length, `top_ngram_thresholds` limits the fraction of text covered by
the most frequent repeated word sequence. Ties use the largest covered fraction.
`dup_ngram_thresholds` limits the fraction covered by all repeated sequences.
Spans include whitespace between words. Overlapping spans count each character
once, so fractions stay between 0 and 1. A fraction must exceed its threshold
to trigger a drop. An empty threshold map disables that family of checks.

The thresholds come from the Gopher repetition rules, adapted to word sequences
and character coverage. They need validation on your SFT data, especially code
and lists. The stage makes no LLM calls and adds no dependencies. Its report
contains drop counts by reason and `max_observed_fractions` for evaluated metrics.
Dropped records include the field and offending metric in their event details.

#### `canary`
Filters out generations containing sensitive phrases or secrets.
```yaml
- type: canary
  phrases: ["SECRET_TOKEN"]
```

#### PDF boilerplate removal

Enable repeated-block removal inside `pdf_source`, before chunking:

```yaml
- type: pdf_source
  path: report.pdf
  boilerplate_strip: true
  min_block_bytes: 500
  min_doc_frequency: 2
  min_remaining_bytes: 100
```

The default `boilerplate_strip: false` preserves existing chunks. The three
numeric options require positive block size, at least two distinct pages,
and a non-negative remaining-byte threshold. Detection compares aligned UTF-8
byte blocks within one PDF. It extends confirmed blocks through identical
adjacent bytes and merges overlapping cuts. Repeats without a shared aligned
block can be missed. It removes all detected occurrences, including the first.

Original page numbers remain intact. Chunk character offsets refer to the
cleaned, trimmed page texts joined with two newlines. `boilerplate_spans.jsonl`
contains original trimmed-page UTF-8 byte offsets, removed-content BLAKE2b
hashes, and distinct-page occurrence counts. These byte offsets differ from
chunk character offsets. No removed text is stored in the span report.

`stats.json` reports `boilerplate_bytes_removed`, `boilerplate_blocks_cut`
as the count of merged removal spans, `pages_affected`, and
`docs_stripped_empty`. If cleanup leaves fewer than `min_remaining_bytes`
across the page texts, or no text, the stage emits zero chunks and reports
the empty result. A PDF with no extracted text still raises the existing error.

#### `reasoning_language_gate`

Conversation records can carry a separate optional `reasoning` string. JSONL
and CSV seed sources preserve it; prompt-based and taxonomy generators preserve
it when returned in structured JSON. Normalization trims it, and Parquet
preserves it. Missing reasoning does not change existing payload hashes or
answer-only exports. No new prompt instruction forces a model to emit reasoning.

ChatML exports retained reasoning as `<think>\nTRACE\n</think>\n\nANSWER` in
the assistant content. Alpaca uses the same text in `output`. This preserves
both strings in the conversational `messages` format accepted by
[Hugging Face TRL](https://huggingface.co/docs/trl/dataset_formats).
Apply the target model's chat template when preparing training tokens.
JSONL keeps `reasoning` and `response` separate, so you can adapt them to other
model-specific reasoning formats. A stripped or absent trace exports only the
answer, without a think block.

```yaml
- type: reasoning_language_gate
  reference_field: instruction
  checked_fields: [reasoning, response]
  on_mismatch: strip_trace
  min_confidence: 0.6
  lid_model_path: models/lid.176.ftz
```

Install the optional backend with `uv sync --extra language`. Download a
language-identification model separately. The compressed `lid.176.ftz` model
is available from [fastText's official model page](https://fasttext.cc/docs/en/language-identification.html).
The model has its own CC BY-SA 3.0 license. Model paths resolve relative to the
configuration file. The stage does not download models or call an LLM.
To run the optional native integration test, install the language extra and
set `ARKA_TEST_LID_MODEL` to that local model path before `just check`.

`reference_field` accepts `instruction`, `response`, or `system`.
`checked_fields` is a non-empty list containing any of `reasoning`, `response`,
`instruction`, or `system`. The confidence threshold must be between zero and
one. Both reference and checked-field predictions must reach it before a
mismatch triggers an action. Empty fields and low-confidence predictions pass
unjudged. Records without a non-empty trace pass unchanged.

The default `strip_trace` removes reasoning and keeps the answer, even when
the mismatching checked field is the response. It recomputes the content hash,
preserves IDs and lineage, and adds `reasoning_stripped` with detected languages
and confidences. `drop_record` instead writes a `reasoning_language_mismatch`
drop. LLM healing is not implemented and is rejected as an action.

An unset `lid_model_path` logs a warning, passes records, and reports
`enabled: false`. A supplied missing or invalid model raises an error;
it does not silently disable the gate. Missing backend installation also
raises an error with the install command.

`stats.json` reports `stripped_count`, drop counts, per-field mismatch and
confident comparison counts, mismatch rates among confident comparisons,
low-confidence field counts, and `unjudged_reference_count`. Drop details
and stripping events record the predictions used for the decision.

#### `decontaminate`

Check conversation text against local evaluation JSONL files before `select`:

```yaml
- type: decontaminate
  eval_paths: [evals/math.jsonl, evals/science.jsonl]
  eval_text_field: question
  fields: [instruction]
  n: 5
  idf: per_benchmark
  window_factor: 8
  threshold: 0.9
  calibrate: false
  action: drop
```

Paths resolve relative to the configuration file. `eval_paths` must be non-empty
and cannot repeat a file. Each non-empty JSONL row must contain a non-empty
string at `eval_text_field`. Rows shorter than `n` normalized words are counted
and skipped; files without usable rows fail with file context. Tokenization
case-folds text and splits on punctuation and underscores. `n` and
`window_factor` must be positive integers. `fields` is a non-empty list of
`instruction`, `response`, or `system`, defaulting to `instruction`.

The index considers only shared n-grams. It measures unique matching n-gram
weight divided by the evaluation item's total unique n-gram weight, following
the overlap definition in [AllenAI's decon documentation](https://github.com/allenai/decon/blob/main/doc/simple-details.md).
Arka uses `log((item_count + 1) / (document_frequency + 1))` weights, computed
per benchmark by default or across all files with `idf: global`. Common n-grams
have zero weight. A single-item scope uses unit weights. Items with only
zero-weight n-grams are unscorable and are reported rather than flagged.
Matches must fit inside `window_factor` times the item's word count. Separate
fields cannot combine to form a match. This is a question-text matcher, not the
full decon engine's answer and passage scoring system.

`threshold` must be greater than zero and at most one, and is required unless
`calibrate: true`. Scores at or above it match. `action: drop` writes
`benchmark_overlap` drops; `action: flag` keeps the record and appends a flagged
event. Both store a typed `scores.decon_match` with benchmark, evaluation line
number, score, field, and normalized word offsets of the matching span.
`dropped.parquet` also exposes nullable `decon_benchmark` and `decon_score`
columns. `stats.json` reports per-benchmark match counts, flags, index size,
short skipped items, and unscorable items.

To select a threshold, first run with `calibrate: true` and no threshold. It
keeps records unchanged, plants case-swapped, whitespace-jittered, and one-word
dictionary substitutions in copies of the first conversation instruction
without indexed overlap, or an empty prefix if none is available,
and sweeps thresholds from 0.01 to 1.00. The report contains
`calibration_curve`, plant and input counts, and the highest
`recommended_threshold` achieving at least 0.99 planted recall, or null if none
does. `clean_rows_dropped` counts supplied input rows that would match; it is
not a verified false-positive count unless those rows are known clean.
Inspect those rows before applying a threshold. Exact n-grams can miss
substantial paraphrases. Published error rates are not guarantees for Arka.

#### `identity_claim`

Drops records containing foreign model or lab identity claims in the configured
conversation fields. Place it after generation and before LLM judging.

```yaml
- type: identity_claim
  check_fields: [response]
  own_model_names: [Arka-7B]
  patterns: []
```

`check_fields` accepts `instruction`, `response`, and `system`, and defaults to
`[response]`. Instructions do not trigger drops unless explicitly included.
`own_model_names` defaults to an empty list. A case-insensitive substring match
for any configured own name in any checked field lets the record pass.

An empty `patterns` list uses built-in first-person and creator pattern groups.
Creator patterns require a first-person subject, so ordinary discussion such as
"ChatGPT was developed by OpenAI" passes.
These cover GPT, ChatGPT, Claude, Gemini, Llama, Qwen, DeepSeek, Mistral, Mixtral,
Grok, PaLM, Bard, Bing, and Copilot, plus their listed labs. A non-empty list
replaces the built-ins and reports matches under the `custom` group. Invalid
regular expressions fail configuration validation. Patterns ignore case.

For matching only, Arka normalizes curly apostrophes and removes known-model
and known-lab denial clauses such as "I'm not ChatGPT" and "I wasn't trained
by OpenAI". A later positive claim still triggers a drop. Custom patterns are
responsible for denial handling for names outside the built-in list.

The filter preserves original text and drops with reason `identity_claim`.
`stats.json` reports `pattern_group_counts`, counting each matched group once
per dropped record. Pattern matches are approximate; inspect drops on your
generated data before relying on the defaults.

#### `artifact_filter`

Removes localized generation residue and drops records with unreliable text.
Place this stage immediately after generation, before deduplication and judging.

```yaml
- type: artifact_filter
  fields: [instruction, response]
  max_strips_per_record: 3
  group_action:
    openers: strip
    closers: strip
    self_reference: drop
    placeholders: drop
    tool_fingerprints: drop
```

`fields` defaults to `[instruction, response]` and also accepts `system`.
`max_strips_per_record` defaults to 3 and accepts 0 to prohibit stripping.
`groups` is a mapping from group names to lists of case-insensitive regular
expressions. Omit it to use the built-ins, or supply it to replace them.
The default groups are:

- `openers` matches "Certainly!", "Sure!", and "Of course!" at the start.
- `closers` matches "I hope this helps" and "Would you like me to...?" at the end.
- `self_reference` matches AI language-model and knowledge-cutoff disclaimers.
- `placeholders` matches specific unfilled tokens such as `[Your Name]` and `[link here]`.
- `tool_fingerprints` matches leaked citation markers.

`group_action` overrides the action for each named group. Unspecified openers
and closers are stripped; other groups, including custom groups, cause drops.
For example, a custom strip rule can be configured as:

```yaml
- type: artifact_filter
  groups: {custom: ['REMOVE_ME']}
  group_action: {custom: strip}
  fields: [response]
```

The matcher protects triple-backtick and triple-tilde code fences, inline code,
and double-quoted examples. It also protects an unclosed code fence through the
end of the field. This is a small literal-text guard, not a complete Markdown
parser. Check your own formatting and domains before relying on the defaults.

Stripping removes only matched spans. Overlapping spans count as one removal;
duplicate patterns do not count twice. The stage checks the updated text again
to catch repeated anchored openers. It drops with `artifact_strip_limit` before
exceeding the configured limit, or `artifact_empty_content` if stripping empties
an instruction or response. Other drops use `artifact_<group>` reason codes.
Invalid regexes and matches that consume no text raise an error.

Edits create copies and retain record IDs and lineage. The stage updates
`content_hash` so exact deduplication uses the cleaned payload. Each edit adds
a `stripped` event with field names and before/after lengths. `stats.json` reports
`group_match_counts`, `stripped_count`, `strip_match_count`, and drop counts.
Group counts include matches that cause drops. `stripped_count` counts edited
records, including records dropped after an earlier edit. `strip_match_count`
counts spans actually removed. If this stage is absent, records are unchanged.

#### `semantic_similarity`
Filters out generations that are too similar to their seed.
```yaml
- type: semantic_similarity
  threshold: 0.90
```

#### `sentence_variance`
Filters out responses with repetitive structures by checking sentence length variance.
```yaml
- type: sentence_variance
  min_cv: 0.15
```

#### `ifd`
Calculates and filters by Instruction-Following Difficulty (IFD) score.
```yaml
- type: ifd
  min_score: 0.20
```

#### `labeling_score`
Rates instruction/response pairs using a custom rubric.
```yaml
- type: labeling_score
  rubric_path: ./rubrics/sft_quality.yaml
  min_overall_score: 3.5
```

#### `reward_model_scoring`
Scores records with a designated reward model LLM.
```yaml
- type: reward_model_scoring
  min_score: 0.0
```

#### `pair_delta_filter`
Requires evolved pairs to improve over their parent by a minimum margin.
```yaml
- type: pair_delta_filter
  score_field: "quality"
  min_delta: 0.3
```

#### `select`
Ranks and selects top-N elements using composite scores.
```yaml
- type: select
  target_count: 100
  strategy: top_n
  weights:
    quality: 1.0
```

Add `coverage` to reserve minimum counts for explicit taxonomy paths before
filling the remaining slots by weighted score:

```yaml
- type: select
  target_count: 100
  weights: {quality: 1.0}
  coverage:
    - factor: topic
      path: [science, physics]
      min_count: 5
    - factor: format
      path: [question_answer]
      min_count: 10
  coverage_policy: fail_fast
  coverage_search_limit: 10000
```

`coverage` defaults to an empty list, which preserves the existing top-N behavior.
Each entry has a required `factor` and non-empty `path`. `min_count` defaults to
1. Paths exclude the taxonomy root. Path length specifies the protected depth;
descendant assignments count toward their parent path. Assignments come only
from `scores.quality_per_dim.taxonomy_nodes`. Missing assignments count toward
no minimum. A record can satisfy minimums for several factors or parent paths.

Selection first reserves records that cover the most unmet minimums, with score
order breaking ties. If this cannot fit the target, Arka searches combinations
of assignment groups. `coverage_search_limit` bounds this search to 10000 states
by default. A search limit is reported as `search_limit`, not as proof that the
constraints are infeasible. The reservation does not optimize the total score
across all feasible combinations. Remaining slots use the existing weighted
score, and score ties retain input order. Empty weights give equal scores.

`coverage_policy` is `fail_fast` by default. Unmet constraints stop the stage
after writing a diagnostic report. `best_effort` keeps the greedy reservation,
fills available slots, and reports the shortfall. The report contains
`taxonomy_coverage` with input and selected counts for each configured path,
`unmet_constraints`, `coverage_status`, and `score_difference_from_top_n`.
The score difference is the selected total minus the unconstrained top-N total.
A failed stage emits no selected records and reports a null score difference.

This control preserves requested representation. It does not infer labels or
guarantee improvements in the trained model.

---

## 5. Embeddings (`embeddings`)

Configures how diversity embeddings are calculated.

```yaml
embeddings:
  provider: huggingface
  model: all-MiniLM-L6-v2
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `provider` | `str` | `"huggingface"`| `"huggingface"` (local via FastEmbed) or `"openai"`. |
| `model` | `str` | `"all-MiniLM-L6-v2"`| The embedding model identifier. |

---

## 6. Labeling Engine (`labeling_engine`)

Settings governing the LLM-as-a-judge system when running a `labeling_score` stage.

```yaml
labeling_engine:
  rubric_path: ./rubrics/sft_quality.yaml
  mode: single
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `rubric_path` | `str` | `None` | Path to the rubric definition. |
| `mode` | `str` | `"single"` | Labeling mode: `"single"` or `"multi"`. |

---

## 7. Output (`output`)

Defines how and where the final fine-tuning dataset is written.

```yaml
output:
  format: chatml
  path: ./output/dataset.jsonl
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `format` | `str` | **Required** | Output schema format: `"jsonl"`, `"chatml"`, or `"alpaca"`. |
| `path` | `str` | **Required** | Target path where the dataset will be saved. |
