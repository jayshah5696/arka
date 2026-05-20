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

#### `canary`
Filters out generations containing sensitive phrases or secrets.
```yaml
- type: canary
  phrases: ["SECRET_TOKEN"]
```

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
