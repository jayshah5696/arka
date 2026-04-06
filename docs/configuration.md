# Configuration Guide

Arka uses a YAML configuration file to define its execution parameters. This document explains the available configuration options, which map to the internal Pydantic models.

## High-Level Structure

A full configuration file (`ResolvedConfig`) typically contains the following top-level keys:

```yaml
version: "1"
run_id: "optional-run-id"
llm: ...
executor: ...
data_source: ...
generator: ...
dedup: ...
filters: ...
embeddings: ...
labeling_engine: ...
output: ...
```

---

## 1. Top-Level Options

- `version` (`str`): The configuration format version (e.g., `"1"`).
- `run_id` (`str`, optional): An explicit identifier for the run. If not provided, one may be generated or passed via CLI.

---

## 2. LLM (`llm`)

Defines the connection to the Large Language Model provider.

- `provider` (`Literal["openai"]`): The LLM provider. Currently, only `"openai"` is supported (used for OpenAI and OpenAI-compatible providers like OpenRouter).
- `model` (`str`): The model identifier (e.g., `"gpt-4o"`, `"google/gemini-flash"`).
- `api_key` (`str`): The API key for authentication. Can use environment variable substitution like `${OPENAI_API_KEY}`.
- `base_url` (`HttpUrl`): The base URL for the API (e.g., `https://api.openai.com/v1`, `https://openrouter.ai/api/v1`).
- `timeout_seconds` (`float`, default: `30.0`): Timeout for API requests.
- `max_retries` (`int`, default: `3`): Number of times to retry a failed request.
- `supports_json_schema` (`bool`, optional): Whether the provider natively supports structured output/JSON schema.
- `openai_compatible` (`OpenAICompatibleConfig`, optional): Additional configuration for OpenAI-compatible providers (like OpenRouter).
  - `referer` (`HttpUrl`, optional): Used to identify the calling application to providers like OpenRouter.
  - `title` (`str`, optional): Used to identify the application title to providers.

---

## 3. Executor (`executor`)

Controls how tasks are executed concurrently.

- `mode` (`Literal["threadpool", "realtime", "provider_batch"]`, default: `"threadpool"`): The execution strategy.
- `max_workers` (`int`, default: `4`): The maximum number of concurrent workers/threads.

---

## 4. Data Source (`data_source`)

Defines where to get the initial seed data.

- `type` (`str`): The type of data source (e.g., `"seeds"`, `"csv"`, `"pdf"`).
- `path` (`str`, optional): File path to the data source. Required for most source types.
- `chunk_strategy` (`Literal["fixed"]`, optional): Strategy for chunking large documents (used for `"pdf"`). Defaults to `"fixed"`.
- `chunk_size_chars` (`int`, optional): Size of chunks in characters. Defaults to `3000` for PDFs.
- `chunk_overlap_chars` (`int`, optional): Overlap between chunks in characters. Defaults to `300` for PDFs.

---

## 5. Generator (`generator`)

Configures how synthetic data is generated from the seeds.

- `type` (`str`): The generator type (e.g., `"prompt_based"`, `"evol_instruct"`).
- `target_count` (`int`): The total number of items to aim for generating.
- `generation_multiplier` (`int`): Multiplier for generation to account for drops during deduplication/filtering.
- `prompt_template` (`str`): The prompt template used for generation. Can contain placeholders like `{seed_instruction}` and `{seed_response}`.
- `temperature` (`float`, default: `0.7`): The temperature for sampling.
- `max_tokens` (`int`, default: `512`): Maximum tokens to generate.
- `rounds` (`int`, optional): Number of evolution rounds (required for `"evol_instruct"`).
- `branching_factor` (`int`, optional): Number of branches per round (required for `"evol_instruct"`).
- `operators` (`list[str]`, optional): List of evolution operators to apply (required for `"evol_instruct"`).
- `filter` (`EvolFilterConfig`, optional): Filtering rules specific to Evol-Instruct.
  - `min_edit_distance_chars` (`int`, default: `20`): Minimum edit distance from the original.
  - `min_instruction_chars` (`int`, default: `20`): Minimum length of generated instruction.
  - `refusal_keywords` (`list[str]`, default: `["I cannot", "I'm unable", "As an AI"]`): Keywords that trigger rejection of a generation.

---

## 6. Deduplication (`dedup`)

Settings for detecting and removing duplicate generations.

- `exact` (`ExactDedupConfig`):
  - `enabled` (`bool`, default: `False`): Enable exact deduplication.
- `near` (`NearDedupConfig`):
  - `enabled` (`bool`, default: `False`): Enable near deduplication.
  - `shingle_size` (`int`, default: `5`): The shingle size for MinHash.
  - `num_hashes` (`int`, default: `128`): Number of hashes for MinHash signatures.
  - `jaccard_threshold` (`float`, default: `0.7`): Threshold above which items are considered duplicates.

---

## 7. Filters (`filters`)

Quality and property filters applied after generation. Contains a `target_count` and nested filter configs.

- `target_count` (`int`): The target number of valid items required after all filtering stages.
- `length` (`LengthFilterConfig`):
  - `enabled` (`bool`, default: `False`)
  - `min_instruction_chars` (`int`, default: `10`)
  - `max_instruction_chars` (`int`, default: `4096`)
  - `min_response_chars` (`int`, default: `10`)
  - `max_response_chars` (`int`, default: `16384`)
- `language` (`LanguageFilterConfig`):
  - `enabled` (`bool`, default: `False`)
  - `allowed` (`list[str]`, default: `["en"]`)
- `ifd` (`IFDFilterConfig`): (Instruction Following Difficulty)
  - `enabled` (`bool`, default: `False`)
  - `min_score` (`float`, default: `0.2`)
- `labeling_engine` (`LabelingFilterConfig`):
  - `enabled` (`bool`, default: `False`)
  - `rubric_path` (`str`, optional): Path to the evaluation rubric YAML file.
  - `min_overall_score` (`float`, optional): Minimum required overall score.

---

## 8. Embeddings (`embeddings`)

Settings for diversity embeddings.

- `provider` (`Literal["huggingface", "openai"]`, default: `"huggingface"`): Provider for computing embeddings.
- `model` (`str`, default: `"all-MiniLM-L6-v2"`): Embedding model identifier.
- `api_key` (`str`, optional): API key for external providers.
- `base_url` (`HttpUrl`, optional): Base URL for external APIs.
- `timeout_seconds` (`float`, optional)
- `max_retries` (`int`, optional)
- `openai_compatible` (`OpenAICompatibleConfig`, optional): Same as under `LLMConfig`.

---

## 9. Labeling Engine (`labeling_engine`)

Settings for the labeling/evaluation engine.

- `rubric_path` (`str`, optional): Path to a rubric definition for structured evaluation.
- `mode` (`Literal["single", "multi"]`, default: `"single"`): Labeling strategy (e.g., single judge vs. multiple).

---

## 10. Output (`output`)

Defines the output format and location for the final dataset.

- `format` (`Literal["jsonl", "chatml", "alpaca"]`): Output file format.
- `path` (`str`): File path where the output dataset will be saved.
