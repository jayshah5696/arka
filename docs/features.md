# Arka Features

Arka (अर्क) is a config-driven synthetic data generation framework. It transforms seed data through multiple pipeline stages to output high-quality datasets suitable for supervised fine-tuning (SFT).

This document provides a high-level overview of the main features and pipeline stages implemented in Arka.

## 1. Data Ingestion & Normalization

Arka can ingest seed data from various source formats and normalize them into a uniform structure.

- **Supported Formats:**
  - **JSONL Seeds:** Pre-formatted instruction/response pairs in JSON lines format.
  - **CSV Seeds:** Tabular data ingestion.
  - **PDF Sources:** Supports extracting and chunking text from PDF files using configurable chunk sizes and overlaps.

## 2. Generation Strategies

The generation phase creates new synthetic examples based on the ingested seeds.

- **Prompt-Based Generation:** Uses a customizable prompt template to ask a Large Language Model (LLM) to generate a new, self-contained instruction-response pair inspired by the seed example.
- **Evol-Instruct:** A multi-round generation strategy that incrementally makes instructions more complex or specific. Supports setting branching factors, rounds, and specific evolution operators.
- **LLM Support:** Compatible with OpenAI and other OpenAI-compatible APIs (e.g., OpenRouter). Ensures structured output using JSON schemas.

## 3. Deduplication

Prevents the dataset from being flooded with near-identical generations.

- **Exact Deduplication:** Filters out generations that match an existing record character for character.
- **Near Deduplication:** Employs MinHash and Locality Sensitive Hashing (LSH) (or similar techniques via shingles and Jaccard similarity) to identify and remove generations that are structurally or semantically very similar to existing ones.

## 4. Quality Filtering

Applies multiple checks to ensure the generated data meets quality standards.

- **Length Filters:** Enforces minimum and maximum character lengths for both instructions and responses.
- **Language Filters:** Ensures the output strictly conforms to a specified list of languages (e.g., english only).
- **IFD (Instruction Following Difficulty):** Filters out records that fall below a configured minimum IFD score.
- **Labeling/Rubric Filters:** Integrates with an LLM-as-a-judge (single or multi-judge) scoring mechanism based on a customizable YAML rubric. Filters out items failing to meet a minimum overall score.
- **Evol Filters:** Specifically for Evol-Instruct, discards generations that trigger refusal keywords or fall below minimum edit distances compared to previous rounds.

## 5. Embeddings & Diversity

To calculate similarity beyond text matching, Arka calculates diversity embeddings.

- **Embedding Models:** Supports computing embeddings via HuggingFace (e.g., `all-MiniLM-L6-v2` via FastEmbed) or external providers like OpenAI.

## 6. Execution & Checkpointing

Designed to handle robust generation runs.

- **Execution Modes:** Supports configurable concurrency via thread pools.
- **Resumable Runner:** Uses SQLite checkpoints, allowing runs to be resumed if interrupted.
- **Rich Artifacts:** Outputs `data.parquet`, `dropped.parquet`, `clusters.parquet`, `stats.json`, `manifest.json`, `run_report.json`, `samples.jsonl`, and `canaries.json` inside the `runs/<run_id>/` directory for deep inspection.

## 7. Output Export

Final datasets can be formatted specifically for popular training paradigms.

- **Formats Supported:**
  - `jsonl`
  - `chatml`
  - `alpaca`
