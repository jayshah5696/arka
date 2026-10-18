# PDF ingestion

## Sub-features

Exact chunk character slices, original page numbering across an empty page, zero overlap, generated lineage, and optional repeated-header removal with original-byte spans.

## How to get to it (user POV)

Set `pdf_source` in the config and run `uv run arka --config PATH --run-id NAME`. Enable `boilerplate_strip` to remove repeated blocks before chunking.

## Driving it with CLI

Run `just verify` for the default cleanup helper. It creates pages 1 and 3 with empty page 2 and checks chunk slices against extracted text. Inspect `cleanup/pdf/runs/pdf/stages/01_source/data.parquet` and the final dataset. Resume must preserve export bytes.

The optional native stack helper creates two pages with a repeated 600-byte header and checks 1,200 removed bytes, retained body text, and page provenance. Inspect its source Parquet, `stats.json`, and `boilerplate_spans.jsonl`. Use the shared skill's native prerequisites first.

## Gotchas

Cleanup character offsets refer to cleaned text. Removal spans refer to original trimmed-page UTF-8 bytes. Misaligned repeats can escape detection. An empty extraction is an error; cleanup below the remaining-byte threshold emits no chunks. Unit tests cover these boundaries; the CLI fixtures prove the positive ingestion paths.
