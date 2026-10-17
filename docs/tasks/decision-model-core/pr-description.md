## Why

Arka's stages need a shared way to ask typed decision questions and reuse answers. This adds a decision client and stage integrations with no automatic LLM fallback. A judge panel shares one request per record across compatible later stages, while the inverse critic always runs separately.

## Changes

- Add OpenAI, OpenRouter, TypeSafe/Jev, and Clef adapters, typed questions and answers, per-answer SQLite caching, retries, worker limits, token accounting, and stage overrides.
- Connect decisions to seed and PDF checks, best-of-N candidate selection, evolution checks and operator selection, taxonomy fidelity, language and identity checks, borderline semantic similarity, and benchmark paraphrase checks.
- Add decision backends for rubric labeling, double critic, complexity, and reward scoring. Selection reads decision scores and confidence and can label taxonomy membership before coverage reservation. Pair checks can compare both A/B orders.
- Add keep, drop, and escalate policies, an escalation queue, output handling for unresolved records, and decision usage in reports and the CLI summary.
- Preserve scores from earlier stages during labeling. Parallelize prompt generation while writing raw responses in plan order for resume.
- Add the configuration guide, generated schema, runnable example, transport and integration tests, a local HTTP CLI helper, and a bounded live experiment script.

## Reviewer notes

The comparison base is `main`. The branch includes the three latest CLI changes from `main`. Existing non-decision backends remain the defaults. Decision mode never falls back to a generative judge. Deterministic filters and exact/MinHash dedup keep their existing implementations.

`httpx` becomes a direct dependency. The OpenAI SDK is not upgraded. API keys are excluded from resolved configs. Provider-reported cost takes precedence over an optional input-token price estimate. OpenRouter uses its alpha Decisions endpoint and the Jev question map. `executor.mode: provider_batch` now fails validation because it has no implementation.

This PR publishes the core functionality and a small live OpenRouter experiment. Representative agreement/speed/cost comparisons and the calibration-audit stage are not completed here. The existing LLM reward backend's separate parsing and scale improvements are also outside this implementation. Escalations remain unresolved for review; no automatic LLM resolution is added. No issue-closing keywords are used because the seven issues contain broader acceptance criteria.

Related: #199, #200, #202, #205, #206, #210, #214.

## Verification

Checked implementation head `33c198f`, including current `main` at `5c251ef`:

- `just check`: lint and formatting passed; 543 tests passed. The optional native fastText integration test was skipped.
- `uv build`: wheel and source distribution built.
- `just validate-examples` and `just dry-run-examples`: all nine examples passed.
- `uv run python scripts/verify_decisions_cli.py`: production CLI and HTTP transport passed against a loopback fixture. Three input records made nine requests: three seed requests, three shared judge-panel requests, and three separate inverse-critic requests. Later judges reused cached answers. Selection exported two records, resolved config omitted fixture secrets, and resume made zero extra HTTP requests.

- Live OpenRouter Decisions API using `openai/gpt-6-luna-decisions`, served as `openai/gpt-6-luna-decisions-20261006`: yes/no, choice, and score answers parsed successfully. Six synthetic records used 12 panel requests versus 30 separate judge requests. All 36 typed answers matched exactly. Filtering retained all three correct responses and dropped all three wrong responses. Cache reuse and resume made zero new requests. Instrumented runs made no LLM calls.
- Live installed CLI: a substantive seed passed seed gating, language, and identity stages and exported one record. A short seed followed the configured drop-on-uncertainty policy. Total live requests across all experiments were 47, with $0 cost reported by OpenRouter. This does not establish future pricing or production accuracy.
- `uv run python scripts/verify_cleanup_cli.py --output /tmp/arka-cleanup-20261017-live-followup`: PDF and taxonomy CLI checks passed, with 14 loopback calls and zero extra calls on resume.

The helpers and [experiment report](https://github.com/jayshah5696/arka/blob/c-b/decision-model-core/docs/verification/2026-10-17-openrouter-decisions.md) are committed. The report includes measured results and remaining verification limits. Live generation, semantic, contamination, pairwise, and taxonomy checks remain to be measured on representative inputs.
