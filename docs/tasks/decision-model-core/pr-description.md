## Why

Arka's stages need a shared way to ask typed decision questions and reuse answers. This adds a decision client and stage integrations with no automatic LLM fallback. A judge panel shares one request per record across compatible later stages, while the inverse critic always runs separately.

## Changes

- Add OpenAI, TypeSafe/Jev, and Clef adapters, typed questions and answers, per-answer SQLite caching, retries, worker limits, token accounting, and stage overrides.
- Connect decisions to seed and PDF checks, best-of-N candidate selection, evolution checks and operator selection, taxonomy fidelity, language and identity checks, borderline semantic similarity, and benchmark paraphrase checks.
- Add decision backends for rubric labeling, double critic, complexity, and reward scoring. Selection reads decision scores and confidence and can label taxonomy membership before coverage reservation. Pair checks can compare both A/B orders.
- Add keep, drop, and escalate policies, an escalation queue, output handling for unresolved records, and decision usage in reports and the CLI summary.
- Preserve scores from earlier stages during labeling. Parallelize prompt generation while writing raw responses in plan order for resume.
- Add the configuration guide, generated schema, runnable example, transport and integration tests, and a local HTTP CLI verification helper.

## Reviewer notes

The comparison base is `main`. The branch includes the three latest CLI changes from `main`. Existing non-decision backends remain the defaults. Decision mode never falls back to a generative judge. Deterministic filters and exact/MinHash dedup keep their existing implementations.

`httpx` becomes a direct dependency. The OpenAI SDK is not upgraded. API keys are excluded from resolved configs. Input-token cost is reported only when a price is configured. `executor.mode: provider_batch` now fails validation because it has no implementation.

This PR publishes the core functionality. Live provider smoke tests, representative agreement/speed/cost comparisons, and the calibration-audit stage are not completed here. The existing LLM reward backend's separate parsing and scale improvements are also outside this implementation. Escalations remain unresolved for review; no automatic LLM resolution is added. No issue-closing keywords are used because the seven issues contain broader acceptance criteria.

Related: #199, #200, #202, #205, #206, #210, #214.

## Verification

Checked implementation head `2deb372` after merging current `main`:

- `just check`: lint and formatting passed; 537 tests passed. The optional native fastText integration test was skipped.
- `uv build`: wheel and source distribution built.
- `just validate-examples` and `just dry-run-examples`: all nine examples passed.
- `uv run python scripts/verify_decisions_cli.py`: production CLI and HTTP transport passed against a loopback fixture. Three input records made nine requests: three seed requests, three shared judge-panel requests, and three separate inverse-critic requests. Later judges reused cached answers. Selection exported two records, resolved config omitted fixture secrets, and resume made zero extra HTTP requests.

The CLI helper is committed and can reproduce the fixture check. It makes no paid provider calls. Fixture results establish transport and pipeline behavior, not provider accuracy or calibration.
