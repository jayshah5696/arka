# PR #1, #2, #3 — Feature PR Review

## PR #1 — Latent Density Sampling
**Verdict:** Close. Code quality too low to salvage.

## PR #2 — UMAP Visualization
**Verdict:** Close. Adds heavy deps as core dependencies. Revisit as optional extra.

## PR #3 — Privacy Guardrails
**Verdict:** Close as-is, but cherry-pick the useful core (config models + filter stages + tests). Strip committed run artifacts and mock_llm.py.
