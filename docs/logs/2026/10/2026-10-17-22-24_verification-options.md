# Arka verification skill options

## Progress messages

I’ll use the create-verification-skill workflow to inspect Arka’s current CLI and verification tools. Then I’ll suggest a few options and recommend a scope before creating anything.

Arka already has a `verify-arka` skill. It focuses on the earlier cleanup PR stack and does not include the newer decision-model CLI check. I’d update that skill around current user workflows. I’m checking the local decision-model recipe to see how much we can reuse.

## Recommendation

Update the existing `.agents/skills/verify-arka/` skill. Start with deterministic local CLI verification. Keep provider checks and native model checks as explicit optional recipes.

Arka is a batch CLI. Readiness means the installed command works. Its verification helpers use loopback HTTP servers on OS-selected ports, with isolated run directories. There is no persistent app server to start.

## Options

1. Refresh the current skill. Replace old PR-specific instructions, add the decision CLI helper, and document current prerequisites. This has the smallest scope, but users must still choose among several helpers.
2. Add one local verification entry point and refresh the skill. Reuse the existing cleanup and decision helpers. Save a combined summary with commands, exit codes, source revision, and evidence paths. Keep native language and Hugging Face checks optional. This is the recommended balance.
3. Add local verification plus an explicit live provider recipe. Reuse the existing OpenRouter live helper. Record requests, usage, and provider responses. This adds evidence about credentials and provider compatibility, but requires credentials and can incur costs. It does not establish dataset quality.

## Proposed feature map

- Configuration validation and stage preview.
- Seed ingestion, PDF provenance, and generated text cleanup.
- Decision scoring, filtering, and selection.
- Checkpoints, resume, and request reuse.
- Dataset exports and reasoning preservation.

Define observable outcomes, such as two exported records, matching output after resume, and zero new provider requests. Keep old PR-head checks as historical recipes. Separate deterministic fixture results from live provider results and representative dataset audits.

## Evidence gathered

- `uv run arka --version` passed and reported version 0.1.0.
- `uv run arka --help` passed.
- `uv run python scripts/verify_decisions_cli.py` passed. It verified three input records, nine loopback requests, two exported records, and zero extra requests on resume.
- Evidence remains at `/tmp/arka-decisions-1e_kjim8/` after helper shutdown.
- The current decision helper chooses its own temporary output directory. It has no `--output` argument and does not save the resume subprocess log. A future common entry point should address those evidence gaps.
- No product code or verification skill was changed. The full test suite and build were not run for this options review.
