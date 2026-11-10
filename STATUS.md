# Status

Arka has an implemented CLI and a configurable pipeline for synthetic SFT data.
It supports seed and PDF ingestion, generation, deduplication, filtering,
decision scoring, bounded answer repair, distilled scoring, and source audits.
Runs save stage artifacts and SQLite checkpoints for resume.

Use `just check` for lint, formatting, and tests. Use `uv build` to check the
package build. Use `just verify` for installed CLI checks with local HTTP
fixtures. These fixtures do not establish production model quality.

See [features](docs/features.md) for supported behavior and
[verification paths](.agents/skills/verify-arka/features/README.md) for the
scope and prerequisites of runtime checks. The broader goals in
[scope](docs/SCOPE.md) include work that is not implemented.

The [project review](docs/project-review-2026-11-09.md) records confirmed fixes
and issues that need review.
