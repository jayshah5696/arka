# Boundary modeling decision

Date: 2026-04-04 19:32

Added an architecture decision record to make the Pydantic/dataclass split explicit.

Created:

- `docs/decisions/0001-boundary-modeling.md`

Decision summary:

- use Pydantic for boundary models
- use frozen dataclasses for internal execution containers

The ADR also documents:

- why boundary validation matters in Arka
- when a dataclass should be promoted to Pydantic
- the current borderline case (`StageStat`)
- nearby cleanup opportunities:
  - shared strict base model
  - deduplicated OpenAI client construction
  - removal of stale old-style dict tests
