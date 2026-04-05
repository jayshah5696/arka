# ADR 0001: Boundary modeling with Pydantic, internal execution containers with dataclasses

- Status: Accepted
- Date: 2026-04-04

## Context

Arka currently uses both Pydantic models and frozen dataclasses.

This is intentional, but until now it was only implicit in the code.
That made it reasonable for reviewers to question whether the mixed approach was accidental or drifting.

The project has two different kinds of structures:

1. **Boundary / schema-bearing models**
   - loaded from YAML / JSON
   - parsed from LLM output
   - serialized into artifacts
   - reconstructed from persisted data
   - expected to reject malformed or unexpected input

2. **Internal execution containers**
   - assembled entirely inside trusted application code
   - primarily used to move execution state between runner components
   - not user-authored inputs
   - not currently treated as canonical persisted schemas

These two categories have different needs.

## Decision

Arka will use the following rule:

- **Use Pydantic for boundary models.**
- **Use frozen dataclasses for internal execution containers.**

### Pydantic is the default for boundary models

Use Pydantic when a model:

- crosses into the system from config, YAML, JSON, CSV-derived structured data, or LLM output
- needs validation
- needs stable serialization/deserialization
- should reject unknown fields
- represents durable typed domain data

Current examples:

- `src/arka/config/models.py`
- `src/arka/records/models.py`
- `src/arka/llm/models.py`
- `src/arka/labeling/rubric.py`
- `src/arka/labeling/models.py`
- `src/arka/labeling/judges.py` response models

### Dataclass is acceptable for internal execution containers

Use frozen dataclasses when a type:

- is created only from already-validated internal state
- is lightweight runner plumbing
- does not currently require schema validation or parsing
- is not the main durable domain model

Current examples:

- `RunPaths`
- `StageContext`
- `RunResult`
- `StageStat`
- `OpenAIClientFactory`

## Why

### Why Pydantic at boundaries

Because this project depends heavily on:

- config validation
- typed record safety
- rubric and judge-output parsing
- artifact round-tripping
- explicit failure on malformed inputs

Pydantic gives us:

- strict schema validation
- `model_validate()`
- `model_dump()` / `model_dump_json()`
- field defaults and nested validation
- explicit rejection of unexpected fields via `extra="forbid"`

### Why dataclasses internally

Because not every internal container needs parsing or validation machinery.

Frozen dataclasses are a good fit for:

- clarity
- low ceremony
- immutability semantics
- simple internal transport objects

## Consequences

### Positive

- boundary safety remains strong
- internal runner structures stay lightweight
- the distinction between domain models and runner plumbing becomes intentional

### Tradeoff

The codebase is not fully uniform.
That means contributors must understand the rule, not just the syntax.

This ADR exists to make that rule explicit.

## Guardrails

### Promote a dataclass to Pydantic if it starts becoming schema-like

If an internal dataclass starts accumulating one or more of the following, it should be reconsidered:

- direct parsing from external input
- persistence as a canonical schema object
- rich nested optional fields
- validation rules
- cross-module serialization responsibilities

`StageStat` is the main current borderline case. It is still acceptable as a dataclass today, but if reporting grows much further it may be promoted to Pydantic.

### Prefer one shared strict Pydantic base

Where strict Pydantic behavior is desired, the codebase should converge on a shared strict base model instead of redefining `StrictModel` in multiple modules.

### Avoid duplicate construction paths

If an object like the OpenAI client has multiple construction paths, choose one canonical path rather than duplicating logic in several places.

## Deferred cleanup items

This ADR does not mandate immediate refactors, but it highlights these follow-ups:

1. create a shared strict base model
2. deduplicate OpenAI client construction
3. remove stale tests that still reflect old `dict`-based flows
4. keep reducing `Any` where concrete typed records can replace it

## Bottom line

The mixed use of Pydantic and dataclasses in Arka is **intentional and acceptable**.

The rule is:

- **Pydantic at boundaries**
- **frozen dataclasses in trusted internal execution plumbing**
