# Config examples and model rationale

Date: 2026-04-04 19:18

Created a first-pass config example catalog and prepared an explanation for why the codebase mixes dataclasses and Pydantic models.

## Added config example catalog

Created:

- `docs/config-examples.md`
- `config.examples.verify-openrouter.yaml`
- `config.examples.resume-openrouter.yaml`
- `config.examples.csv-seeds.yaml`

Also updated:

- `README.md` to point at the config example catalog
- `tests/unit/test_example_configs.py` to ensure the new example configs load through `ConfigLoader`

## Why dataclass in some places and Pydantic in others

Current design split:

### Use Pydantic when

The object crosses a boundary or needs schema enforcement / serialization:

- config models (`src/arka/config/models.py`)
- record models (`src/arka/records/models.py`)
- rubric / judge response / label result models
- LLM response models

Reasons:

- parse external YAML / JSON safely
- validate shape and types
- reject unknown fields (`extra="forbid"`)
- use `model_validate()` and `model_dump()`
- round-trip through artifacts cleanly

### Use dataclass when

The object is an internal, already-trusted container with no parsing burden:

- `RunPaths`
- `StageContext`
- `RunResult`
- `StageStat`
- `OpenAIClientFactory`

Reasons:

- lighter-weight
- simpler syntax for internal transport objects
- no validation overhead needed once config is already validated
- `frozen=True` gives immutability semantics cheaply

## Core rationale

The split is intentional:

- **Pydantic at boundaries**
- **dataclasses inside the trusted core for simple containers**

So:

- external / persisted / schema-heavy data => Pydantic
- internal execution context / derived bookkeeping => dataclass

## Caveat

There is a consistency tradeoff: a pure-Pydantic design would be more uniform.
A hybrid design is more pragmatic and cheaper at runtime for internal-only objects.

If desired later, `StageContext`, `RunResult`, and `StageStat` could be converted to Pydantic for uniformity. Right now the current mix is reasonable because those types are not parsed from user or model output.
