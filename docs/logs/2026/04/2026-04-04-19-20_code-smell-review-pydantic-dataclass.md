# Code smell review: Pydantic vs dataclass and related consistency issues

Date: 2026-04-04 19:20

## Short answer

Yes — the mixed use of **Pydantic models** and **dataclasses** is worth reviewing.

It is **not automatically a bug or bad design**, but in this repo there are a few places where the boundary is currently inconsistent and may become a maintenance smell if not made intentional.

## High-level judgment

### Using both is acceptable when the roles are different
A healthy split is:
- **Pydantic** for validated external or semi-external data
  - config
  - records
  - rubric/judge outputs
  - LLM outputs
- **dataclass** for small internal transport structs
  - path containers
  - runner return values
  - simple immutable stage stats/context

That is a reasonable design.

### But right now the repo has some fuzzy boundaries
The main smell is not “dataclass + Pydantic exist together.”
The main smell is:

> some structures are treated like validated domain models, while nearby ones are treated like lightweight containers even though they participate in the same workflow.

That is where confusion can grow.

---

## What currently looks acceptable

### Good use of Pydantic
These are clearly good fits for Pydantic:
- `src/arka/config/models.py`
- `src/arka/records/models.py`
- `src/arka/llm/models.py`
- `src/arka/labeling/rubric.py`
- `src/arka/labeling/models.py`
- judge response models

Why:
- they are schema-bearing
- they ingest structured data
- they serialize/deserialize
- validation matters

### Good use of dataclass
These are acceptable as dataclasses:
- `RunPaths`
- `RunResult`
- `StageStat`
- `OpenAIClientFactory`

Reason:
- these are mostly internal containers
- they are not user-authored inputs
- they do not need heavy validation in current form

---

## The main smell areas

## 1. `StageContext` as dataclass is okay, but borderline
`StageContext` currently carries:
- `run_id`
- `stage_name`
- `work_dir`
- `config: ResolvedConfig`
- `executor_mode`
- `max_workers`

This is fine as a dataclass for now.

But it sits right at the boundary between:
- validated config world
- runner orchestration world
- stage execution world

So it is not wrong, just a borderline case.

### Recommendation
Keep it as a dataclass for now.
Do not change it unless it starts accumulating:
- optional fields
- serialization behavior
- validation rules
- nested runtime metadata

Then promote it to Pydantic.

## 2. `StageStat` as dataclass is also borderline
This one is a bit smellier than `StageContext`.

Why:
- it is serialized into manifest/report
- it includes optional nested structured fields like `drop_reasons` and `quality_distribution`
- it is starting to behave like a schema, not just a transport object

This is still acceptable, but if `StageStat` keeps growing, it should probably become a Pydantic model.

### Recommendation
- okay for now
- if more report fields get added, migrate `StageStat` to Pydantic instead of continuing to expand the dataclass

## 3. `OpenAIClientFactory` as a dataclass adds little value
This is the weakest dataclass use.

Current structure:
- frozen dataclass with one field: `config`
- one `build()` method

This is not harmful, but it is not buying much either.
It could just be:
- a plain class
- or a function

### Recommendation
Not urgent, but this is a mild smell of over-structuring.

---

## Bigger smells than dataclass/Pydantic mixing
These matter more.

## 1. Repeated `StrictModel` definitions across modules
This is a clearer code smell than the dataclass split.

You currently redefine variants of:
- `StrictModel(BaseModel)`
- `model_config = ConfigDict(extra="forbid")`

in multiple places.

That creates drift risk.

### Recommendation
Create one shared base, e.g.:
- `src/arka/common/models.py`

with:
- `StrictModel`

Then import it everywhere.

This is a high-value cleanup.

## 2. Repeated OpenAI client construction logic
OpenAI client setup logic appears in both:
- `src/arka/llm/client.py`
- `src/arka/llm/openai_client.py`

That is a real duplication smell.

### Recommendation
Choose one source of truth:
- either `OpenAIClientFactory.build()` becomes canonical
- or `_build_openai_client()` inside `LLMClient` becomes canonical

But do not maintain both.

This is one of the clearest current design smells.

## 3. Generic `Any` still shows up in important seams
Examples:
- `Record.payload: dict[str, Any]`
- `RecordScores.quality_per_dim: dict[str, Any]`
- `LabelingQualityFilterStage(... llm_client: Any | None = None)`
- several loader/output helpers

Some of this is unavoidable in scaffold phase, but it is still a smell to monitor.

### Recommendation
Biggest priority is not removing all `Any` now.
Biggest priority is pushing real stage flows toward typed concrete record classes.

## 4. Tests still contain a stale type regression
In `tests/unit/test_pipeline_runner.py` there is still a resumed source test using:
- `list[dict]`

That is a small but meaningful smell because it reflects the earlier architecture.

### Recommendation
Update stale tests so the test suite reinforces the new typed model, not the old dict model.

## 5. `Record.payload` base-type looseness remains the biggest architectural smell
This is still the biggest one.

Yes, `ConversationRecord` is typed.
But the base `Record` still allows generic payload dicts, which means pipeline code can quietly remain generic.

That is tolerable now, but should not become permanent.

---

## Summary judgment on the Pydantic + dataclass question

### Did we design like this?
**Yes, directionally this mixed design is defensible.**

The intended split appears to be:
- Pydantic for validated data models
- dataclass for internal small immutable containers

That is a valid architecture.

### Is it currently a smell?
**Mildly, in a few places — but not because mixing them is wrong.**

The actual smells are:
1. duplicated base model definitions
2. duplicated OpenAI client construction
3. generic `Any` / dict seams still lingering
4. `StageStat` drifting toward schema territory while still being a dataclass

---

## Recommended cleanup priority

### High priority
1. deduplicate `StrictModel`
2. deduplicate OpenAI client construction path
3. update stale tests that still use `list[dict]`

### Medium priority
4. watch `StageStat`; convert to Pydantic if it grows more
5. reduce `Any` in important seams as concrete record types expand

### Low priority
6. consider simplifying `OpenAIClientFactory`

---

## Bottom line

The repo does **not** currently look badly designed just because it uses both dataclasses and Pydantic.

But the review did surface some real consistency smells nearby.

If I had to name the top two cleanup targets right now, they would be:
- **shared `StrictModel` base**
- **single source of truth for OpenAI client construction**
