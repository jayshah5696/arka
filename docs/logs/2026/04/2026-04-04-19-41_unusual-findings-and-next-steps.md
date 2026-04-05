# Unusual findings and next steps

Date: 2026-04-04 19:41

## Overall

The project is progressing well. The direction is still good, and the recent cleanups improved consistency.

There are no alarming architectural regressions. But there are a few **unusual or noteworthy things** that stand out now.

## Unusual / noteworthy findings

### 1. The repo is moving fast enough that docs and code are slightly out of sync
Example:
- `README.md` says `LabelingEngine single-judge scaffold`
- the code already has a working single-judge quality filter integrated into the pipeline and live OpenRouter verification logs

This is not bad, but it means the docs are now lagging the real implementation slightly.

### 2. OpenRouter/Gemini is becoming the de facto real test path even though OpenAI is still the canonical provider model
This is practical and useful, but it is a subtle drift worth noticing.

Current state:
- config model still says `provider: Literal["openai"]`
- OpenRouter is being used via OpenAI-compatible mode
- live verification uses Gemini through OpenRouter

That is okay operationally, but it means the implementation story is really:
> OpenAI-shaped client interface with OpenRouter-backed practical usage

Not a problem — just worth acknowledging explicitly.

### 3. Duplicate OpenAI client construction still exists
This remains one of the clearest code smells.

Both exist:
- `src/arka/llm/openai_client.py`
- `_build_openai_client()` inside `src/arka/llm/client.py`

This is now more noticeable because the rest of the codebase has become cleaner.

### 4. `StageStat` is drifting toward a boundary model
It is still a dataclass, but now contains:
- `dropped_count`
- `drop_reasons`
- `quality_distribution`

This is still okay, but it is starting to look more like a schema than a tiny internal struct.

### 5. Label stage hardening is getting good, but failure taxonomy is still the next real missing piece
You now have:
- dropped low-quality examples persisted
- stats and quality distribution
- live verification

So the next missing layer is no longer basic functionality.
It is explicit failure classification and handling.

### 6. Config example work is becoming a real project surface
This is good, but now there are enough config files that they need to be treated as a maintained interface, not just helpers.

Current shape is sensible:
- canonical root configs
- commented example configs
- catalog doc

This should be preserved as a first-class documentation asset.

---

## What is next

## Best next step

### Implement label-path failure taxonomy and persistence

This is the highest-value next move.

Why this is next:
- the single-judge path already works in happy-path mode
- low-quality dropping is already implemented
- what is missing now is making failures inspectable instead of opaque

### Concretely, add reasoned handling for cases like:
- `label_parse_failure`
- `label_auth_failure`
- `label_retryable_api_failure`
- `invalid_structured_response`

And decide policy for each:
- drop and persist?
- fail whole run?
- retry?
- classify as warning?

This will make the current quality filter path trustworthy enough to support future multi-judge work.

---

## What should come after that

### 1. Deduplicate OpenAI client construction
This is now a worthwhile cleanup because the surrounding architecture has stabilized enough.

### 2. Clarify `manifest.json` vs `run_report.json`
They are still close. Once failure taxonomy lands, it will be easier to separate machine metadata from human summary.

### 3. Then move to multi-judge/conflict detection
Only after the single-judge path has:
- score persistence
- dropped-record persistence
- stage stats
- failure taxonomy

At that point, multi-judge becomes additive instead of premature complexity.

---

## What I would NOT do next

Do not prioritize:
- more provider expansion
- trajectory data
- preference pairs
- PDF ingestion deepening
- additional routing flexibility

Those are all lower value than finishing the current quality filter path properly.

---

## Bottom line

Nothing looks unusually bad.

The only notable unusual pattern is that the project is becoming **OpenAI-interface canonical but OpenRouter/Gemini practically exercised**. That is okay as long as you stay honest about it.

### Recommended next step

> **Finish label-path failure taxonomy and failure artifact handling.**

That is the current highest-leverage move.
