# Continued project review

Date: 2026-04-04 15:13

## Overall status

The project is materially healthier now. The implementation has moved closer to the approved spec and no longer looks like a generic Python scaffold pretending to be the target system.

Big improvement areas:
- typed records now exist in code
- stage protocol exists
- runner emits stage events
- minimal run report exists
- temporary defers are explicitly documented

So the trend is good.

## What is strong now

### 1. `records/` extraction was the right move
Creating `src/arka/records/models.py` is a strong correction. This gives the project an actual internal data model rather than spreading schema concerns across the runner and tests.

### 2. Stage protocol is simple and appropriate for current scale
`src/arka/pipeline/stages.py` is intentionally small, which is good for the current target size. No need for a bigger abstraction yet.

### 3. `OutputWriter` shape is practical
The storage strategy is understandable:
- JSONL for exported payloads
- Parquet for internal stage artifacts
- typed records restored from parquet rows

That matches the design intent well enough for v0.

### 4. `temporary-defers.md` is a good discipline signal
This is one of the strongest meta-engineering moves in the repo. Explicit defers are much better than accidental drift.

---

## Remaining architectural mismatches / concerns

## 1. `Record.payload` in the base class is still too loose
Current:
- `Record.payload: dict[str, Any]`
- `ConversationRecord.payload` is typed

This is okay as a transition, but it means most of the system can still silently fall back to generic dict payloads.

Why this matters:
- if future stages operate mostly on base `Record`, schema quality will erode again
- pipeline code will accumulate payload-key string lookups instead of type-aware logic

Recommendation:
- okay for scaffold phase
- but once the source stage is implemented, move quickly toward using concrete record types in real stage flows, not generic `Record`

## 2. Resume semantics are functionally okay but not analytically clean
In resume mode, the manifest stage stats currently show:
- `count_in == count_out == len(restored stage output)`
- `status = resumed`

This is acceptable for a thin scaffold, but it does not preserve the original pre-resume execution facts.

That means the manifest becomes a report of the *resume session*, not a stable audit of the original stage execution.

Recommendation:
- short term: acceptable
- medium term: load original stage counts from SQLite stage_runs instead of inventing resumed counts from current file length

## 3. `run_report.json` is still very close to `manifest.json`
This is not wrong, but currently the two artifacts are almost duplicate documents.

That suggests one of two futures:
- either keep manifest machine-oriented and report human-oriented
- or merge them and avoid artifact duplication

Recommendation:
- for now okay
- but before Slice 2/3 grows, define sharper separation:
  - `manifest.json` = run metadata + artifact registry
  - `run_report.json` = human-facing run summary

## 4. `StageEvent.action = "transformed"` for every stage is too blunt
This is understandable for now, but it weakens the usefulness of stage events.

For example, source stages are not really “transformed”; they are more like `created` or `emitted`.
Resume cases are not represented on records at all.

Recommendation:
- not urgent
- but source stage should likely emit `created` or `emitted`
- future scoring/dedup/drop paths should use semantically distinct actions

## 5. `LLMConfig.provider` still hardcodes only `openai`
This is acceptable only because the defer is now documented.
Still, this is a real divergence from the approved spec, which intended OpenAI + Anthropic in Slice 1.

Recommendation:
- okay for this commit range
- but do not let this sit too long if LabelingEngine work begins
- multi-judge work especially will want at least the adapter boundary ready

## 6. `OutputWriter.write_jsonl()` exports payload only
This matches the current plan for final dataset export, but it is worth noting that it drops provenance and scores from final export.

That may be correct for training-ready JSONL, but you may later want:
- a final training export
- and a richer retained artifact dataset with metadata

Recommendation:
- not a problem now
- just make sure later slices preserve a richer final Parquet artifact even if JSONL export stays payload-only

---

## Code-quality observations

### Good
- tests are targeted and readable
- package layout is clean
- `ConfigDict(extra="forbid")` is the right discipline
- `record_model_for_name()` is a useful little restoration hook

### Watch-outs
- there are still `dict`-heavy seams in tests and payload handling
- `RecordScores.quality_per_dim: dict[str, Any]` is broader than ideal
- some stage/test code still relies on generic base `Record` where concrete record types would be stronger

---

## What I would focus on next in review terms

If continuing implementation, the best next checkpoint is still:

### source stage + smoke pipeline

Why:
- it will force the first real typed record flow
- it will expose whether the current record/storage abstractions are ergonomic or annoying
- it will validate whether manifest/report/artifact layout is actually useful in practice

This will teach more than jumping straight to LabelingEngine.

---

## Bottom line

The project is improving in the right direction.

It is now:
- structurally credible
- disciplined enough to keep building
- still carrying a few scaffold simplifications, but not in a dangerous way

Main thing to watch:
> don’t let the generic base `Record` and generic payload dicts become the permanent operating model.

That is the biggest remaining architectural risk.
