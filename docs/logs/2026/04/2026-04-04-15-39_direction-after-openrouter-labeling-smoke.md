# Direction after OpenRouter labeling smoke

Date: 2026-04-04 15:39

## Short answer

Yes, this is a meaningful milestone.

A real LLM-backed pipeline path now exists:
- source stage
- normalize stage
- labeling filter stage
- live OpenRouter-backed smoke validation

That means the project has moved beyond scaffold-only status.

## Should we continue in this direction?

**Yes — but with a narrowing move, not a broadening move.**

The right direction is:

> **finish the single-judge quality filtering path properly before expanding into multi-judge or more provider complexity**

So the direction should be:
1. harden the current single-judge filter pipeline
2. make artifacts/reporting/drop handling correct
3. only then move into multi-judge / conflict detection

## Why this is the right direction

Because right now the project has the first true vertical path that resembles the approved product:
- real source input
- typed records
- real LLM scoring
- filtering based on score
- end-to-end artifact output

This is exactly the moment where teams often make the wrong move and branch outward too early into:
- multi-judge
- more providers
- more routing options
- more abstractions

That would be premature.

The more valuable move is to turn the current path into a trustworthy one.

## What is good about the current direction

### 1. Real smoke validation beats more abstraction
A live run against OpenRouter/Gemini tells you much more than another round of internal scaffolding.

### 2. LabelingEngine integration is the right product-pressure test
It forces:
- rubric loading
- prompt construction
- parsing robustness
- score propagation into records
- real filtering behavior

That is excellent pressure on the architecture.

### 3. The pipeline is now close enough to user value to justify hardening
At this point, improving output trustworthiness matters more than adding breadth.

## What should NOT be the next direction

### Do not go to multi-judge yet
Not yet.

Reason:
- the single-judge path still lacks proper dropped-record persistence and reporting integration
- without that, multi-judge will increase complexity faster than confidence

### Do not prioritize more OpenRouter/provider flexibility yet
The OpenRouter smoke run was useful as validation, but routing flexibility should now become background, not foreground.

### Do not jump to preference data yet
DPO/pairwise work will build on the same labeling/judging foundation. Harden that first.

## Recommended next steps (ordered)

### Step 1 — Finish the quality filter stage as a first-class artifact producer
Implement:
- `dropped.parquet` for low-quality records
- explicit drop reason codes such as:
  - `low_quality_score`
  - `label_parse_failure`
  - `rubric_load_error` (if handled as non-fatal in future)
- stage-level kept/dropped stats

Why first:
- this turns the filter from “functional” into “inspectable”
- it aligns directly with the approved spec

### Step 2 — Push filter stats into `run_report.json`
Add:
- count in / count out / dropped for label stage
- score distribution summary
- maybe mean/std/min/max of `scores.quality`

Why second:
- once a filter starts discarding real examples, you need observability

### Step 3 — Separate manifest vs report roles more clearly
Current run artifacts are still close together.

Define:
- `manifest.json` = machine-oriented artifact ledger
- `run_report.json` = human-oriented pipeline summary

### Step 4 — Improve error taxonomy in the label path
Current structured parsing is more robust now, which is good.
Next, make failures explicit in a way that the filter stage can reason about:
- retryable API failure
- auth failure
- parse failure
- invalid structured response

This matters before multi-judge.

### Step 5 — Only then start Slice 3 style multi-judge/conflict detection
Once single-judge quality filtering is observable and trustworthy, then multi-judge becomes worth the extra complexity.

## Concrete recommendation

### Recommended immediate direction

**Continue with quality-filter hardening.**

Specifically:
1. persist dropped records from `LabelingQualityFilterStage`
2. record per-stage quality filter stats
3. surface these in `run_report.json`
4. improve label-path error handling

### Recommended direction after that

Then move to:
- multi-judge / conflict detection

### Not recommended yet

- broader provider expansion
- preference pair generation
- trajectory data
- more transport/routing flexibility

## Bottom line

The current direction is correct.

But the correct next move is **depth, not breadth**.

You now have a real working path. Make that path trustworthy and inspectable before adding more judge sophistication.
