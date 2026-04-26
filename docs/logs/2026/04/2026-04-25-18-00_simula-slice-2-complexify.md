# Simula Slice 2 — Complexification Operator

**Date:** 2026-04-25
**Branch:** `wt/simula`
**Slice:** 2 — Simula §2.2 "complexify" as a new Evol-Instruct operator.

## What landed

`complexify` is now a registered operator in the existing Evol-Instruct registry
(`src/arka/pipeline/evol_instruct.py`). Adding it required exactly two things:

1. Append `"complexify"` to `SUPPORTED_EVOL_OPERATORS`
2. Add the prompt to `_OPERATOR_PROMPTS`

Everything else — config validation, dispatch loop, lineage tagging
(`lineage.operator = "complexify"`), parent-deduplication, refusal filtering —
came for free from arka's existing operator infrastructure. **No production
code changed beyond those 12 lines of additions.** That is the cleanest
extension we've shipped yet.

The operator prompt is rigid by design and has three locks:

- "PRESERVING the original task type, the original topic, and the output format"
- "Pick exactly ONE complexification axis" (constraint / edge case / depth / extra reasoning step)
- "Do NOT switch to a different topic; that is the job of breadth_mutation"

These distinguish complexify from breadth_mutation (which is supposed to drift).

### Files

**New:**
- `tests/unit/test_complexify_operator.py` — 6 tests
- `tools/simula_eval/configs/02-complexify.yaml` — slice config (evol_instruct, rounds=1, only complexify)

**Modified:**
- `src/arka/pipeline/evol_instruct.py` — registry entry + prompt template

### Tests (red→green)

```
6 passed
- complexify is registered in SUPPORTED_EVOL_OPERATORS
- prompt asks for harder version
- prompt explicitly preserves task & forbids topic switching
- prompt has correct shape (user role, JSON 'instruction' key)
- config layer accepts operators=['complexify']
- end-to-end: EvolInstructRoundStage records lineage.operator='complexify'
```

Full suite: **275 passing** (was 269, +6). Lint clean.

## Live run on humanize-rl seeds

Same 50 seeds, same model (`gemini-3.1-flash-lite-preview`), same dedup +
filter chain as slice 0. Generator swapped from `prompt_based` to
`evol_instruct, rounds=1, branching_factor=3, operators=[complexify]`.

```
| stage             | slice 0 (baseline) | slice 2 (complexify)         |
|-------------------|--------------------|------------------------------|
| generator         | 50 → 300 (prompt)  | 50 → 200 (evol/complexify)   |
| 02d_near_dedup    | 300 → 217 [83]     | 200 → 199 [1]                |
| 02a_length_filter | 217 → 217 [0]      | 199 → 198 [1]                |
| FINAL             | 217                | 198                          |
```

| metric | 00-baseline | 02-complexify | delta |
|---|---:|---:|---:|
| final_count | 217 | 198 | -9% |
| **avg pairwise cosine distance** | **0.2480** | **0.3631** | **+46%** ↑ |
| **text_chars_median** | **396** | **1297** | **+227%** ↑ |
| text_chars_max | 626 | 2476 | +295% |
| **near-dup drop rate** | **27.7%** | **0.5%** | **-98%** ↓ |
| wallclock | ~5m | ~9m (serial evol loop) | +4m |

## What the operator actually does

A real example from the run (input → complexified output):

```
SEED instruction: "Write about a development standard your team uses."

COMPLEXIFIED instruction:
"Write a team process proposal for switching to 2-week sprints, including a
formal impact analysis that explains how this transition will specifically
mitigate the risk of scope creep and resource idle time while maintaining
current velocity."
```

Four explicit constraints stacked on a single seed: (a) "switching to 2-week
sprints" specificity, (b) "including a formal impact analysis" deliverable,
(c) "specifically mitigate scope creep" + "resource idle time" requirements,
(d) "while maintaining current velocity" tradeoff.

The response is a 2200-character structured RFC with sections (Executive
Summary / Process Framework / Cost-Benefit / Timeline / Risk Mitigation), not
the 200-char one-paragraph reply that the prompt-based generator was producing.

## Honest read

**What worked spectacularly**
- Diversity jumped 46% on the headline embedding metric.
- Near-dup rate collapsed from 28% to 0.5%. Each complexified output is
  meaningfully unique — exactly what Simula's "local complexity" axis is
  supposed to deliver.
- The integration was a 12-line registry add. The fact that this slotted in
  without touching `EvolInstructRoundStage` is a strong vote of confidence in
  arka's existing operator architecture.

**What changed about the dataset character**
- Outputs are 3× longer at the median. **This is a feature, not a bug, but it
  changes who the dataset is good for.** Slice 0's outputs were short
  conversational drafts that match Jay's seed style. Slice 2's outputs are
  long structured documents (RFCs, proposals, formal analyses).
- For SFT on a personal writing assistant whose seeds are 250-char emails and
  blog snippets, **slice 2's outputs may overshoot the target voice**. The
  right fix is the Simula `c` parameter (only complexify a fraction of
  records). I haven't wired that knob yet — a single-operator
  `evol_instruct` round complexifies 100% of branches.

**What is NOT in this slice**
- The Simula `c` (complexification fraction) parameter. Today, "use complexify
  on a fraction" requires a config like `operators: [complexify, complexify,
  passthrough_or_other]` and is a brittle proxy. A first-class
  `complexification_fraction: 0.5` field on `GeneratorConfig` is the right
  call. Tracked as followup.
- A `passthrough` operator that returns the parent unchanged. Combined with
  `complexification_fraction` it would let us run "complexify 30% of the
  time, leave the rest alone" cleanly.

**Pre-existing arka issue surfaced (not slice 2's problem)**
- `EvolInstructRoundStage.run()` is a serial `for parent: for branch: ...`
  loop. With 150 branches × 2 LLM calls = 300 calls, and a per-call latency
  of ~1.5s, this stage ran for ~7 minutes single-threaded even with
  `executor.max_workers: 6` configured. The other LLM-heavy stages
  (`prompt_based` generator, `double_critic`) parallelize via
  `ThreadPoolExecutor`. The evol stage doesn't. Worth fixing before slices 3
  & 4 amplify the load.

## Followups

1. **`complexification_fraction: float` on `GeneratorConfig`** — the Simula
   `c` knob. Implement as a per-branch random gate around `complexify`.
2. **Parallelize `EvolInstructRoundStage`** — wrap the parent×branch loop in a
   `ThreadPoolExecutor`, mirror the worker-bound logic from `double_critic`.
3. **A `passthrough` operator** so users can compose "complexify some, leave
   some alone" via the `operators=[...]` list.
4. **Consider dampening the prompt** for short-form domains. Right now
   complexify always pushes toward more constraints / longer responses. A
   future variant could complexify within a length budget.
