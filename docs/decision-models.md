# Decision models

Decision stages ask typed questions and store probabilities. They never invoke
an LLM judge as a fallback. Generation still uses the configured `llm` client.
Existing stage backends remain available when explicitly selected.

Start with [the example](../examples/09-decision-models.yaml). Validate it with
`uv run arka --config examples/09-decision-models.yaml --validate-config`.

## Client configuration

```yaml
decisions:
  provider: openai
  model: gpt-6-luna
  api_key: ${OPENAI_API_KEY}
  base_url: https://api.openai.com/v1
  timeout_seconds: 10
  max_retries: 4
  max_state_chars: 24000
  input_cost_per_million: null
```

`provider`, `model`, `api_key`, and `base_url` are required. The other values
shown above are defaults. Set `input_cost_per_million` to the price for your
model to report input-token cost. Arka does not infer a price from a model name.
Both top-level and stage-local API keys are excluded from resolved config files.

| Provider | Model example | Base URL | Appended endpoint |
| --- | --- | --- | --- |
| `openai` | `gpt-6-luna` | `https://api.openai.com/v1` | `/decisions` |
| `typesafe` | `jev-latest` | `https://api.typesafe.ai/v1` | `/systemone` |
| `clef` | `clef` or `clef-flash` | `https://api.cloudflare.com/client/v4/accounts/<account>/ai/run` | `/@cf/cloudflare/<model>` |

Adapters translate provider formats into `YesNoAnswer`, `ChoiceAnswer`, and
`ScoreAnswer`. Score answers use zero-based level indices. Both choice and
score answers contain a full probability distribution and a confidence value.
Arka rejects missing answers, mismatched types, invalid distributions, and
provider refusals. It retries connection errors, HTTP 429, HTTP 529, and 5xx
responses. `max_retries` is the maximum number of attempts, including the first.

Use `decision_override` to change `provider`, `model`, `base_url`, or `api_key`
for a stage or an enabled check. New adapters register in
`arka.decisions.backends.BACKENDS` and need no stage changes.

## Answers and uncertainty

Answers are stored under `scores.decisions`, using names such as
`reward_model.helpfulness`. The SQLite answer cache includes the provider,
model, endpoint, truncated input, and question definition. A repeated request
sends only uncached questions. Changed input or settings produce a new key.

Decision judges, seed gating, language, identity, and taxonomy fidelity support
`on_uncertain: keep | drop | escalate`. The default is `keep`. An escalation
keeps the record and adds an `escalated` stage event for review. There is no
automatic resolution or second judge. HTTP or answer failures in these stages
follow the same policy and are counted in `stats.json`. Enabled checks inside
other stages stop the stage on a provider failure.

The common probability band accepts at `p >= accept_at`, rejects at
`p < reject_at`, and treats the remaining values as uncertain. Defaults are
`accept_at: 0.85` and `reject_at: 0.20`. The thresholds must be in [0, 1], with
`reject_at < accept_at`. Choice and score confidence below `min_confidence`
also follows the uncertainty policy. Its default is `0.5`.

## Stages and checks

| Stage | Decision setting | Behavior |
| --- | --- | --- |
| `seed_gate` | Always uses decisions | Checks usable content, density, and optional domain membership before generation. |
| `pdf_source` | `decision_filter.enabled: true` | Drops front or back matter at `P >= reject_at`, default `0.70`. |
| `prompt_based_generator` | `best_of_n.enabled: true` | Ranks candidates per seed by quality, then answer relevance, then plan order. |
| `taxonomy_generator` | `best_of_n` and `fidelity_check` | Ranks candidates per sampled node set and checks that each requested factor is present. |
| `evol_instruct_generator` | `filter.backend: decision` | Checks refusals on the response, increased difficulty, and solvability. |
| `evol_instruct_generator` | `operator_selection: decision` | Chooses distinct operators by probability before generating branches. |
| `language` | `backend: decision` | Classifies every configured field and checks agreement. `none` allows code-only text. |
| `identity_claim` | `backend: decision` or `regex_then_decision` | Checks foreign identity claims, including paraphrases. |
| `reasoning_language_gate` | `low_confidence_backend: decision` | Checks fields that fastText cannot classify confidently. |
| `semantic_similarity` | `decision_band.enabled: true` | Checks copying and specific leaks only within the cosine band. |
| `decontaminate` | `paraphrase_check.enabled: true` | Checks whether near-miss benchmark matches pose the same problem. |
| `labeling_engine` | `backend: decision` | Converts rubric dimensions to score questions, then computes the weighted overall score. |
| `double_critic` | `backend: decision` | Asks correctness and incorrectness in separate requests. |
| `complexity_elo` | `backend: decision` | Stores absolute complexity and a compatibility mapping in `complexity_elo`. |
| `reward_model` | `backend: decision` | Stores a helpfulness score on a 0 to 9 scale. |
| `pair_delta` | `pairwise_check.enabled: true` | Requires pointwise improvement and pairwise preference, with optional A/B swaps. |
| `select` | Decision weight paths or `label_taxonomy.enabled: true` | Ranks by answers and labels missing taxonomy memberships before coverage reservation. |

Exact dedup, MinHash, length, repetition, artifact removal, and canary checks
remain deterministic. Prompt generation runs in parallel and writes raw
responses in plan order for resume.

### Check options

- `seed_gate` accepts optional `min_density` on the 0 to 4 scale, `domains`
  as a map of names to descriptions, and `allowed_domains`. Place it before
  any generator. Child records preserve a seed's domain label.
- `best_of_n` uses `keep_per_seed: 1` by default. `min_score` is optional on
  the 0 to 4 scale. Losing candidates remain in `dropped.parquet`.
- `fidelity_check` uses the common bands and uncertainty policy.
  `relabel: true` removes failed factors instead of dropping the record.
  It preserves the requested labels in `taxonomy_nodes_requested`.
- Evolution uses `refusal_reject_at: 0.70` and `evolution_accept_at: 0.60`.
  `min_instruction_chars` still applies. Decision mode does not use the
  refusal keyword list or edit-distance threshold.
- `language` uses `check_fields: [instruction, response]`. `backend: fasttext`
  is also available and requires `lid_model_path`. The heuristic remains the
  default. Decision confidence determines whether a classification is used.
- `identity_claim` can run regex first and only ask about survivors.
  It uses the common band on the probability of having no foreign claim.
- `reasoning_language_gate` uses `mismatch_at: 0.70`. It treats
  `P(same_language) < 1 - mismatch_at` as a mismatch and applies `on_mismatch`.
- `semantic_similarity` uses `lower: 0.80`, `upper: 0.97`, and
  `reject_at: 0.70`. It keeps below the band and drops above it without a
  decision call. Inside the band, either copying or leakage can trigger a drop.
- `paraphrase_check` uses `lower: 0.15` and `reject_at: 0.70`. It only runs
  below the n-gram threshold. Confirmed matches follow `action: drop | flag`.
- `double_critic` defaults to `accept_at: 0.75` and `reject_at: 0.40`.
  Both critics must pass the accept rule. Either critic can trigger rejection.
- `complexity_elo` accepts 2 to 10 custom `levels`. Its default has 6 levels.
  The compatibility value is `1000 + 100 * (score - 2.5)`. This is a linear
  mapping, not an Elo update.
- `pairwise_check` defaults to `min_preference: 0.70` and `swap_order: true`.
  A/B preference differences above `0.4` cause a position-inconsistency drop.
  The parent must be in the stage input or supplied through `parent_records`.
- `label_taxonomy` requires `taxonomy_path` and defaults to
  `min_confidence: 0.60`. Existing memberships are preserved. Large taxonomies
  are labeled one level at a time, with at most 255 children per question.

## Panels and selection

Place `decision_panel` after deterministic filters and before decision judges.
`include: auto` selects later stages with `backend: decision`. A list such as
`include: [labeling_engine, reward_model]` restricts the panel. Only stages with
the same effective decision configuration share a panel. Changed payloads are
judged again. The inverse critic always has its own request. If a panel request
fails, the later stage retries the decision provider directly.

`select.weights` and `pair_delta.score_field` accept dotted paths into
`quality_per_dim` and `decisions`. Selection normalizes decision score answers
to [0, 1]. Yes/no answers use their probability; choices use their highest
probability. Missing values contribute zero. `confidence_weighting: true`
multiplies a selection term by its answer confidence. `tie_break: confidence`
uses mean decision confidence and then record ID. Legacy scalar weights keep
their existing scale.

## Artifacts and verification

Each decision stage reports requests, input tokens, cache hits, cost when
configured, and latency percentiles. The manifest and run report aggregate
decision usage. `escalations.parquet` stores escalation events. Set
`output.on_unresolved_escalation: drop` to exclude unresolved records and save
them in `unresolved_escalations.parquet`. The default is `keep`.

Run `uv run python scripts/verify_decisions_cli.py` for a local HTTP check of
the CLI, typed answers, panel reuse, output artifacts, and resume. Run
`just check` for lint, formatting, and tests. Provider accuracy, calibration,
and paid live requests require separate measurement on your dataset.
