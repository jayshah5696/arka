# Polymorphic Config Design — Holistic Analysis

**Date:** 2026-04-16  
**Status:** Design analysis — no code changes yet

---

## The Problem

Every filter and dedup stage carries an `enabled: bool = False` flag. This creates several DX and architectural issues:

1. **Noise in YAML** — Users must explicitly declare `enabled: false` for stages they don't want, or they silently exist as defaults.
2. **Double gating** — `StageBuilder` checks `enabled` to decide whether to instantiate the stage, *and then the stage itself* re-checks `enabled` inside `.run()`. Two redundant guards for the same intent.
3. **No ordering control** — Filter execution order is hardcoded in `StageBuilder._filter_stages()`. Users can't reorder filters without touching Python code.
4. **Flat namespace bloat** — `FiltersConfig` has 10 typed fields, each with `enabled: bool`. Adding a new filter means touching models, stage_builder, and the stage itself.

## Design Options Considered

### Option A: Discriminated Union List (`type` field)

```yaml
filters:
  target_count: 10
  stages:
    - type: length
      min_instruction_chars: 40
      max_response_chars: 16384
    - type: language
      allowed: [en]
    - type: canary
      phrases: ["SECRET"]
    - type: labeling_engine
      rubric_path: ./rubrics/sft_quality.yaml
      min_overall_score: 3.5

dedup:
  stages:
    - type: exact
    - type: near
      lsh_bands: 16
```

**Pros:**
- Presence = enabled. Absence = disabled. Zero ambiguity.
- Order in list = execution order. Users control pipeline flow.
- Adding a new filter = one new Pydantic model + register it. No `StageBuilder` if/else chain.
- Pydantic v2 discriminated unions validate this natively with `Discriminator('type')`.
- Mirrors how DataTrove, Dagster, Prefect, and most modern pipeline tools work.

**Cons:**
- Breaking change to every example YAML file.
- Need a registry/discriminator mapping type → config model.

### Option B: Dict-keyed approach (key = type name)

```yaml
filters:
  target_count: 10
  stages:
    length:
      min_instruction_chars: 40
    language:
      allowed: [en]
    canary:
      phrases: ["SECRET"]
```

**Pros:**
- Slightly less verbose than list (no `type:` key).

**Cons:**
- YAML dicts are unordered by spec (though PyYAML preserves insertion order). Relying on dict ordering for execution order is fragile and surprising.
- Can't have two instances of the same filter type (e.g., two `length` filters with different thresholds for different passes).
- Pydantic doesn't have native discriminated-union support for this shape — needs custom parsing.
- Harder to extend: adding a new filter means adding a new optional field to the parent model (same problem we have today, just without `enabled`).

### Option C: Hybrid (keep named fields, remove `enabled`)

```yaml
filters:
  target_count: 10
  length:
    min_instruction_chars: 40
  language:
    allowed: [en]
```

Where presence of the key = enabled, absence = disabled. Fields become `Optional[XConfig] = None` instead of having `enabled: bool`.

**Pros:**
- Smallest diff from current code.
- No `enabled` flags.

**Cons:**
- Still no ordering control.
- Still flat namespace — adding filters still means editing `FiltersConfig`.
- Still need `StageBuilder` if/else chain (checking `is not None` instead of `.enabled`).
- Can't run the same filter type twice with different params.

---

## Recommendation: Option A (Discriminated Union List)

It's the only option that solves *all four* problems: noise, double-gating, ordering, and extensibility.

### Proposed YAML Shape

```yaml
# ── Minimal example (what 01-minimal.yaml becomes) ──
version: "1"

llm:
  provider: openai
  model: google/gemini-3.1-flash-lite-preview
  api_key: ${OPENROUTER_API_KEY}
  base_url: https://openrouter.ai/api/v1

executor:
  mode: threadpool
  max_workers: 2

data_source:
  type: seeds
  path: ./seeds/07-humanizer-rewrite.jsonl

generator:
  type: prompt_based
  target_count: 10
  prompt_template: |
    ...

# No dedup section at all — means no dedup stages run.
# No filters.stages — means no filter stages run.

filters:
  target_count: 10

output:
  format: jsonl
  path: ./output/01-minimal-dataset.jsonl
```

```yaml
# ── Full-featured example (what 06 becomes) ──
dedup:
  - type: exact
  - type: near
    lsh_bands: 16

filters:
  target_count: 10
  stages:
    - type: length
      min_instruction_chars: 40
      max_instruction_chars: 4096
      min_response_chars: 40
      max_response_chars: 16384
    - type: language
      allowed: [en]
    - type: labeling_engine
      rubric_path: ../rubrics/sft_quality.yaml
      min_overall_score: 3.5
```

```yaml
# ── Privacy example (what 08 becomes) ──
filters:
  target_count: 5
  stages:
    - type: semantic_similarity
      threshold: 0.95
    - type: canary
      phrases: ["SUPER_SECRET_PROJECT_X"]
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Dedup: list or nested?** | Top-level list `dedup: [...]` | Dedup stages are just as composable as filters. Same principle: presence = enabled, list order = execution order. |
| **`filters.stages` vs flat `filters: [...]`** | `filters.stages` (list inside object) | `FiltersConfig` still needs `target_count`. So `filters` stays an object with a `stages` list inside it. |
| **Generator** | Keep as single typed config | Generator is fundamentally a single-choice (prompt_based OR evol_instruct OR transform). It's already polymorphic via `type`. No list needed. |
| **Always-on stages** (NormalizeConversation, SeedSource) | Stay hardcoded in StageBuilder | These aren't user-configurable. They're structural. Putting them in config would be false flexibility. |
| **Discriminator field** | `type: Literal[...]` on each config model | Native Pydantic v2 pattern: `Annotated[Union[...], Discriminator('type')]`. Zero custom parsing. |
| **`enabled` field** | Remove entirely | Presence in list = enabled. The `enabled` field becomes meaningless and must be removed. |
| **Double-gating in stages** | Remove `.enabled` checks from `.run()` | `StageBuilder` won't instantiate disabled stages, so stages never need to self-check. Simplifies every stage. |

### Pydantic Model Shape (sketch)

```python
from typing import Annotated, Literal, Union
from pydantic import Discriminator, Tag

class LengthFilterConfig(StrictModel):
    type: Literal["length"] = "length"
    min_instruction_chars: int = 10
    # ... no enabled field

class CanaryFilterConfig(StrictModel):
    type: Literal["canary"] = "canary"
    phrases: list[str] = Field(default_factory=list)
    # ... no enabled field

# Union type
FilterStageConfig = Annotated[
    Union[
        Annotated[LengthFilterConfig, Tag("length")],
        Annotated[LanguageFilterConfig, Tag("language")],
        Annotated[CanaryFilterConfig, Tag("canary")],
        # ...
    ],
    Discriminator("type"),
]

class FiltersConfig(StrictModel):
    target_count: int
    stages: list[FilterStageConfig] = Field(default_factory=list)

# Same pattern for dedup
DedupStageConfig = Annotated[
    Union[
        Annotated[ExactDedupConfig, Tag("exact")],
        Annotated[NearDedupConfig, Tag("near")],
    ],
    Discriminator("type"),
]

class ResolvedConfig(StrictModel):
    # ...
    dedup: list[DedupStageConfig] = Field(default_factory=list)
    filters: FiltersConfig
```

### StageBuilder After

```python
# Registry maps config type → stage class
FILTER_REGISTRY: dict[str, type[Stage]] = {
    "length": LengthFilterStage,
    "language": LanguageFilterStage,
    "canary": CanaryFilterStage,
    "semantic_similarity": SemanticSimilarityFilterStage,
    "ifd": IFDFilterStage,
    "labeling_engine": LabelingQualityFilterStage,
    # ...
}

def _filter_stages(self) -> list[Stage]:
    return [
        FILTER_REGISTRY[cfg.type](...)  # instantiate from config
        for cfg in self.config.filters.stages
    ]
```

No more if/else chain. Adding a new filter = add model + add one registry entry.

### Migration & Backward Compatibility

Two approaches:

1. **Clean break** — Update all 8 example YAMLs. Bump `version: "2"`. Old configs fail with a clear error pointing to migration docs.
2. **Shim layer** — `ConfigLoader` detects the old `enabled` pattern and auto-converts to list form with a deprecation warning. Remove in version 3.

Given this is a pre-1.0 project with 8 examples, **clean break** is the right call. The shim adds complexity for zero real users.

### Blast Radius

| File | Change |
|------|--------|
| `config/models.py` | Remove `enabled` from all filter/dedup configs. Add `type: Literal[...]`. Create union types. Reshape `FiltersConfig` and `DedupConfig`. |
| `config/loader.py` | No change (Pydantic handles discriminated unions natively). |
| `pipeline/stage_builder.py` | Replace if/else chains with registry lookup over `config.filters.stages` and `config.dedup`. |
| `pipeline/filter_stages.py` | Remove `if not config.enabled` guards from every `.run()` method. |
| `pipeline/cheap_filters.py` | Remove `if not cfg.enabled` guards. |
| `pipeline/dedup_stages.py` | Remove `if not config.enabled` guards. |
| `pipeline/ifd_stage.py` | Remove `if not config.enabled` guard. |
| `pipeline/scoring_stages.py` | Remove `if not config.enabled` guards. |
| `pipeline/runner.py` | Update one reference to `filters.labeling_engine`. |
| `examples/*.yaml` | Rewrite dedup/filters sections. Remove all `enabled:` keys. |
| `docs/configuration.md` | Update to document list-based config. |
| `tests/unit/test_config_loader.py` | Update YAML fixtures. |
| `tests/unit/test_privacy_filters.py` | Update config construction (remove `enabled=True`). |
| `tests/unit/test_dedup_stages.py` | Update config construction (remove `enabled=True`). |
| Other test files | Update any that construct filter/dedup configs with `enabled`. |

### Execution Plan (TDD)

1. Write failing tests for new polymorphic config parsing (list-based filters, list-based dedup).
2. Update `config/models.py` — new models, union types, reshaped `FiltersConfig`/`DedupConfig`.
3. Green the config tests.
4. Write failing tests for registry-based `StageBuilder`.
5. Update `StageBuilder` — registry pattern.
6. Green the builder tests.
7. Remove `enabled` checks from all stage `.run()` methods.
8. Update remaining tests that construct old-style configs.
9. Update all 8 example YAML files.
10. Update `docs/configuration.md`.
11. Full test suite green, 238+ tests passing.
