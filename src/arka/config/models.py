from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import (
    Discriminator,
    Field,
    HttpUrl,
    PrivateAttr,
    SecretStr,
    Tag,
    model_validator,
)

from arka.common.models import StrictModel
from arka.common.text_patterns import DEFAULT_ARTIFACT_GROUPS
from arka.config.legacy import (
    LegacyConfigNamespace,
    get_legacy_data_source,
    get_legacy_dedup,
    get_legacy_filters,
    get_legacy_generator,
    inject_legacy_dict,
)
from arka.config.legacy import (
    migrate_old_config as legacy_migrate_old_config,
)
from arka.decisions.models import DecisionConfig, DecisionOverride
from arka.privacy.config import PIIConfig


class DecisionStageOptions(StrictModel):
    decision_override: DecisionOverride | None = None
    accept_at: float = Field(default=0.85, ge=0, le=1)
    reject_at: float = Field(default=0.20, ge=0, le=1)
    on_uncertain: Literal["keep", "drop", "escalate"] = "keep"
    min_confidence: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def validate_bands(self):
        if self.reject_at >= self.accept_at:
            raise ValueError("reject_at must be smaller than accept_at")
        return self


class DecisionPanelConfig(StrictModel):
    type: Literal["decision_panel"] = "decision_panel"
    include: Literal["auto"] | list[str] = "auto"
    decision_override: DecisionOverride | None = None


class DecisionCheckConfig(StrictModel):
    enabled: bool = False
    decision_override: DecisionOverride | None = None
    reject_at: float = Field(default=0.70, ge=0, le=1)


class BestOfNConfig(DecisionCheckConfig):
    keep_per_seed: int = Field(default=1, ge=1)
    min_score: float | None = Field(default=None, ge=0, le=4)


class FidelityCheckConfig(DecisionStageOptions):
    enabled: bool = False
    relabel: bool = False


class DecisionBandConfig(DecisionCheckConfig):
    lower: float = Field(default=0.80, ge=0, le=1)
    upper: float = Field(default=0.97, ge=0, le=1)

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.lower > self.upper:
            raise ValueError("lower must not exceed upper")
        return self


class ParaphraseCheckConfig(DecisionCheckConfig):
    lower: float = Field(default=0.15, ge=0, le=1)


class PairwiseCheckConfig(DecisionCheckConfig):
    min_preference: float = Field(default=0.70, ge=0, le=1)
    swap_order: bool = True


class TaxonomyLabelConfig(DecisionCheckConfig):
    taxonomy_path: str | None = None
    min_confidence: float = Field(default=0.60, ge=0, le=1)

    @model_validator(mode="after")
    def validate_path(self):
        if self.enabled and not self.taxonomy_path:
            raise ValueError(
                "taxonomy_path is required when taxonomy labeling is enabled"
            )
        return self


class SeedGateConfig(DecisionStageOptions):
    type: Literal["seed_gate"] = "seed_gate"
    min_density: float | None = Field(default=None, ge=0, le=4)
    domains: dict[str, str | None] = Field(default_factory=dict, max_length=254)
    allowed_domains: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_domains(self):
        if set(self.allowed_domains) - set(self.domains):
            raise ValueError("allowed_domains must be present in domains")
        return self


class OpenAICompatibleConfig(StrictModel):
    referer: HttpUrl | None = None
    title: str | None = None


class LLMConfig(StrictModel):
    provider: Literal["openai"]
    model: str
    # SECURITY: Using SecretStr and Field(exclude=True) to prevent plaintext API keys from leaking into serialized configs on disk (e.g., config.resolved.yaml)
    api_key: SecretStr = Field(exclude=True)
    base_url: HttpUrl
    timeout_seconds: float = 30.0
    max_retries: int = 3
    supports_json_schema: bool | None = None
    openai_compatible: OpenAICompatibleConfig | None = None


class ExecutorConfig(StrictModel):
    mode: Literal["threadpool", "realtime", "provider_batch"] = "threadpool"
    max_workers: int = 4

    @model_validator(mode="after")
    def reject_provider_batch(self):
        if self.mode == "provider_batch":
            raise ValueError(
                "executor.mode: provider_batch is not implemented; use threadpool"
            )
        return self


class DataSourceConfig(StrictModel):
    type: str
    path: str | None = None
    chunk_strategy: Literal["fixed"] | None = None
    chunk_size_chars: int | None = None
    chunk_overlap_chars: int | None = None

    @model_validator(mode="after")
    def validate_pdf_options(self) -> DataSourceConfig:
        if self.type != "pdf":
            return self
        if not self.path:
            raise ValueError("data_source.path is required when data_source.type='pdf'")
        if self.chunk_strategy is None:
            self.chunk_strategy = "fixed"
        if self.chunk_size_chars is None:
            self.chunk_size_chars = 3000
        if self.chunk_overlap_chars is None:
            self.chunk_overlap_chars = 300
        if self.chunk_size_chars <= 0:
            raise ValueError("data_source.chunk_size_chars must be > 0")
        if self.chunk_overlap_chars < 0:
            raise ValueError("data_source.chunk_overlap_chars must be >= 0")
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError(
                "data_source.chunk_overlap_chars must be smaller than chunk_size_chars"
            )
        return self


class EvolFilterConfig(DecisionStageOptions):
    backend: Literal["heuristic", "decision"] = "heuristic"
    refusal_reject_at: float = Field(default=0.70, ge=0, le=1)
    evolution_accept_at: float = Field(default=0.60, ge=0, le=1)
    min_edit_distance_chars: int = 20
    min_instruction_chars: int = 20
    refusal_keywords: list[str] = Field(
        default_factory=lambda: ["I cannot", "I'm unable", "As an AI"]
    )


class StageLLMOverride(StrictModel):
    """Override top-level LLM settings for a specific stage."""

    model: str | None = None
    base_url: HttpUrl | None = None
    # SECURITY: Using SecretStr and Field(exclude=True) to prevent plaintext API keys from leaking into serialized configs on disk (e.g., config.resolved.yaml)
    api_key: SecretStr | None = Field(default=None, exclude=True)
    temperature: float | None = None
    max_tokens: int | None = None


class GeneratorConfig(StrictModel):
    type: str
    target_count: int = 1
    generation_multiplier: int = 1
    prompt_template: str = (
        "You generate synthetic instruction-response pairs for supervised fine-tuning.\n"
        "Create one new instruction and one strong response inspired by the seed example.\n"
        "The new pair must be self-contained, specific, and meaningfully different from the seed.\n"
        'Return only JSON with keys "instruction" and "response".\n\n'
        "Seed instruction:\n{seed_instruction}\n\n"
        "Seed response:\n{seed_response}\n"
    )
    temperature: float = 0.7
    max_tokens: int = 512
    input_field: str | None = None
    output_field: str | None = None
    preserve_original: bool = False
    llm_override: StageLLMOverride | None = None
    rounds: int | None = None
    branching_factor: int | None = None
    operators: list[str] = Field(default_factory=list)
    filter: EvolFilterConfig = Field(default_factory=EvolFilterConfig)
    # Slice 3 — Simula taxonomy-driven generator. Path to a YAML TaxonomyBundle.
    # Required when type='taxonomy_prompt'; ignored otherwise. Made optional on
    # the model itself so other generator types stay backwards compatible.
    taxonomy_path: str | None = None

    @model_validator(mode="after")
    def validate_generator_options(self) -> GeneratorConfig:
        if self.type == "transform":
            if self.input_field is None:
                raise ValueError(
                    "generator.input_field is required when generator.type='transform'"
                )
            if self.output_field is None:
                raise ValueError(
                    "generator.output_field is required when generator.type='transform'"
                )
            return self
        if self.type == "taxonomy_prompt":
            if not self.taxonomy_path:
                raise ValueError(
                    "generator.taxonomy_path is required when generator.type='taxonomy_prompt'"
                )
            return self
        if self.type != "evol_instruct":
            return self
        if self.rounds is None or self.rounds < 1:
            raise ValueError("generator.rounds must be >= 1 for evol_instruct")
        if self.branching_factor is None or self.branching_factor < 1:
            raise ValueError(
                "generator.branching_factor must be >= 1 for evol_instruct"
            )
        if not self.operators:
            raise ValueError("generator.operators must be non-empty for evol_instruct")
        from arka.pipeline.evol_instruct import SUPPORTED_EVOL_OPERATORS

        unknown = sorted(set(self.operators) - set(SUPPORTED_EVOL_OPERATORS))
        if unknown:
            raise ValueError(
                f"generator.operators contains unsupported names: {unknown}"
            )
        return self


class ExactDedupConfig(StrictModel):
    type: Literal["exact"] = "exact"


class NearDedupConfig(StrictModel):
    type: Literal["near"] = "near"
    shingle_size: int = 5
    num_hashes: int = 128
    lsh_bands: int = 16
    jaccard_threshold: float = 0.7


DedupStageConfig = Annotated[
    Annotated[ExactDedupConfig, Tag("exact")] | Annotated[NearDedupConfig, Tag("near")],
    Discriminator("type"),
]


class SentenceVarianceFilterConfig(StrictModel):
    type: Literal["sentence_variance"] = "sentence_variance"
    min_cv: float = 0.15


class LengthFilterConfig(StrictModel):
    type: Literal["length"] = "length"
    min_instruction_chars: int = 10
    max_instruction_chars: int = 4096
    min_response_chars: int = 10
    max_response_chars: int = 16384


class LanguageFilterConfig(DecisionStageOptions):
    backend: Literal["heuristic", "fasttext", "decision"] = "heuristic"
    check_fields: list[Literal["instruction", "response", "system"]] = Field(
        default_factory=lambda: ["instruction", "response"], min_length=1
    )
    lid_model_path: str | None = None
    type: Literal["language"] = "language"
    allowed: list[str] = Field(default_factory=lambda: ["en"])


class RepetitionFilterConfig(StrictModel):
    type: Literal["repetition"] = "repetition"
    check_fields: list[Literal["instruction", "response", "system"]] = Field(
        default_factory=lambda: ["response"], min_length=1
    )
    min_text_chars: int = Field(default=200, ge=0)
    top_ngram_thresholds: dict[
        Literal[2, 3, 4], Annotated[float, Field(ge=0, le=1)]
    ] = Field(default_factory=lambda: {2: 0.20, 3: 0.18, 4: 0.16})
    dup_ngram_thresholds: dict[
        Literal[5, 6, 7, 8, 9, 10], Annotated[float, Field(ge=0, le=1)]
    ] = Field(
        default_factory=lambda: {
            5: 0.15,
            6: 0.14,
            7: 0.13,
            8: 0.12,
            9: 0.11,
            10: 0.10,
        }
    )
    max_char_run: int = Field(default=50, ge=1)


class IdentityClaimFilterConfig(DecisionStageOptions):
    backend: Literal["regex", "regex_then_decision", "decision"] = "regex"
    type: Literal["identity_claim"] = "identity_claim"
    check_fields: list[Literal["instruction", "response", "system"]] = Field(
        default_factory=lambda: ["response"], min_length=1
    )
    own_model_names: list[Annotated[str, Field(min_length=1)]] = Field(
        default_factory=list
    )
    patterns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_patterns(self) -> IdentityClaimFilterConfig:
        for pattern in self.patterns:
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(
                    f"Invalid identity-claim regex {pattern!r}: {exc}"
                ) from exc
        return self


class ArtifactFilterConfig(StrictModel):
    type: Literal["artifact_filter"] = "artifact_filter"
    groups: dict[Annotated[str, Field(min_length=1)], list[str]] = Field(
        default_factory=lambda: {
            group: list(patterns) for group, patterns in DEFAULT_ARTIFACT_GROUPS.items()
        }
    )
    group_action: dict[str, Literal["strip", "drop"]] = Field(default_factory=dict)
    max_strips_per_record: int = Field(default=3, ge=0)
    fields: list[Literal["instruction", "response", "system"]] = Field(
        default_factory=lambda: ["instruction", "response"], min_length=1
    )

    @model_validator(mode="after")
    def validate_patterns(self) -> ArtifactFilterConfig:
        unknown = set(self.group_action) - set(self.groups)
        if unknown:
            raise ValueError(
                f"Artifact actions reference unknown groups: {sorted(unknown)}"
            )
        for group, patterns in self.groups.items():
            for pattern in patterns:
                try:
                    compiled = re.compile(pattern, re.IGNORECASE)
                except re.error as exc:
                    raise ValueError(
                        f"Invalid artifact regex in {group}: {exc}"
                    ) from exc
                if compiled.search("") is not None:
                    raise ValueError(f"Artifact regex in {group} must consume text")
        return self

    def action_for(self, group: str) -> Literal["strip", "drop"]:
        return self.group_action.get(
            group, "strip" if group in ("openers", "closers") else "drop"
        )


class LabelingFilterConfig(DecisionStageOptions):
    backend: Literal["llm", "decision"] = "llm"
    type: Literal["labeling_engine"] = "labeling_engine"
    rubric_path: str | None = None
    min_overall_score: float | None = None


class LabelingScoreConfig(DecisionStageOptions):
    backend: Literal["llm", "decision"] = "llm"
    type: Literal["labeling_score"] = "labeling_score"
    rubric_path: str | None = None
    min_overall_score: float | None = None


class QualityRepairConfig(StrictModel):
    type: Literal["quality_repair"] = "quality_repair"
    rubric_path: str = Field(min_length=1)
    min_scores: dict[str, Annotated[float, Field(allow_inf_nan=False)]] = Field(
        min_length=1
    )
    repair_dimensions: list[str] = Field(min_length=1)
    max_repairs: int = Field(ge=0)
    backend: Literal["llm", "decision"] = "llm"
    min_confidence: float = Field(default=0.5, ge=0, le=1)
    decision_override: DecisionOverride | None = None
    judge_llm_override: StageLLMOverride | None = None
    llm_override: StageLLMOverride | None = None
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1)
    evidence_stage: str = Field(default="01_source", pattern=r"^[\w-]+$")


class DistilledQualityTrainConfig(StrictModel):
    type: Literal["distilled_quality_train"] = "distilled_quality_train"
    training_paths: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    rubric_path: str = Field(min_length=1)
    min_scores: dict[str, Annotated[float, Field(allow_inf_nan=False)]] = Field(
        min_length=1
    )
    holdout_fraction: float = Field(default=0.25, gt=0, lt=1)
    evaluation_threshold: float = Field(default=0.5, ge=0, le=1)
    split_seed: int = 0


class DistilledQualityScoreConfig(StrictModel):
    type: Literal["distilled_quality_score"] = "distilled_quality_score"
    model_path: str = Field(min_length=1)
    rubric_path: str = Field(min_length=1)
    min_scores: dict[str, Annotated[float, Field(allow_inf_nan=False)]] = Field(
        min_length=1
    )
    min_probability: float | None = Field(default=None, ge=0, le=1)
    evaluate: bool = False


class SourceAuditConfig(StrictModel):
    type: Literal["source_audit"] = "source_audit"
    rubric_path: str = Field(min_length=1)
    min_scores: dict[str, Annotated[float, Field(allow_inf_nan=False)]] = Field(
        min_length=1
    )
    sample_size: int = Field(ge=1)
    sample_seed: int = 0
    backend: Literal["llm", "decision"] = "llm"
    min_confidence: float = Field(default=0.5, ge=0, le=1)
    decision_override: DecisionOverride | None = None
    judge_llm_override: StageLLMOverride | None = None
    evidence_stage: str = Field(default="01_source", pattern=r"^[\w-]+$")


class IFDFilterConfig(StrictModel):
    type: Literal["ifd"] = "ifd"
    min_score: float = 0.2


class RewardModelFilterConfig(DecisionStageOptions):
    backend: Literal["llm", "decision"] = "llm"
    type: Literal["reward_model"] = "reward_model"
    min_score: float | None = None
    llm_override: StageLLMOverride | None = None


class PairDeltaFilterConfig(StrictModel):
    type: Literal["pair_delta"] = "pair_delta"
    score_field: str = "quality"
    min_delta: float = 0.30
    length_ratio_max: float | None = None
    pairwise_check: PairwiseCheckConfig = Field(default_factory=PairwiseCheckConfig)


class ReasoningLanguageGateConfig(StrictModel):
    low_confidence_backend: Literal["skip", "decision"] = "skip"
    mismatch_at: float = Field(default=0.70, ge=0, le=1)
    decision_override: DecisionOverride | None = None
    type: Literal["reasoning_language_gate"] = "reasoning_language_gate"
    reference_field: Literal["instruction", "response", "system"] = "instruction"
    checked_fields: list[Literal["reasoning", "response", "instruction", "system"]] = (
        Field(default_factory=lambda: ["reasoning", "response"], min_length=1)
    )
    on_mismatch: Literal["strip_trace", "drop_record"] = "strip_trace"
    min_confidence: float = Field(default=0.6, ge=0, le=1)
    lid_model_path: str | None = Field(default=None, min_length=1)


class DecontaminateConfig(StrictModel):
    paraphrase_check: ParaphraseCheckConfig = Field(
        default_factory=ParaphraseCheckConfig
    )
    type: Literal["decontaminate"] = "decontaminate"
    eval_paths: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    eval_text_field: str = Field(default="question", min_length=1)
    fields: list[Literal["instruction", "response", "system"]] = Field(
        default_factory=lambda: ["instruction"], min_length=1
    )
    n: int = Field(default=5, ge=1)
    idf: Literal["per_benchmark", "global"] = "per_benchmark"
    window_factor: int = Field(default=8, ge=1)
    threshold: float | None = Field(default=None, gt=0, le=1)
    calibrate: bool = False
    action: Literal["drop", "flag"] = "drop"

    @model_validator(mode="after")
    def validate_threshold(self) -> DecontaminateConfig:
        if not self.calibrate and self.threshold is None:
            raise ValueError("threshold is required unless calibrate is true")
        return self


class TaxonomyMinimum(StrictModel):
    factor: str = Field(min_length=1)
    path: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    min_count: int = Field(default=1, ge=1)


class CompositeSelectConfig(StrictModel):
    confidence_weighting: bool = False
    tie_break: Literal["id", "confidence"] = "id"
    label_taxonomy: TaxonomyLabelConfig = Field(default_factory=TaxonomyLabelConfig)
    type: Literal["select"] = "select"
    target_count: int | None = None
    weights: dict[str, float] = Field(default_factory=dict)
    strategy: str = "top_n"
    coverage: list[TaxonomyMinimum] = Field(default_factory=list)
    coverage_policy: Literal["fail_fast", "best_effort"] = "fail_fast"
    coverage_search_limit: int = Field(default=10000, ge=1)


class SemanticSimilarityFilterConfig(StrictModel):
    decision_band: DecisionBandConfig = Field(default_factory=DecisionBandConfig)
    type: Literal["semantic_similarity"] = "semantic_similarity"
    threshold: float = 0.9


class CanaryFilterConfig(StrictModel):
    type: Literal["canary"] = "canary"
    phrases: list[str] = Field(default_factory=list)


class DoubleCriticFilterConfig(DecisionStageOptions):
    """Simula §2.2 double-critic. Two independent yes/no critic calls per record.

    Decision mode also keeps the inverse questions in separate requests.
    """

    type: Literal["double_critic"] = "double_critic"
    backend: Literal["llm", "decision"] = "llm"
    accept_at: float = Field(default=0.75, ge=0, le=1)
    reject_at: float = Field(default=0.40, ge=0, le=1)
    llm_override: StageLLMOverride | None = None


class ComplexityEloFilterConfig(DecisionStageOptions):
    """Slice 5 — Simula §2.3 batch-Elo complexity scoring.

    Annotates each ConversationRecord with a comparable `complexity_elo` and
    does NOT drop. Filter or select stages downstream can consume the score.
    """

    type: Literal["complexity_elo"] = "complexity_elo"
    backend: Literal["elo", "decision"] = "elo"
    levels: list[str] = Field(
        default_factory=lambda: [
            "Trivial: one fact or step",
            "Simple: a few steps and common knowledge",
            "Moderate: several steps or domain knowledge",
            "Hard: multiple reasoning steps and domain expertise",
            "Expert: deep expertise or synthesis",
            "Frontier: research-level task",
        ],
        min_length=2,
        max_length=10,
    )
    batch_size: int = 5
    samples_per_record: int = 4
    k_factor: float = 32.0
    llm_override: StageLLMOverride | None = None


FilterStageConfig = Annotated[
    Annotated[LengthFilterConfig, Tag("length")]
    | Annotated[LanguageFilterConfig, Tag("language")]
    | Annotated[RepetitionFilterConfig, Tag("repetition")]
    | Annotated[IdentityClaimFilterConfig, Tag("identity_claim")]
    | Annotated[ArtifactFilterConfig, Tag("artifact_filter")]
    | Annotated[DecontaminateConfig, Tag("decontaminate")]
    | Annotated[ReasoningLanguageGateConfig, Tag("reasoning_language_gate")]
    | Annotated[SentenceVarianceFilterConfig, Tag("sentence_variance")]
    | Annotated[IFDFilterConfig, Tag("ifd")]
    | Annotated[LabelingFilterConfig, Tag("labeling_engine")]
    | Annotated[RewardModelFilterConfig, Tag("reward_model")]
    | Annotated[PairDeltaFilterConfig, Tag("pair_delta")]
    | Annotated[CompositeSelectConfig, Tag("select")]
    | Annotated[SemanticSimilarityFilterConfig, Tag("semantic_similarity")]
    | Annotated[CanaryFilterConfig, Tag("canary")]
    | Annotated[DoubleCriticFilterConfig, Tag("double_critic")]
    | Annotated[ComplexityEloFilterConfig, Tag("complexity_elo")],
    Discriminator("type"),
]


class FiltersConfig(StrictModel):
    target_count: int
    stages: list[FilterStageConfig] = Field(default_factory=list)

    def get_stage_config(self, type_name: str) -> StrictModel | None:
        """Look up a filter stage config by type name. Returns None if not present."""
        for stage in self.stages:
            if stage.type == type_name:
                return stage
        return None


class OutputConfig(StrictModel):
    on_unresolved_escalation: Literal["keep", "drop"] = "keep"
    format: Literal["jsonl", "chatml", "alpaca"]
    path: str


class EmbeddingsConfig(StrictModel):
    provider: Literal["huggingface", "openai"] = "huggingface"
    model: str = "all-MiniLM-L6-v2"
    batch_size: int = 256
    # SECURITY: Using SecretStr and Field(exclude=True) to prevent plaintext API keys from leaking into serialized configs on disk (e.g., config.resolved.yaml)
    api_key: SecretStr | None = Field(default=None, exclude=True)
    base_url: HttpUrl | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None
    openai_compatible: OpenAICompatibleConfig | None = None


class LabelingEngineConfig(StrictModel):
    rubric_path: str | None = None
    mode: Literal["single", "multi"] = "single"


def resolve_llm_override(
    base: LLMConfig, override: StageLLMOverride | None
) -> LLMConfig:
    """Merge a stage-local LLM override onto the top-level LLM config."""
    if override is None:
        return base
    updates: dict[str, object] = {}
    if override.model is not None:
        updates["model"] = override.model
    if override.base_url is not None:
        updates["base_url"] = override.base_url
    if override.api_key is not None:
        updates["api_key"] = override.api_key
    if not updates:
        return base
    return base.model_copy(update=updates)


class SeedSourceConfig(StrictModel):
    type: Literal["seed_source"] = "seed_source"
    path: str


class PDFSourceConfig(StrictModel):
    decision_filter: DecisionCheckConfig = Field(default_factory=DecisionCheckConfig)
    type: Literal["pdf_source"] = "pdf_source"
    path: str
    chunk_strategy: Literal["fixed"] = "fixed"
    chunk_size_chars: int = 3000
    chunk_overlap_chars: int = 300
    boilerplate_strip: bool = False
    min_block_bytes: int = Field(default=500, gt=0)
    min_doc_frequency: int = Field(default=2, ge=2)
    min_remaining_bytes: int = Field(default=100, ge=0)

    @model_validator(mode="after")
    def validate_pdf_options(self) -> PDFSourceConfig:
        if self.chunk_size_chars <= 0:
            raise ValueError("chunk_size_chars must be > 0")
        if self.chunk_overlap_chars < 0:
            raise ValueError("chunk_overlap_chars must be >= 0")
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError(
                "chunk_overlap_chars must be smaller than chunk_size_chars"
            )
        return self


class NormalizeConversationConfig(StrictModel):
    type: Literal["normalize_conversation"] = "normalize_conversation"


class PromptBasedGeneratorConfig(StrictModel):
    best_of_n: BestOfNConfig = Field(default_factory=BestOfNConfig)
    type: Literal["prompt_based_generator"] = "prompt_based_generator"
    target_count: int = 1
    generation_multiplier: int = 1
    prompt_template: str = (
        "You generate synthetic instruction-response pairs for supervised fine-tuning.\n"
        "Create one new instruction and one strong response inspired by the seed example.\n"
        "The new pair must be self-contained, specific, and meaningfully different from the seed.\n"
        'Return only JSON with keys "instruction" and "response".\n\n'
        "Seed instruction:\n{seed_instruction}\n\n"
        "Seed response:\n{seed_response}\n"
    )
    temperature: float = 0.7
    max_tokens: int = 512
    llm_override: StageLLMOverride | None = None


class TransformGeneratorConfig(StrictModel):
    type: Literal["transform_generator"] = "transform_generator"
    target_count: int = 1
    generation_multiplier: int = 1
    input_field: str
    output_field: str
    preserve_original: bool = False
    prompt_template: str = (
        "You generate synthetic instruction-response pairs for supervised fine-tuning.\n"
        "Create one new instruction and one strong response inspired by the seed example.\n"
        "The new pair must be self-contained, specific, and meaningfully different from the seed.\n"
        'Return only JSON with keys "instruction" and "response".\n\n'
        "Seed instruction:\n{seed_instruction}\n\n"
        "Seed response:\n{seed_response}\n"
    )
    temperature: float = 0.7
    max_tokens: int = 512
    llm_override: StageLLMOverride | None = None


class EvolInstructGeneratorConfig(StrictModel):
    operator_selection: Literal["round_robin", "decision"] = "round_robin"
    decision_override: DecisionOverride | None = None
    type: Literal["evol_instruct_generator"] = "evol_instruct_generator"
    target_count: int = 1
    generation_multiplier: int = 1
    rounds: int = Field(ge=1)
    branching_factor: int = Field(ge=1)
    operators: list[str] = Field(min_length=1)
    filter: EvolFilterConfig = Field(default_factory=EvolFilterConfig)
    temperature: float = 0.7
    max_tokens: int = 512
    llm_override: StageLLMOverride | None = None

    @model_validator(mode="after")
    def validate_evol_options(self) -> EvolInstructGeneratorConfig:
        from arka.pipeline.evol_instruct import SUPPORTED_EVOL_OPERATORS

        unknown = sorted(set(self.operators) - set(SUPPORTED_EVOL_OPERATORS))
        if unknown:
            raise ValueError(f"operators contains unsupported names: {unknown}")
        if self.operator_selection == "decision" and self.branching_factor > len(
            set(self.operators)
        ):
            raise ValueError(
                "decision operator selection needs at least branching_factor distinct operators"
            )
        return self


class TaxonomyGeneratorConfig(StrictModel):
    best_of_n: BestOfNConfig = Field(default_factory=BestOfNConfig)
    fidelity_check: FidelityCheckConfig = Field(default_factory=FidelityCheckConfig)
    type: Literal["taxonomy_generator"] = "taxonomy_generator"
    target_count: int = 1
    generation_multiplier: int = 1
    taxonomy_path: str
    temperature: float = 0.7
    max_tokens: int = 512
    llm_override: StageLLMOverride | None = None


PipelineStageConfig = Annotated[
    Annotated[SeedSourceConfig, Tag("seed_source")]
    | Annotated[DecisionPanelConfig, Tag("decision_panel")]
    | Annotated[SeedGateConfig, Tag("seed_gate")]
    | Annotated[PDFSourceConfig, Tag("pdf_source")]
    | Annotated[NormalizeConversationConfig, Tag("normalize_conversation")]
    | Annotated[PromptBasedGeneratorConfig, Tag("prompt_based_generator")]
    | Annotated[TransformGeneratorConfig, Tag("transform_generator")]
    | Annotated[EvolInstructGeneratorConfig, Tag("evol_instruct_generator")]
    | Annotated[TaxonomyGeneratorConfig, Tag("taxonomy_generator")]
    | Annotated[ExactDedupConfig, Tag("exact")]
    | Annotated[NearDedupConfig, Tag("near")]
    | Annotated[LengthFilterConfig, Tag("length")]
    | Annotated[LanguageFilterConfig, Tag("language")]
    | Annotated[RepetitionFilterConfig, Tag("repetition")]
    | Annotated[IdentityClaimFilterConfig, Tag("identity_claim")]
    | Annotated[ArtifactFilterConfig, Tag("artifact_filter")]
    | Annotated[DecontaminateConfig, Tag("decontaminate")]
    | Annotated[ReasoningLanguageGateConfig, Tag("reasoning_language_gate")]
    | Annotated[SentenceVarianceFilterConfig, Tag("sentence_variance")]
    | Annotated[IFDFilterConfig, Tag("ifd")]
    | Annotated[LabelingFilterConfig, Tag("labeling_engine")]
    | Annotated[LabelingScoreConfig, Tag("labeling_score")]
    | Annotated[QualityRepairConfig, Tag("quality_repair")]
    | Annotated[DistilledQualityTrainConfig, Tag("distilled_quality_train")]
    | Annotated[DistilledQualityScoreConfig, Tag("distilled_quality_score")]
    | Annotated[SourceAuditConfig, Tag("source_audit")]
    | Annotated[RewardModelFilterConfig, Tag("reward_model")]
    | Annotated[PairDeltaFilterConfig, Tag("pair_delta")]
    | Annotated[CompositeSelectConfig, Tag("select")]
    | Annotated[SemanticSimilarityFilterConfig, Tag("semantic_similarity")]
    | Annotated[CanaryFilterConfig, Tag("canary")]
    | Annotated[DoubleCriticFilterConfig, Tag("double_critic")]
    | Annotated[ComplexityEloFilterConfig, Tag("complexity_elo")],
    Discriminator("type"),
]

__all__ = ["LegacyConfigNamespace", "ResolvedConfig"]


class ResolvedConfig(StrictModel):
    _computed_hash: str | None = PrivateAttr(default=None)

    version: str
    run_id: str | None = None
    llm: LLMConfig
    decisions: DecisionConfig | None = None
    executor: ExecutorConfig
    pipeline: list[PipelineStageConfig] = Field(default_factory=list)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    labeling_engine: LabelingEngineConfig = Field(default_factory=LabelingEngineConfig)
    output: OutputConfig
    pii: PIIConfig | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_decision_pipeline(self):
        from arka.pipeline.registry import get_stage_spec

        generator_seen = False
        if sum(stage.type == "quality_repair" for stage in self.pipeline) > 1:
            raise ValueError("Only one quality_repair stage is supported per run")
        for kind in (
            "source_audit",
            "distilled_quality_train",
            "distilled_quality_score",
        ):
            if sum(stage.type == kind for stage in self.pipeline) > 1:
                raise ValueError(f"Only one {kind} stage is supported per run")
        if (
            any(stage.type == "distilled_quality_train" for stage in self.pipeline)
            and len(self.pipeline) != 1
        ):
            raise ValueError(
                "Run distilled_quality_train in a separate pipeline using training_paths"
            )
        for stage in self.pipeline:
            if (
                stage.type == "labeling_engine"
                and stage.backend == "decision"
                and not stage.rubric_path
            ):
                raise ValueError(
                    "labeling_engine backend decision requires rubric_path"
                )
            if stage.type.endswith("generator"):
                generator_seen = True
            if stage.type == "seed_gate" and generator_seen:
                raise ValueError("seed_gate must run before a generator stage")

            spec = get_stage_spec(stage.type)
            needs_decisions = (
                spec.check_requires_decisions(stage)
                if spec is not None
                else (
                    getattr(stage, "backend", None)
                    in ("decision", "regex_then_decision")
                    or stage.type in ("seed_gate", "decision_panel")
                    or getattr(stage, "low_confidence_backend", None) == "decision"
                    or getattr(stage, "operator_selection", None) == "decision"
                    or getattr(getattr(stage, "filter", None), "backend", None)
                    == "decision"
                )
            )
            if needs_decisions and self.decisions is None:
                raise ValueError(f"decisions config is required for {stage.type}")
        return self

    def get_stage_config(self, stage_type: str) -> Any:
        for stage in self.pipeline:
            if getattr(stage, "type", None) == stage_type:
                return stage
        return None

    @model_validator(mode="before")
    @classmethod
    def _migrate_old_config_validator(cls, data: Any) -> Any:
        return legacy_migrate_old_config(data, warn=False)

    @classmethod
    def migrate_old_config(cls, data: Any, *, warn: bool = True) -> Any:
        return legacy_migrate_old_config(data, warn=warn)

    @property
    def data_source(self) -> Any:
        return get_legacy_data_source(self)

    @property
    def generator(self) -> Any:
        return get_legacy_generator(self)

    @property
    def dedup(self) -> list[Any]:
        return get_legacy_dedup(self)

    @property
    def filters(self) -> Any:
        return get_legacy_filters(self)

    def dict(self, *args, **kwargs) -> dict[str, Any]:
        data = super().dict(*args, **kwargs)
        return inject_legacy_dict(self, data, mode="python")

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        mode = kwargs.get("mode", "python")
        data = super().model_dump(*args, **kwargs)
        return inject_legacy_dict(self, data, mode=mode)
