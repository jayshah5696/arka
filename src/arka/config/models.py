from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Discriminator, Field, HttpUrl, SecretStr, Tag, model_validator

from arka.common.models import StrictModel


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


class EvolFilterConfig(StrictModel):
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


class LanguageFilterConfig(StrictModel):
    type: Literal["language"] = "language"
    allowed: list[str] = Field(default_factory=lambda: ["en"])


class LabelingFilterConfig(StrictModel):
    type: Literal["labeling_engine"] = "labeling_engine"
    rubric_path: str | None = None
    min_overall_score: float | None = None


class IFDFilterConfig(StrictModel):
    type: Literal["ifd"] = "ifd"
    min_score: float = 0.2


class RewardModelFilterConfig(StrictModel):
    type: Literal["reward_model"] = "reward_model"
    min_score: float | None = None
    llm_override: StageLLMOverride | None = None


class PairDeltaFilterConfig(StrictModel):
    type: Literal["pair_delta"] = "pair_delta"
    score_field: str = "quality"
    min_delta: float = 0.30
    length_ratio_max: float | None = None


class CompositeSelectConfig(StrictModel):
    type: Literal["select"] = "select"
    target_count: int | None = None
    weights: dict[str, float] = Field(default_factory=dict)
    strategy: str = "top_n"


class SemanticSimilarityFilterConfig(StrictModel):
    type: Literal["semantic_similarity"] = "semantic_similarity"
    threshold: float = 0.9


class CanaryFilterConfig(StrictModel):
    type: Literal["canary"] = "canary"
    phrases: list[str] = Field(default_factory=list)


class DoubleCriticFilterConfig(StrictModel):
    """Simula §2.2 double-critic. Two independent yes/no critic calls per record.

    No tunable knobs in slice 1 — the inverse-prompt property is preserved by
    construction. Future fields (alternate prompts, majority-of-N, llm_override)
    land here without breaking the YAML schema.
    """

    type: Literal["double_critic"] = "double_critic"
    llm_override: StageLLMOverride | None = None


class ComplexityEloFilterConfig(StrictModel):
    """Slice 5 — Simula §2.3 batch-Elo complexity scoring.

    Annotates each ConversationRecord with a comparable `complexity_elo` and
    does NOT drop. Filter or select stages downstream can consume the score.
    """

    type: Literal["complexity_elo"] = "complexity_elo"
    batch_size: int = 5
    samples_per_record: int = 4
    k_factor: float = 32.0
    llm_override: StageLLMOverride | None = None


FilterStageConfig = Annotated[
    Annotated[LengthFilterConfig, Tag("length")]
    | Annotated[LanguageFilterConfig, Tag("language")]
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
    format: Literal["jsonl", "chatml", "alpaca"]
    path: str


class EmbeddingsConfig(StrictModel):
    provider: Literal["huggingface", "openai"] = "huggingface"
    model: str = "all-MiniLM-L6-v2"
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


class ResolvedConfig(StrictModel):
    version: str
    run_id: str | None = None
    llm: LLMConfig
    executor: ExecutorConfig
    data_source: DataSourceConfig
    generator: GeneratorConfig
    dedup: list[DedupStageConfig] = Field(default_factory=list)
    filters: FiltersConfig
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    labeling_engine: LabelingEngineConfig = Field(default_factory=LabelingEngineConfig)
    output: OutputConfig
