from __future__ import annotations

from typing import Annotated, Any, Literal

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


class SeedSourceConfig(StrictModel):
    type: Literal["seed_source"] = "seed_source"
    path: str


class PDFSourceConfig(StrictModel):
    type: Literal["pdf_source"] = "pdf_source"
    path: str
    chunk_strategy: Literal["fixed"] = "fixed"
    chunk_size_chars: int = 3000
    chunk_overlap_chars: int = 300

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
        return self


class TaxonomyGeneratorConfig(StrictModel):
    type: Literal["taxonomy_generator"] = "taxonomy_generator"
    target_count: int = 1
    generation_multiplier: int = 1
    taxonomy_path: str
    temperature: float = 0.7
    max_tokens: int = 512
    llm_override: StageLLMOverride | None = None


PipelineStageConfig = Annotated[
    Annotated[SeedSourceConfig, Tag("seed_source")]
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


def _recursive_dump(val: Any, mode: str = "python") -> Any:
    if isinstance(val, LegacyConfigNamespace):
        return {
            k: _recursive_dump(v, mode)
            for k, v in val.__dict__.items()
            if not k.startswith("_")
        }
    if hasattr(val, "model_dump"):
        try:
            return val.model_dump(mode=mode)
        except TypeError:
            return val.model_dump()
    elif hasattr(val, "dict"):
        try:
            return val.dict()
        except TypeError:
            pass
    if isinstance(val, list):
        return [_recursive_dump(item, mode) for item in val]
    if isinstance(val, dict):
        return {k: _recursive_dump(v, mode) for k, v in val.items()}
    return val


class LegacyConfigNamespace:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def model_copy(self, update=None):
        import copy

        new_obj = copy.copy(self)
        if update:
            for k, v in update.items():
                setattr(new_obj, k, v)
        return new_obj

    def dict(self, *args, **kwargs):
        raw = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return _recursive_dump(raw, mode="python")

    def model_dump(self, *args, **kwargs):
        mode = kwargs.get("mode", "python")
        raw = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return _recursive_dump(raw, mode=mode)

    def get_stage_config(self, stage_type: str) -> Any:
        stages = getattr(self, "stages", [])
        for s in stages:
            if getattr(s, "type", None) == stage_type:
                return s
        return None

    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.dict().items())
        return f"LegacyConfigNamespace({attrs})"


class ResolvedConfig(StrictModel):
    version: str
    run_id: str | None = None
    llm: LLMConfig
    executor: ExecutorConfig
    pipeline: list[PipelineStageConfig] = Field(default_factory=list)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    labeling_engine: LabelingEngineConfig = Field(default_factory=LabelingEngineConfig)
    output: OutputConfig

    def get_stage_config(self, stage_type: str) -> Any:
        for stage in self.pipeline:
            if getattr(stage, "type", None) == stage_type:
                return stage
        return None

    @model_validator(mode="before")
    @classmethod
    def migrate_old_config(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "pipeline" in data:
            return data

        import warnings
        warnings.warn(
            "Legacy configuration format detected. Please migrate to the unified sequential pipeline config format. "
            "You can use the offline migration script `scripts/migrate_config.py` to automatically update your files.",
            DeprecationWarning,
            stacklevel=2,
        )

        pipeline = []

        # 1. data_source
        data_source = data.get("data_source")
        if data_source:
            ds_type = data_source.get("type")
            if ds_type not in ("seeds", "pdf"):
                raise ValueError(f"Unsupported data_source.type: {ds_type}")
            ds_cfg = {k: v for k, v in data_source.items() if v is not None}
            if ds_type == "seeds":
                ds_cfg["type"] = "seed_source"
                pipeline.append(ds_cfg)
                pipeline.append({"type": "normalize_conversation"})
            elif ds_type == "pdf":
                ds_cfg["type"] = "pdf_source"
                pipeline.append(ds_cfg)

        # 2. generator
        generator = data.get("generator")
        if generator:
            gen_type = generator.get("type")
            if gen_type not in (
                "prompt_based",
                "transform",
                "evol_instruct",
                "taxonomy_prompt",
            ):
                raise ValueError(f"Unsupported generator.type: {gen_type}")
            target_count = (
                generator.get("target_count")
                or (data.get("filters") or {}).get("target_count")
                or 100
            )

            g_cfg = {k: v for k, v in generator.items() if v is not None}

            if gen_type == "prompt_based":
                g_cfg.setdefault("target_count", target_count)
                g_cfg["type"] = "prompt_based_generator"
                pipeline.append(g_cfg)
            elif gen_type == "transform":
                g_cfg.pop("target_count", None)
                g_cfg["type"] = "transform_generator"
                pipeline.append(g_cfg)
            elif gen_type == "evol_instruct":
                g_cfg.setdefault("target_count", target_count)
                g_cfg["type"] = "evol_instruct_generator"
                pipeline.append(g_cfg)
            elif gen_type == "taxonomy_prompt":
                g_cfg.setdefault("target_count", target_count)
                g_cfg["type"] = "taxonomy_generator"
                pipeline.append(g_cfg)

        # 3. dedup
        dedup = data.get("dedup") or []
        for d in dedup:
            pipeline.append(d)

        # 4. filters
        filters = data.get("filters") or {}
        stages = filters.get("stages") or []
        for stage in stages:
            pipeline.append(stage)

        new_data = {
            k: v
            for k, v in data.items()
            if k not in ("data_source", "generator", "dedup", "filters")
        }
        new_data["pipeline"] = pipeline
        return new_data

    @property
    def data_source(self) -> Any:
        if "data_source" in self.__dict__:
            return self.__dict__["data_source"]
        for stage in self.pipeline:
            if stage.type == "seed_source":
                return LegacyConfigNamespace(type="seeds", path=stage.path)
            elif stage.type == "pdf_source":
                return LegacyConfigNamespace(
                    type="pdf",
                    path=stage.path,
                    chunk_strategy=getattr(stage, "chunk_strategy", "fixed"),
                    chunk_size_chars=getattr(stage, "chunk_size_chars", 3000),
                    chunk_overlap_chars=getattr(stage, "chunk_overlap_chars", 300),
                    max_chunks=getattr(stage, "max_chunks", None),
                )
        return None

    @property
    def generator(self) -> Any:
        if "generator" in self.__dict__:
            return self.__dict__["generator"]
        for stage in self.pipeline:
            if stage.type == "prompt_based_generator":
                return LegacyConfigNamespace(
                    type="prompt_based",
                    target_count=stage.target_count,
                    generation_multiplier=stage.generation_multiplier,
                    prompt_template=getattr(stage, "prompt_template", None),
                    temperature=getattr(stage, "temperature", 0.7),
                    max_tokens=getattr(stage, "max_tokens", 512),
                    llm_override=getattr(stage, "llm_override", None),
                )
            elif stage.type == "transform_generator":
                return LegacyConfigNamespace(
                    type="transform",
                    target_count=stage.target_count,
                    generation_multiplier=stage.generation_multiplier,
                    prompt_template=getattr(stage, "prompt_template", None),
                    system_prompt=getattr(stage, "system_prompt", None),
                    temperature=getattr(stage, "temperature", 0.7),
                    max_tokens=getattr(stage, "max_tokens", 512),
                    input_field=getattr(stage, "input_field", "payload.instruction"),
                    output_field=getattr(stage, "output_field", "payload.response"),
                    preserve_original=getattr(stage, "preserve_original", False),
                    llm_override=getattr(stage, "llm_override", None),
                )
            elif stage.type == "evol_instruct_generator":
                return LegacyConfigNamespace(
                    type="evol_instruct",
                    rounds=stage.rounds,
                    branching_factor=getattr(stage, "branching_factor", 1),
                    operators=getattr(stage, "operators", []),
                    filter=getattr(stage, "filter", None),
                    temperature=getattr(stage, "temperature", 0.7),
                    max_tokens=getattr(stage, "max_tokens", 512),
                    target_count=stage.target_count,
                    prompt_template=getattr(stage, "prompt_template", None),
                    llm_override=getattr(stage, "llm_override", None),
                )
            elif stage.type == "taxonomy_generator":
                return LegacyConfigNamespace(
                    type="taxonomy_prompt",
                    target_count=stage.target_count,
                    taxonomy_path=getattr(stage, "taxonomy_path", None),
                    temperature=getattr(stage, "temperature", 0.7),
                    max_tokens=getattr(stage, "max_tokens", 512),
                    llm_override=getattr(stage, "llm_override", None),
                )
        return None

    @property
    def dedup(self) -> list[Any]:
        if "dedup" in self.__dict__:
            return self.__dict__["dedup"]
        return [stage for stage in self.pipeline if stage.type in ("exact", "near")]

    @property
    def filters(self) -> Any:
        if "filters" in self.__dict__:
            return self.__dict__["filters"]
        stages = [
            stage
            for stage in self.pipeline
            if stage.type
            not in (
                "seed_source",
                "pdf_source",
                "normalize_conversation",
                "prompt_based_generator",
                "transform_generator",
                "evol_instruct_generator",
                "taxonomy_generator",
                "exact",
                "near",
            )
        ]
        target_count = 100
        for stage in self.pipeline:
            if hasattr(stage, "target_count") and stage.target_count is not None:
                target_count = stage.target_count
                break
        return LegacyConfigNamespace(target_count=target_count, stages=stages)

    def dict(self, *args, **kwargs) -> dict[str, Any]:
        data = super().dict(*args, **kwargs)
        if "data_source" not in data:
            try:
                ds = self.data_source
                if ds is not None:
                    data["data_source"] = _recursive_dump(ds, mode="python")
            except Exception:
                pass
        if "generator" not in data:
            try:
                g = self.generator
                if g is not None:
                    data["generator"] = _recursive_dump(g, mode="python")
            except Exception:
                pass
        if "dedup" not in data:
            try:
                d = self.dedup
                if d is not None:
                    data["dedup"] = [_recursive_dump(x, mode="python") for x in d]
            except Exception:
                pass
        if "filters" not in data:
            try:
                f = self.filters
                if f is not None:
                    data["filters"] = _recursive_dump(f, mode="python")
            except Exception:
                pass
        return data

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        data = super().model_dump(*args, **kwargs)
        mode = kwargs.get("mode", "python")
        if "data_source" not in data:
            try:
                ds = self.data_source
                if ds is not None:
                    data["data_source"] = _recursive_dump(ds, mode=mode)
            except Exception:
                pass
        if "generator" not in data:
            try:
                g = self.generator
                if g is not None:
                    data["generator"] = _recursive_dump(g, mode=mode)
            except Exception:
                pass
        if "dedup" not in data:
            try:
                d = self.dedup
                if d is not None:
                    data["dedup"] = [_recursive_dump(x, mode=mode) for x in d]
            except Exception:
                pass
        if "filters" not in data:
            try:
                f = self.filters
                if f is not None:
                    data["filters"] = _recursive_dump(f, mode=mode)
            except Exception:
                pass
        return data
