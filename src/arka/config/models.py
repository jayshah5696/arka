from __future__ import annotations

from typing import Literal

from pydantic import Field, HttpUrl, model_validator

from arka.common.models import StrictModel


class OpenAICompatibleConfig(StrictModel):
    referer: HttpUrl | None = None
    title: str | None = None


class LLMConfig(StrictModel):
    provider: Literal["openai"]
    model: str
    api_key: str
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


class GeneratorConfig(StrictModel):
    type: str
    target_count: int
    generation_multiplier: int
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
    rounds: int | None = None
    branching_factor: int | None = None
    operators: list[str] = Field(default_factory=list)
    filter: EvolFilterConfig = Field(default_factory=EvolFilterConfig)

    @model_validator(mode="after")
    def validate_generator_options(self) -> GeneratorConfig:
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
    enabled: bool = False


class NearDedupConfig(StrictModel):
    enabled: bool = False
    shingle_size: int = 5
    num_hashes: int = 128
    jaccard_threshold: float = 0.7


class DedupConfig(StrictModel):
    exact: ExactDedupConfig = Field(default_factory=ExactDedupConfig)
    near: NearDedupConfig = Field(default_factory=NearDedupConfig)


class LengthFilterConfig(StrictModel):
    enabled: bool = False
    min_instruction_chars: int = 10
    max_instruction_chars: int = 4096
    min_response_chars: int = 10
    max_response_chars: int = 16384


class LanguageFilterConfig(StrictModel):
    enabled: bool = False
    allowed: list[str] = Field(default_factory=lambda: ["en"])


class LabelingFilterConfig(StrictModel):
    enabled: bool = False
    rubric_path: str | None = None
    min_overall_score: float | None = None


class IFDFilterConfig(StrictModel):
    enabled: bool = False
    min_score: float = 0.2


class FiltersConfig(StrictModel):
    target_count: int
    length: LengthFilterConfig = Field(default_factory=LengthFilterConfig)
    language: LanguageFilterConfig = Field(default_factory=LanguageFilterConfig)
    ifd: IFDFilterConfig = Field(default_factory=IFDFilterConfig)
    labeling_engine: LabelingFilterConfig = Field(default_factory=LabelingFilterConfig)


class OutputConfig(StrictModel):
    format: Literal["jsonl", "chatml", "alpaca"]
    path: str


class EmbeddingsConfig(StrictModel):
    provider: Literal["huggingface", "openai"] = "huggingface"
    model: str = "all-MiniLM-L6-v2"
    api_key: str | None = None
    base_url: HttpUrl | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None
    openai_compatible: OpenAICompatibleConfig | None = None


class LabelingEngineConfig(StrictModel):
    rubric_path: str | None = None
    mode: Literal["single", "multi"] = "single"


class DensityControllerConfig(StrictModel):
    enabled: bool = False
    reference_run_id: str | None = None
    k_neighbors: int = 10
    sparse_threshold_percentile: int = 20
    strategy: Literal["seed_prioritization"] = "seed_prioritization"


class ResolvedConfig(StrictModel):
    version: str
    run_id: str | None = None
    llm: LLMConfig
    executor: ExecutorConfig
    data_source: DataSourceConfig
    generator: GeneratorConfig
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    filters: FiltersConfig
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    labeling_engine: LabelingEngineConfig = Field(default_factory=LabelingEngineConfig)
    density_controller: DensityControllerConfig = Field(
        default_factory=DensityControllerConfig
    )
    output: OutputConfig
