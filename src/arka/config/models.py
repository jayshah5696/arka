from __future__ import annotations

from typing import Literal

from pydantic import Field, HttpUrl

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


class GeneratorConfig(StrictModel):
    type: str
    target_count: int
    generation_multiplier: int


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


class FiltersConfig(StrictModel):
    target_count: int
    length: LengthFilterConfig = Field(default_factory=LengthFilterConfig)
    language: LanguageFilterConfig = Field(default_factory=LanguageFilterConfig)
    labeling_engine: LabelingFilterConfig = Field(default_factory=LabelingFilterConfig)


class OutputConfig(StrictModel):
    format: str
    path: str


class LabelingEngineConfig(StrictModel):
    rubric_path: str | None = None
    mode: Literal["single", "multi"] = "single"


class ResolvedConfig(StrictModel):
    version: str
    run_id: str | None = None
    llm: LLMConfig
    executor: ExecutorConfig
    data_source: DataSourceConfig
    generator: GeneratorConfig
    filters: FiltersConfig
    labeling_engine: LabelingEngineConfig = Field(default_factory=LabelingEngineConfig)
    output: OutputConfig
