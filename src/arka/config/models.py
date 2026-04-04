from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, HttpUrl


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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


class FiltersConfig(StrictModel):
    target_count: int


class OutputConfig(StrictModel):
    format: str
    path: str


class ResolvedConfig(StrictModel):
    version: str
    run_id: str | None = None
    llm: LLMConfig
    executor: ExecutorConfig
    data_source: DataSourceConfig
    generator: GeneratorConfig
    filters: FiltersConfig
    output: OutputConfig
