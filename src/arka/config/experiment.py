"""Version 2 authoring boundary. Execution uses the existing typed stage configs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, SecretStr, model_validator

from arka.common.models import StrictModel
from arka.config.models import (
    EmbeddingsConfig,
    LabelingEngineConfig,
    LLMConfig,
    OpenAICompatibleConfig,
    OutputConfig,
    PipelineStageConfig,
    ResolvedConfig,
    StageLLMOverride,
)
from arka.decisions.models import DecisionConfig
from arka.privacy.config import PIIConfig


class ExecutionSettings(StrictModel):
    max_workers: int = Field(default=4, ge=1)
    timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    max_retries: int = Field(default=3, ge=1)


class NamedModel(StrictModel):
    provider: Literal["openai"]
    model: str = Field(min_length=1)
    api_key: SecretStr = Field(exclude=True)
    base_url: HttpUrl
    supports_json_schema: bool | None = None
    openai_compatible: OpenAICompatibleConfig | None = None

    def connection(self, execution: ExecutionSettings) -> LLMConfig:
        return LLMConfig(
            **{name: getattr(self, name) for name in type(self).model_fields},
            timeout_seconds=execution.timeout_seconds,
            max_retries=execution.max_retries,
        )


class ExperimentStage(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
    model: str | None = Field(default=None, min_length=1)
    config: PipelineStageConfig


class ExperimentConfig(StrictModel):
    version: Literal["2"]
    models: dict[str, NamedModel] = Field(min_length=1)
    default_model: str
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    pipeline: list[ExperimentStage] = Field(min_length=1)
    output: OutputConfig
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    decisions: DecisionConfig | None = None
    labeling_engine: LabelingEngineConfig = Field(default_factory=LabelingEngineConfig)
    pii: PIIConfig | None = None

    @model_validator(mode="after")
    def validate_bindings(self):
        ids = [stage.id for stage in self.pipeline]
        if len(ids) != len(set(ids)):
            raise ValueError("pipeline stage IDs must be unique")
        references = [self.default_model, *(s.model for s in self.pipeline if s.model)]
        for name in references:
            if name not in self.models:
                raise ValueError(f"Unknown model reference: {name!r}")
        for stage in self.pipeline:
            self._validate_overrides(stage.config, stage.id)
        return self

    @classmethod
    def _validate_overrides(cls, value: BaseModel, identity: str) -> None:
        if isinstance(value, StageLLMOverride) and any(
            getattr(value, key) is not None for key in ("model", "base_url", "api_key")
        ):
            raise ValueError(
                f"Stage {identity!r}: use a named model for connection overrides"
            )
        for name in type(value).model_fields:
            child = getattr(value, name)
            if isinstance(child, BaseModel):
                cls._validate_overrides(child, identity)

    def resolve(self) -> ResolvedConfig:
        """Bind operational settings without serializing or copying secret values."""
        transport = {
            "timeout_seconds": self.execution.timeout_seconds,
            "max_retries": self.execution.max_retries,
        }
        models = {
            name: model.connection(self.execution)
            for name, model in self.models.items()
        }
        resolved = ResolvedConfig(
            version="1",
            llm=models[self.default_model],
            executor={"mode": "threadpool", "max_workers": self.execution.max_workers},
            pipeline=[stage.config for stage in self.pipeline],
            output=self.output,
            embeddings=self.embeddings.model_copy(update=transport),
            decisions=self.decisions.model_copy(update=transport)
            if self.decisions
            else None,
            labeling_engine=self.labeling_engine,
            pii=self.pii,
        )
        resolved._experiment = self
        resolved._stage_ids = tuple(stage.id for stage in self.pipeline)
        resolved._stage_models = tuple(
            models[stage.model or self.default_model] for stage in self.pipeline
        )
        return resolved
