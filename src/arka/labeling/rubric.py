from __future__ import annotations

import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RubricDimension(StrictModel):
    name: str
    description: str
    scale_min: int
    scale_max: int
    criteria: dict[int, str]


class RubricExample(StrictModel):
    instruction: str
    response: str
    scores: dict[str, int]
    reasoning: str


class Rubric(StrictModel):
    version: str
    description: str
    dimensions: list[RubricDimension]
    overall_weights: dict[str, float]
    few_shot: list[RubricExample] = Field(default_factory=list)

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


class RubricValidationError(ValueError):
    """Raised when rubric loading or validation fails."""


class RubricLoader:
    def load(self, path: Path) -> Rubric:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            rubric = Rubric.model_validate(data)
        except (ValidationError, yaml.YAMLError) as exc:
            raise RubricValidationError(str(exc)) from exc
        self._validate_weight_dimensions(rubric)
        return rubric

    def _validate_weight_dimensions(self, rubric: Rubric) -> None:
        dimension_names = {dimension.name for dimension in rubric.dimensions}
        weight_names = set(rubric.overall_weights)
        if weight_names != dimension_names:
            raise RubricValidationError(
                "overall_weights must match rubric dimensions exactly"
            )
