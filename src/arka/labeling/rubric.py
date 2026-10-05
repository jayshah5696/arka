from __future__ import annotations

import hashlib
import io
from pathlib import Path

import yaml
from pydantic import Field, ValidationError

from arka.common.models import StrictModel


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
    expected_verdict: str | None = None


class Rubric(StrictModel):
    version: str
    description: str
    dimensions: list[RubricDimension]
    overall_weights: dict[str, float]
    few_shot: list[RubricExample] = Field(default_factory=list)

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class RubricValidationError(ValueError):
    """Raised when rubric loading or validation fails."""


class RubricLoader:
    def load(self, path: Path) -> Rubric:
        try:
            # DX: Use a named StringIO to give the yaml parser a filename for better line hints
            raw_text = path.read_text(encoding="utf-8")
            stream = io.StringIO(raw_text)
            stream.name = str(path)
            data = yaml.safe_load(stream) or {}
            rubric = Rubric.model_validate(data)
        except ValidationError as exc:
            raise RubricValidationError(str(exc)) from exc
        except yaml.YAMLError as exc:
            # DX: Provide actionable line and column hints for raw YAML syntax errors
            mark = getattr(exc, "problem_mark", None)
            if mark is not None:
                name = mark.name
                line = mark.line + 1
                column = mark.column + 1
                problem = getattr(exc, "problem", str(exc))
                msg = f"YAML syntax error in {name} at line {line}, column {column}: {problem}"
            else:
                msg = f"YAML syntax error: {exc}"
            raise RubricValidationError(msg) from exc
        self._validate_weight_dimensions(rubric)
        return rubric

    def _validate_weight_dimensions(self, rubric: Rubric) -> None:
        dimension_names = {dimension.name for dimension in rubric.dimensions}
        weight_names = set(rubric.overall_weights)
        if weight_names != dimension_names:
            raise RubricValidationError(
                "overall_weights must match rubric dimensions exactly"
            )
        self._validate_expected_verdicts(rubric)

    def _validate_expected_verdicts(self, rubric: Rubric) -> None:
        if not rubric.few_shot:
            return
        verdicts = [example.expected_verdict for example in rubric.few_shot]
        if any(verdict is None for verdict in verdicts):
            raise RubricValidationError(
                "few_shot examples must declare expected_verdict: 'pass' or 'fail'"
            )
        normalized = {str(verdict) for verdict in verdicts if verdict is not None}
        if not normalized.issubset({"pass", "fail"}):
            raise RubricValidationError(
                "few_shot.expected_verdict must be either 'pass' or 'fail'"
            )
