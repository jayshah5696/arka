from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, HttpUrl, SecretStr, model_validator

from arka.common.models import StrictModel
from arka.llm.models import TokenUsage

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class DecisionConfig(StrictModel):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key: SecretStr = Field(exclude=True)
    base_url: HttpUrl
    timeout_seconds: float = Field(default=10, gt=0)
    max_retries: int = Field(default=4, ge=1)
    max_state_chars: int = Field(default=24000, ge=1)
    input_cost_per_million: float | None = Field(default=None, ge=0)


class DecisionOverride(StrictModel):
    provider: str | None = None
    model: str | None = None
    base_url: HttpUrl | None = None
    api_key: SecretStr | None = Field(default=None, exclude=True)


class Question(StrictModel):
    name: str = Field(min_length=1, max_length=100, pattern=r"^[\w.-]+$")
    instructions: str = Field(min_length=1)
    exclude_from_panel: bool = False


class YesNoQuestion(Question):
    type: Literal["yes_no"] = "yes_no"
    yes: str | None = None
    no: str | None = None


class ChoiceQuestion(Question):
    type: Literal["choice"] = "choice"
    choices: dict[str, str | None] = Field(min_length=1, max_length=255)


class ScoreQuestion(Question):
    type: Literal["score"] = "score"
    levels: list[str] = Field(min_length=2, max_length=10)


DecisionQuestion = Annotated[
    YesNoQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")
]


class YesNoAnswer(StrictModel):
    type: Literal["yes_no"] = "yes_no"
    probability: Probability

    @property
    def confidence(self) -> float:
        return abs(2 * self.probability - 1)


class DistributionAnswer(StrictModel):
    confidence: Probability

    @model_validator(mode="after")
    def validate_distribution(self):
        probabilities = self.probabilities
        if not probabilities or not math.isclose(
            sum(probabilities.values()), 1, abs_tol=1e-4
        ):
            raise ValueError("probabilities must sum to 1")
        return self


class ChoiceAnswer(DistributionAnswer):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, Probability]

    @model_validator(mode="after")
    def validate_choice(self):
        if self.choice not in self.probabilities:
            raise ValueError("choice is missing from probabilities")
        return self


class ScoreAnswer(DistributionAnswer):
    type: Literal["score"] = "score"
    score: float = Field(ge=0, allow_inf_nan=False)
    probabilities: dict[int, Probability]

    @model_validator(mode="after")
    def validate_score(self):
        if set(self.probabilities) != set(range(len(self.probabilities))):
            raise ValueError("score levels must be contiguous indices starting at zero")
        expected = sum(k * p for k, p in self.probabilities.items())
        if not math.isclose(self.score, expected, abs_tol=1e-3):
            raise ValueError("score must match its probability distribution")
        return self


DecisionAnswer = Annotated[
    YesNoAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")
]


class DecisionRequest(StrictModel):
    state: str
    questions: list[DecisionQuestion] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_names(self):
        names = [q.name for q in self.questions]
        if len(names) != len(set(names)):
            raise ValueError("question names must be unique")
        return self


class DecisionResult(StrictModel):
    answers: dict[str, DecisionAnswer]
    model: str
    provider: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: int = 0
    requests_sent: int = 0
    cache_hits: int = 0
