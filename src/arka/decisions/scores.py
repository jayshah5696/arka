from __future__ import annotations

from arka.decisions.models import ChoiceAnswer, ScoreAnswer, YesNoAnswer
from arka.records.models import Record


def score_value(
    record: Record,
    path: str,
    *,
    normalize: bool = False,
    confidence_weighting: bool = False,
) -> float | None:
    if path.startswith("decisions."):
        value = record.scores.decisions.get(path.removeprefix("decisions."))
    elif path.startswith("quality_per_dim."):
        value = record.scores.quality_per_dim.get(path.removeprefix("quality_per_dim."))
    else:
        value = getattr(record.scores, path, record.scores.quality_per_dim.get(path))
    confidence = 1
    if isinstance(value, ScoreAnswer):
        confidence = value.confidence
        number = (
            value.score / (len(value.probabilities) - 1) if normalize else value.score
        )
    elif isinstance(value, YesNoAnswer):
        confidence, number = value.confidence, value.probability
    elif isinstance(value, ChoiceAnswer):
        confidence, number = value.confidence, max(value.probabilities.values())
    elif isinstance(value, int | float):
        number = float(value)
    else:
        return None
    return number * confidence if confidence_weighting else number
