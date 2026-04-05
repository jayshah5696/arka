from __future__ import annotations

from arka.common.models import StrictModel
from arka.config.models import ResolvedConfig
from arka.labeling.judges import JudgeResponse
from arka.labeling.models import LabelResult
from arka.labeling.rubric import Rubric
from arka.records.models import ConversationPayload, Record


def test_shared_strict_model_base_is_used_across_boundary_models() -> None:
    assert issubclass(ResolvedConfig, StrictModel)
    assert issubclass(Record, StrictModel)
    assert issubclass(ConversationPayload, StrictModel)
    assert issubclass(Rubric, StrictModel)
    assert issubclass(LabelResult, StrictModel)
    assert issubclass(JudgeResponse, StrictModel)
