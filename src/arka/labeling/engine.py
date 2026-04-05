from __future__ import annotations

import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from arka.common.concurrency import bounded_worker_count
from arka.labeling.judges import SingleLLMJudge
from arka.labeling.models import LabelResult
from arka.labeling.rubric import Rubric


class LabelingEngine:
    def __init__(self, llm_client: Any) -> None:
        self._judge = SingleLLMJudge(llm_client=llm_client)

    def label(self, instruction: str, response: str, rubric: Rubric) -> LabelResult:
        return self._judge.label(
            instruction=instruction,
            response=response,
            rubric=rubric,
        )

    def label_batch(
        self,
        pairs: Sequence[tuple[str, str]],
        rubric: Rubric,
        max_workers: int,
    ) -> list[LabelResult]:
        worker_count = bounded_worker_count(len(pairs), max_workers)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(self.label, instruction, response, rubric)
                for instruction, response in pairs
            ]
            pair_results = [future.result() for future in futures]

        self._run_canary_checks(rubric=rubric)
        return pair_results

    def _run_canary_checks(self, rubric: Rubric) -> None:
        if len(rubric.few_shot) < 2:
            return
        known_good = rubric.few_shot[0]
        known_bad = rubric.few_shot[-1]
        good_result = self.label(
            instruction=known_good.instruction,
            response=known_good.response,
            rubric=rubric,
        )
        bad_result = self.label(
            instruction=known_bad.instruction,
            response=known_bad.response,
            rubric=rubric,
        )
        if bad_result.overall >= good_result.overall:
            warnings.warn(
                "known-bad canary scored too high relative to known-good canary",
                stacklevel=2,
            )
