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
        run_canary: bool = True,
    ) -> list[LabelResult]:
        # 1. Identify canary examples if requested
        canary_good_item = None
        canary_bad_item = None
        if run_canary:
            passing_examples = [
                e for e in rubric.few_shot if e.expected_verdict == "pass"
            ]
            failing_examples = [
                e for e in rubric.few_shot if e.expected_verdict == "fail"
            ]
            if passing_examples and failing_examples:
                canary_good_item = passing_examples[0]
                canary_bad_item = failing_examples[0]

        # 2. Prepare items to run concurrently
        items_to_run = list(pairs)
        canary_indices = []
        if canary_good_item is not None:
            canary_indices.append(len(items_to_run))
            items_to_run.append((canary_good_item.instruction, canary_good_item.response))
        if canary_bad_item is not None:
            canary_indices.append(len(items_to_run))
            items_to_run.append((canary_bad_item.instruction, canary_bad_item.response))

        worker_count = bounded_worker_count(len(items_to_run), max_workers)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(self.label, instruction, response, rubric)
                for instruction, response in items_to_run
            ]
            all_results = [future.result() for future in futures]

        # 3. Separate main results and canary results
        pair_results = all_results[:len(pairs)]

        if len(canary_indices) == 2:
            good_result = all_results[canary_indices[0]]
            bad_result = all_results[canary_indices[1]]
            if bad_result.overall >= good_result.overall:
                warnings.warn(
                    "known-bad canary scored too high relative to known-good canary",
                    stacklevel=2,
                )

        return pair_results

