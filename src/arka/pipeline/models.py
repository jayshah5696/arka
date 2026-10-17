from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from arka.config.models import ResolvedConfig, StageLLMOverride

if TYPE_CHECKING:
    from arka.decisions.client import DecisionClient
    from arka.decisions.factory import DecisionClientFactory
    from arka.decisions.models import DecisionOverride
    from arka.llm.client import LLMClient
    from arka.llm.factory import LLMClientFactory
    from arka.pipeline.checkpoint import CheckpointManager


@dataclass(frozen=True)
class StageContext:
    run_id: str
    stage_name: str
    work_dir: Path
    config: ResolvedConfig
    executor_mode: str
    max_workers: int
    executor: ThreadPoolExecutor | None = None
    checkpoint_manager: CheckpointManager | None = None
    # Optional injected factory used by the StageContext.llm_client() seam.
    # Tests can pass a fake; production runs leave it None and the seam
    # falls back to llm.factory.default_factory.
    llm_client_factory: LLMClientFactory | None = field(default=None)
    decision_client_factory: DecisionClientFactory | None = field(default=None)

    def decision_client(
        self, *, override: DecisionOverride | None = None
    ) -> DecisionClient:
        from arka.decisions.factory import build_decision_client

        if self.config.decisions is None:
            raise ValueError("decisions config is required for a decision stage")
        return build_decision_client(
            self.config.decisions,
            override,
            self.decision_client_factory,
            checkpoint=self.checkpoint_manager,
            executor=self.executor,
        )

    def llm_client(self, *, override: StageLLMOverride | None = None) -> LLMClient:
        """Build an LLMClient for the current Run, applying any per-Stage override.

        The single seam Stages should use instead of constructing
        ``LLMClient(config=ctx.config.llm)`` themselves. Resolves the override
        against ``self.config.llm`` and delegates to the injected factory
        (or the default factory). See ``arka.llm.factory`` for the rationale.
        """
        from arka.llm.factory import build_client

        return build_client(
            base_config=self.config.llm,
            override=override,
            factory=self.llm_client_factory,
        )


@dataclass(frozen=True)
class RunResult:
    run_id: str
    final_count: int
    dataset_path: Path | None
    manifest_path: Path
    output_path: Path


class StageErrorInfo(BaseModel):
    """A typed error captured when a Stage raises during a Run.

    Promoted to Pydantic alongside ``StageStat`` per ADR-0001's deferred
    cleanup item: report aggregation has grown enough that boundary-style
    typing buys more than a frozen dataclass does.
    """

    model_config = ConfigDict(frozen=True)

    type: str
    message: str


class StageStat(BaseModel):
    """Per-Stage row aggregated by the runner across one Run.

    Combines the typed StageReport that the Stage wrote to ``stats.json``
    with the runner's view of resume/error/status. Exposed in the Manifest's
    ``stage_stats`` and in the run report's ``stage_yields``.
    """

    model_config = ConfigDict(frozen=True)

    stage: str
    count_in: int
    count_out: int
    status: str
    resumed: bool
    dropped_count: int = 0
    drop_reasons: dict[str, int] = Field(default_factory=dict)
    quality_distribution: dict[str, float] | None = None
    error: StageErrorInfo | None = None
    cost_usd: float | None = None
