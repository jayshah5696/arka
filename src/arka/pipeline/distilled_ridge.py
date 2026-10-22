"""Experimental rubric-score ranking, separate from pass-probability classification."""

from __future__ import annotations

from pathlib import Path

from arka.common.models import StrictModel
from arka.config.models import DistilledRidgeConfig, ResolvedConfig
from arka.distillation import (
    DistilledModel,
    embedding_identity,
    features_for_records,
    input_hash,
)
from arka.pipeline.artifacts import StageReport
from arka.pipeline.models import StageContext, StageResult
from arka.pipeline.quality_artifacts import digest, save_json
from arka.pipeline.stages import Stage
from arka.records.identity import file_hash
from arka.records.models import ConversationRecord, Record, RidgePrediction


class RidgeIdentity(StrictModel):
    settings: str
    inputs: str


class DistilledRidgeStage(Stage):
    name = "02s_distilled_ridge"
    stage_action = "scored"

    def __init__(self, config: DistilledRidgeConfig, project_root: Path):
        self.config = config
        self.path = project_root / config.model_path
        if file_hash(self.path) != config.model_sha256:
            raise ValueError("Ridge model SHA-256 does not match model_sha256")
        self.model = DistilledModel.model_validate_json(self.path.read_bytes())

    @classmethod
    def from_config(
        cls,
        config: DistilledRidgeConfig,
        project_root: Path,
        resolved_config: ResolvedConfig,
    ):
        stage = cls(config, project_root)
        if stage.model.embedding != embedding_identity(resolved_config):
            raise ValueError("Ridge model embedding configuration is incompatible")
        return stage

    def _settings(self, ctx: StageContext) -> str:
        if file_hash(self.path) != self.config.model_sha256:
            raise ValueError("Ridge model changed; use a new run ID")
        if self.model.embedding != embedding_identity(ctx.config):
            raise ValueError("Ridge model embedding configuration is incompatible")
        privacy = ctx.privacy.fingerprint if ctx.privacy else None
        if self.model.privacy_fingerprint != privacy:
            raise ValueError("Ridge model privacy policy is incompatible")
        return digest(
            {
                "config": ctx.config.model_dump(mode="json"),
                "model": self.config.model_sha256,
                "privacy": privacy,
            }
        )

    def validate_resume(
        self, ctx: StageContext, records: list[Record] | None = None
    ) -> None:
        settings = self._settings(ctx)
        path = ctx.work_dir / "ridge_identity.json"
        if not path.exists():
            if (ctx.work_dir / "data.parquet").exists():
                raise ValueError("Ridge model identity is missing; use a new run ID")
            return
        identity = RidgeIdentity.model_validate_json(path.read_text())
        if not ctx.resume or identity.settings != settings:
            raise ValueError(
                "Ridge model or configuration changed; resume or use a new run ID"
            )
        if records is not None and identity.inputs != self._inputs(records):
            raise ValueError("Ridge model input records changed; use a new run ID")

    @staticmethod
    def _inputs(records: list[Record]) -> str:
        return digest(
            [
                r.model_dump(mode="json", exclude={"created_at", "stage_events"})
                for r in records
            ]
        )

    def run(self, records: list[Record], ctx: StageContext) -> StageResult:
        self.validate_resume(ctx, records)
        identity = RidgeIdentity(
            settings=self._settings(ctx), inputs=self._inputs(records)
        )
        records = ctx.protect_records(records)
        conversations = [
            r
            for r in records
            if isinstance(r, ConversationRecord) and not r.payload.turns
        ]
        predictions = iter([])
        if conversations:
            features = features_for_records(
                conversations,
                ctx.config,
                feature_format=self.model.text_format,
                checkpoint_manager=ctx.checkpoint_manager,
            )
            predictions = iter(self.model.predict(features))
        output = []
        for record in records:
            if isinstance(record, ConversationRecord) and not record.payload.turns:
                score = float(next(predictions))
                metadata = RidgePrediction(
                    model_sha256=self.config.model_sha256,
                    input_hash=input_hash(record),
                    target=self.model.target,
                    rubric_hash=self.model.rubric_hash,
                )
                annotated = record.with_annotation(
                    "distilled_ridge", metadata.model_dump(mode="json")
                )
                record = annotated.model_copy(
                    update={
                        "scores": annotated.scores.model_copy(
                            update={
                                "distilled_ridge": score,
                                "quality_per_dim": record.scores.quality_per_dim,
                            }
                        )
                    }
                )
            output.append(record)
        save_json(ctx.work_dir / "ridge_identity.json", identity)
        return StageResult(
            kept=output,
            report=StageReport(
                stage=self.name,
                count_in=len(records),
                count_out=len(output),
                scored_count=len(conversations),
                model_sha256=self.config.model_sha256,
                target=self.model.target,
                rubric_hash=self.model.rubric_hash,
                feature_format=self.model.text_format,
                judge_requests=0,
                cost_complete=False,
            ),
        )
