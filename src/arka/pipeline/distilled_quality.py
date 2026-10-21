"""Explicit training and inference for a frozen-embedding quality classifier."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from arka.common.models import StrictModel
from arka.config.models import DistilledQualityTrainConfig
from arka.decisions.models import ScoreAnswer
from arka.embeddings.embedder import Embedder
from arka.labeling.rubric import Rubric, RubricLoader
from arka.pipeline.artifacts import StageReport
from arka.pipeline.models import StageContext, StageResult
from arka.pipeline.output import OutputWriter
from arka.pipeline.quality_artifacts import digest, save_json, source_group
from arka.pipeline.stages import Stage
from arka.records.identity import content_hash, file_hash
from arka.records.models import ConversationRecord, Record

Finite = Annotated[float, Field(allow_inf_nan=False)]


def load_rubric(path: Path, minimums: dict[str, float]) -> Rubric:
    rubric = RubricLoader().load(path)
    dimensions = {d.name: d for d in rubric.dimensions}
    if (
        not dimensions
        or len(dimensions) != len(rubric.dimensions)
        or set(minimums) - dimensions.keys()
        or any(d.scale_min >= d.scale_max for d in dimensions.values())
        or any(not math.isfinite(w) or w < 0 for w in rubric.overall_weights.values())
        or sum(rubric.overall_weights.values()) <= 0
        or any(
            not dimensions[n].scale_min <= v <= dimensions[n].scale_max
            for n, v in minimums.items()
        )
    ):
        raise ValueError(
            "Quality thresholds must match valid rubric dimensions and scales"
        )
    return rubric


def embedding_identity(ctx: StageContext) -> str:
    cfg = ctx.config.embeddings
    return digest(
        {
            "provider": cfg.provider,
            "model": cfg.model,
            "base_url": str(cfg.base_url or ctx.config.llm.base_url)
            if cfg.provider == "openai"
            else None,
        }
    )


def representation(record: ConversationRecord) -> str:
    # A versioned structure prevents ambiguous concatenation of instruction/answer.
    return json.dumps(
        [record.payload.system, record.payload.instruction, record.payload.response],
        ensure_ascii=False,
    )


def vectors(records: list[ConversationRecord], ctx: StageContext) -> np.ndarray:
    value = Embedder(ctx.config).embed(
        [representation(r) for r in records], checkpoint_manager=ctx.checkpoint_manager
    )
    if value is None:
        raise ValueError("Quality embeddings are missing; no predictions were produced")
    value = np.asarray(value, dtype=float)
    if (
        value.ndim != 2
        or value.shape[0] != len(records)
        or value.shape[1] == 0
        or not np.isfinite(value).all()
    ):
        raise ValueError(
            "Quality embeddings must be finite vectors with matching record count"
        )
    return value


def labels(
    records: list[ConversationRecord], rubric: Rubric, minimums: dict[str, float]
) -> np.ndarray:
    values = []
    for record in records:
        scores = record.scores
        if record.payload.turns:
            raise ValueError(
                "Distilled quality supports single-turn conversations only"
            )
        if record.content_hash != content_hash(record.payload):
            raise ValueError(
                "Training/evaluation text changed after its content hash was recorded"
            )
        if scores.rubric_hash != rubric.hash or not scores.judge_model:
            raise ValueError(
                "Training/evaluation requires matching rubric and teacher provenance for every record"
            )
        for dimension in rubric.dimensions:
            value = scores.quality_per_dim.get(dimension.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or not dimension.scale_min <= value <= dimension.scale_max
            ):
                raise ValueError(
                    "Training/evaluation requires valid teacher scores for every rubric dimension"
                )
            if not scores.judge_prompt_hash:
                answer = scores.decisions.get(f"labeling_engine.{dimension.name}")
                if (
                    not isinstance(answer, ScoreAnswer)
                    or answer.confidence < 0.5
                    or answer.score + dimension.scale_min != value
                ):
                    raise ValueError(
                        "Decision labels require matching answers with confidence at least 0.5"
                    )
        values.append(
            int(all(scores.quality_per_dim[n] >= v for n, v in minimums.items()))
        )
    return np.array(values, dtype=int)


def evaluation(actual: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    selected = probabilities >= threshold
    good = actual == 1
    true_positive = int((selected & good).sum())
    false_rejections = int((~selected & good).sum())
    return {
        "records": len(actual),
        "threshold": threshold,
        "true_positives": true_positive,
        "false_acceptances": int((selected & ~good).sum()),
        "true_negatives": int((~selected & ~good).sum()),
        "false_rejections": false_rejections,
        "useful_record_recall": true_positive / int(good.sum()) if good.any() else None,
        "retained_precision": true_positive / int(selected.sum())
        if selected.any()
        else None,
        "rejection_rate": float((~selected).mean()) if len(actual) else None,
        "teacher_agreement": float((selected == good).mean()) if len(actual) else None,
        "independent_factual_quality": "not_measured",
    }


def split_groups(
    records: list[ConversationRecord], fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Join shared sources, roots and exact input texts before splitting groups."""
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen: dict[str, int] = {}
    for i, record in enumerate(records):
        keys = [
            f"root:{record.lineage.root_id}",
            f"text:{digest(representation(record))}",
        ]
        group = source_group(record)
        if group != "ungrouped":
            keys.append(group)
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    groups: dict[int, list[int]] = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)
    if len(groups) < 2:
        raise ValueError(
            "Training needs at least two independent source/lineage groups"
        )
    ordered = sorted(
        groups.values(),
        key=lambda indices: digest(
            [seed, sorted(digest(representation(records[i])) for i in indices)]
        ),
    )
    count = max(1, min(len(ordered) - 1, math.ceil(len(ordered) * fraction)))
    holdout = sorted(i for group in ordered[:count] for i in group)
    train = sorted(i for group in ordered[count:] for i in group)
    return train, holdout


class DistilledModel(StrictModel):
    version: Literal[1] = 1
    algorithm: Literal["logistic_regression"] = "logistic_regression"
    representation: Literal["system_instruction_response_v1"] = (
        "system_instruction_response_v1"
    )
    rubric_hash: str
    min_scores: dict[str, Finite]
    embedding_identity: str
    privacy_fingerprint: str | None
    dimension: int = Field(gt=0)
    coefficients: list[Finite] = Field(min_length=1)
    intercept: Finite
    train_groups: list[str]
    holdout_groups: list[str]
    train_roots: list[str]
    holdout_roots: list[str]
    record_hashes: list[str]
    teacher_models: list[str]
    label_identity: str
    evaluation: dict

    @model_validator(mode="after")
    def validate_dimensions(self):
        if len(self.coefficients) != self.dimension:
            raise ValueError("Model coefficient dimension mismatch")
        if set(self.train_groups) & set(self.holdout_groups) or set(
            self.train_roots
        ) & set(self.holdout_roots):
            raise ValueError("Model training and holdout groups overlap")
        return self

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.shape[1] != self.dimension:
            raise ValueError(
                "Embedding dimension is incompatible with the quality model"
            )
        logits = matrix @ np.array(self.coefficients) + self.intercept
        return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))


class QualityIdentity(StrictModel):
    version: Literal[1] = 1
    settings: str
    inputs: str | None = None
    output_hash: str | None = None


class DistilledStage(Stage):
    """Validate file dependencies even when the runner skips a completed stage."""

    def __init__(self, config, project_root: Path):
        self.config = config
        self.project_root = project_root

    def _external(self) -> list[Path]:
        paths = (
            self.config.training_paths
            if isinstance(self.config, DistilledQualityTrainConfig)
            else [self.config.model_path]
        )
        return [
            self.project_root / self.config.rubric_path,
            *[self.project_root / p for p in paths],
        ]

    def _settings(self, ctx: StageContext) -> str:
        return digest(
            {
                "config": ctx.config.model_dump(mode="json"),
                "stage": self.config.model_dump(mode="json"),
                "files": [(str(p), file_hash(p)) for p in self._external()],
                "privacy": ctx.privacy.fingerprint if ctx.privacy else None,
            }
        )

    def validate_resume(
        self, ctx: StageContext, records: list[Record] | None = None
    ) -> None:
        path = ctx.work_dir / "quality_identity.json"
        if not path.exists():
            if (ctx.work_dir / "model.json").exists():
                raise ValueError("Quality model identity is missing; use a new run ID")
            return
        identity = QualityIdentity.model_validate_json(path.read_text())
        if not ctx.resume:
            raise ValueError("Quality artifacts exist; resume or use a new run ID")
        if identity.settings != self._settings(ctx):
            raise ValueError(
                "Quality inputs or configuration changed; use a new run ID"
            )
        if identity.output_hash and (
            not (ctx.work_dir / "model.json").exists()
            or identity.output_hash != file_hash(ctx.work_dir / "model.json")
        ):
            raise ValueError("Quality model artifact changed; use a new run ID")
        if records is not None and identity.inputs != digest(
            [
                r.model_dump(mode="json", exclude={"created_at", "stage_events"})
                for r in records
            ]
        ):
            raise ValueError("Quality records changed; use a new run ID")

    def _start(self, ctx: StageContext, records: list[Record]) -> QualityIdentity:
        self.validate_resume(ctx, records)
        identity = QualityIdentity(
            settings=self._settings(ctx),
            inputs=digest(
                [
                    r.model_dump(mode="json", exclude={"created_at", "stage_events"})
                    for r in records
                ]
            ),
        )
        save_json(ctx.work_dir / "quality_identity.json", identity)
        return identity


class DistilledQualityTrainStage(DistilledStage):
    name = "03_distilled_quality_train"

    def run(self, records: list[Record], ctx: StageContext) -> StageResult:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression

        if records:
            raise ValueError("Run quality training separately using training_paths")
        self.validate_resume(ctx, records)
        rubric = load_rubric(
            self.project_root / self.config.rubric_path, self.config.min_scores
        )
        training = [
            r
            for path in self.config.training_paths
            for r in OutputWriter().read_parquet(self.project_root / path)
        ]
        if not training or any(not isinstance(r, ConversationRecord) for r in training):
            raise ValueError("Quality training requires labeled conversation records")
        if ctx.privacy:
            from arka.privacy.records import protect_record

            for record in training:
                protected, summary = protect_record(record, ctx.privacy)
                if summary.drop_reason or protected.payload != record.payload:
                    raise ValueError(
                        "Training text needs privacy protection; protect and judge it before training"
                    )
        target = labels(training, rubric, self.config.min_scores)
        train, holdout = split_groups(
            training, self.config.holdout_fraction, self.config.split_seed
        )
        if len(set(target[train])) < 2 or len(set(target[holdout])) < 2:
            raise ValueError(
                "Training and holdout each require passing and failing labels; supply more independent groups"
            )
        identity = self._start(ctx, records)
        matrix = vectors(training, ctx)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier = LogisticRegression(random_state=0, max_iter=1000).fit(
                matrix[train], target[train]
            )
        probabilities = classifier.predict_proba(matrix[holdout])[:, 1]
        model = DistilledModel(
            rubric_hash=rubric.hash,
            min_scores=self.config.min_scores,
            embedding_identity=embedding_identity(ctx),
            privacy_fingerprint=ctx.privacy.fingerprint if ctx.privacy else None,
            dimension=matrix.shape[1],
            coefficients=classifier.coef_[0].tolist(),
            intercept=float(classifier.intercept_[0]),
            train_groups=sorted(
                {
                    digest(source_group(training[i]))
                    for i in train
                    if source_group(training[i]) != "ungrouped"
                }
            ),
            holdout_groups=sorted(
                {
                    digest(source_group(training[i]))
                    for i in holdout
                    if source_group(training[i]) != "ungrouped"
                }
            ),
            train_roots=sorted({digest(training[i].lineage.root_id) for i in train}),
            holdout_roots=sorted(
                {digest(training[i].lineage.root_id) for i in holdout}
            ),
            record_hashes=sorted({digest(representation(r)) for r in training}),
            teacher_models=sorted({r.scores.judge_model for r in training}),
            label_identity=digest(
                [
                    r.model_dump(mode="json", exclude={"created_at", "stage_events"})
                    for r in training
                ]
            ),
            evaluation=evaluation(
                target[holdout], probabilities, self.config.evaluation_threshold
            ),
        )
        path = ctx.work_dir / "model.json"
        save_json(path, model)
        save_json(
            ctx.work_dir / "quality_identity.json",
            identity.model_copy(update={"output_hash": file_hash(path)}),
        )
        return StageResult(
            kept=[],
            report=StageReport(
                stage=self.name,
                count_in=0,
                count_out=0,
                training_count=len(train),
                holdout_count=len(holdout),
                evaluation=model.evaluation,
                model_path=str(path),
                model_hash=file_hash(path),
                embedding_texts=len(training),
                judge_requests=0,
                cost_complete=False,
                cost_note="Training labels and embedding costs must be included in a separate total-cost comparison",
            ),
        )


class DistilledQualityScoreStage(DistilledStage):
    name = "03_distilled_quality_score"
    stage_action = "scored"

    def _model(self, ctx: StageContext) -> DistilledModel:
        model = DistilledModel.model_validate_json(
            (self.project_root / self.config.model_path).read_text()
        )
        rubric = load_rubric(
            self.project_root / self.config.rubric_path, self.config.min_scores
        )
        if (
            model.rubric_hash != rubric.hash
            or model.min_scores != self.config.min_scores
            or model.embedding_identity != embedding_identity(ctx)
            or model.privacy_fingerprint
            != (ctx.privacy.fingerprint if ctx.privacy else None)
        ):
            raise ValueError(
                "Quality model metadata is incompatible with rubric, embedding settings or privacy policy"
            )
        return model

    def validate_resume(
        self, ctx: StageContext, records: list[Record] | None = None
    ) -> None:
        self._model(ctx)
        super().validate_resume(ctx, records)

    def run(self, records: list[Record], ctx: StageContext) -> StageResult:
        model = self._model(ctx)
        self._start(ctx, records)
        records = ctx.protect_records(records)
        supported = [
            r
            for r in records
            if isinstance(r, ConversationRecord) and not r.payload.turns
        ]
        actual = None
        if self.config.evaluate:
            rubric = load_rubric(
                self.project_root / self.config.rubric_path, self.config.min_scores
            )
            actual = labels(supported, rubric, self.config.min_scores)
        probabilities = (
            model.predict(vectors(supported, ctx)) if supported else np.array([])
        )
        predictions = {
            id(r): float(p) for r, p in zip(supported, probabilities, strict=True)
        }
        model_hash = file_hash(self.project_root / self.config.model_path)
        kept, dropped = [], []
        for record in records:
            if id(record) not in predictions:
                kept.append(record)
                continue
            probability = predictions[id(record)]
            annotated = record.with_annotation(
                "distilled_quality",
                {
                    "model_hash": model_hash,
                    "rubric_hash": model.rubric_hash,
                    "text_hash": digest(representation(record)),
                    "prediction": True,
                },
            )
            # The annotation helper mirrors legacy metadata into quality_per_dim.
            # Keep teacher dimensions numeric and unchanged for this new scorer.
            scores = annotated.scores.model_copy(
                update={
                    "distilled_quality": probability,
                    "quality_per_dim": record.scores.quality_per_dim,
                }
            )
            updated = record.model_copy(update={"scores": scores})
            if (
                self.config.min_probability is not None
                and probability < self.config.min_probability
            ):
                dropped.append(
                    updated.dropped_by(
                        self.name, "distilled_quality_below_min", f"model={model_hash}"
                    )
                )
            else:
                kept.append(updated)
        metrics = (
            evaluation(
                actual,
                probabilities,
                self.config.min_probability
                if self.config.min_probability is not None
                else 0.5,
            )
            if actual is not None
            else None
        )
        known_texts = set(model.record_hashes)
        known_roots = set(model.train_roots + model.holdout_roots)
        known_groups = set(model.train_groups + model.holdout_groups)
        independent = not any(
            digest(representation(r)) in known_texts
            or digest(r.lineage.root_id) in known_roots
            or (
                source_group(r) != "ungrouped"
                and digest(source_group(r)) in known_groups
            )
            for r in supported
        )
        return StageResult(
            kept=kept,
            dropped=dropped,
            report=StageReport(
                stage=self.name,
                count_in=len(records),
                count_out=len(kept),
                dropped_count=len(dropped),
                drop_reasons={"distilled_quality_below_min": len(dropped)}
                if dropped
                else {},
                scored_count=len(supported),
                unsupported_count=len(records) - len(supported),
                model_hash=model_hash,
                evaluation=metrics,
                evaluation_independent=independent if actual is not None else None,
                judge_requests=0,
                embedding_texts=len(supported),
                cost_complete=False,
            ),
        )
