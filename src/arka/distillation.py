"""Train and load a small rubric-score regressor without serializing Python objects."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from arka.common.models import StrictModel
from arka.config.models import ResolvedConfig
from arka.decisions.scores import score_value
from arka.embeddings.embedder import Embedder
from arka.records.identity import content_hash
from arka.records.models import ConversationRecord

if TYPE_CHECKING:
    from arka.pipeline.checkpoint import CheckpointManager

Finite = Annotated[float, Field(allow_inf_nan=False)]
FeatureFormat = Literal["instruction_response_json_v1", "pair_quality_v2"]
PAIR_FEATURE_NAMES = [
    "log_instruction_chars",
    "log_response_chars",
    "log_instruction_words",
    "log_response_words",
    "log_length_ratio",
    "response_unique_word_fraction",
    "instruction_word_coverage",
    "response_word_overlap",
    "response_newline_fraction",
    "response_digit_fraction",
    "response_sentence_punctuation_fraction",
    "embedding_cosine",
]


class EmbeddingIdentity(StrictModel):
    provider: str
    model: str
    endpoint: str | None = None


def embedding_identity(config: ResolvedConfig) -> EmbeddingIdentity:
    embeddings = config.embeddings
    return EmbeddingIdentity(
        provider=embeddings.provider,
        model=embeddings.model,
        endpoint=(
            str(embeddings.base_url or config.llm.base_url)
            if embeddings.provider == "openai"
            else None
        ),
    )


def record_text(record: ConversationRecord) -> str:
    """Version 1 features contain the same instruction/answer pair as rubric judging."""
    return json.dumps(
        {
            "instruction": record.payload.instruction,
            "response": record.payload.response,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def input_hash(record: ConversationRecord) -> str:
    return hashlib.sha256(record_text(record).encode()).hexdigest()


def features_for_records(
    records: list[ConversationRecord],
    config: ResolvedConfig,
    *,
    feature_format: FeatureFormat,
    checkpoint_manager: CheckpointManager | None = None,
) -> np.ndarray:
    """Build the exact feature representation declared by the saved model."""
    if feature_format == "instruction_response_json_v1":
        return validate_features(
            Embedder(config).embed(
                [record_text(r) for r in records],
                checkpoint_manager=checkpoint_manager,
            ),
            count=len(records),
        )
    if feature_format != "pair_quality_v2":
        raise ValueError(f"Unsupported distilled feature format: {feature_format}")
    texts = [r.payload.instruction for r in records] + [
        r.payload.response for r in records
    ]
    vectors = validate_features(
        Embedder(config).embed(
            texts,
            checkpoint_manager=checkpoint_manager,
        ),
        count=len(texts),
    )
    count = len(records)
    a, b = vectors[:count], vectors[count:]
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("Invalid zero or non-finite embedding norm")
    cosine = np.sum(a * b, axis=1) / norms
    features = []
    for r, similarity in zip(records, cosine, strict=True):
        instruction, response = r.payload.instruction, r.payload.response
        wa = re.findall(r"\w+", instruction.lower())
        wb = re.findall(r"\w+", response.lower())
        overlap = len(set(wa) & set(wb))
        features.append(
            [
                np.log1p(len(instruction)),
                np.log1p(len(response)),
                np.log1p(len(wa)),
                np.log1p(len(wb)),
                np.log1p(len(response)) - np.log1p(len(instruction)),
                len(set(wb)) / max(1, len(wb)),
                overlap / max(1, len(set(wa))),
                overlap / max(1, len(set(wb))),
                response.count("\n") / max(1, len(response)),
                sum(c.isdigit() for c in response) / max(1, len(response)),
                sum(c in ".!?" for c in response) / max(1, len(response)),
                float(similarity),
            ]
        )
    return validate_features(np.asarray(features), count=count)


class Evaluation(StrictModel):
    train_count: int = Field(ge=2)
    validation_count: int = Field(ge=1)
    train_groups: int = Field(ge=1)
    validation_groups: int = Field(ge=1)
    mae: Finite
    rmse: Finite
    spearman: Finite | None


class DistilledModel(StrictModel):
    version: Literal[1, 2] = 1
    text_format: FeatureFormat = "instruction_response_json_v1"
    privacy_fingerprint: str | None = None
    feature_names: list[str] | None = None
    feature_mean: list[Finite] | None = None
    feature_scale: list[Finite] | None = None
    embedding: EmbeddingIdentity
    target: str
    rubric_hash: str = Field(min_length=1)
    alpha: float = Field(gt=0, allow_inf_nan=False)
    coefficients: list[Finite] = Field(min_length=1)
    intercept: Finite
    training_ids: list[str] = Field(min_length=2)
    validation_ids: list[str] = Field(min_length=1)
    input_hashes: dict[str, str]
    labels_hash: str
    evaluation: Evaluation

    @model_validator(mode="after")
    def validate_samples(self):
        if self.version == 1:
            if self.text_format != "instruction_response_json_v1" or any(
                field is not None
                for field in [self.feature_names, self.feature_mean, self.feature_scale]
            ):
                raise ValueError(
                    "Version 1 models require the original unscaled features"
                )
        elif (
            self.text_format != "pair_quality_v2"
            or self.feature_names != PAIR_FEATURE_NAMES
            or len(self.coefficients) != len(PAIR_FEATURE_NAMES)
            or self.feature_mean is None
            or len(self.feature_mean) != len(self.coefficients)
            or self.feature_scale is None
            or len(self.feature_scale) != len(self.coefficients)
            or any(scale <= 0 for scale in self.feature_scale)
        ):
            raise ValueError(
                "Version 2 models require compatible feature names and scaling"
            )
        train, validation = set(self.training_ids), set(self.validation_ids)
        if (
            train & validation
            or len(train) != len(self.training_ids)
            or len(validation) != len(self.validation_ids)
            or train | validation != set(self.input_hashes)
            or len(train) != self.evaluation.train_count
            or len(validation) != self.evaluation.validation_count
        ):
            raise ValueError("Invalid training/validation sample identities")
        return self

    def predict(self, vectors: np.ndarray | None) -> np.ndarray:
        x = validate_features(vectors, dimensions=len(self.coefficients))
        if self.version == 2:
            x = (x - np.asarray(self.feature_mean)) / np.asarray(self.feature_scale)
        predictions = x @ np.asarray(self.coefficients) + self.intercept
        if not np.isfinite(predictions).all():
            raise ValueError("Non-finite distilled predictions")
        return predictions


def validate_features(
    vectors: np.ndarray | None,
    *,
    count: int | None = None,
    dimensions: int | None = None,
) -> np.ndarray:
    if vectors is None:
        raise ValueError("Missing embeddings for distilled quality scoring")
    x = np.asarray(vectors, dtype=float)
    if (
        x.ndim != 2
        or not x.shape[0]
        or not x.shape[1]
        or (count is not None and x.shape[0] != count)
        or (dimensions is not None and x.shape[1] != dimensions)
        or not np.isfinite(x).all()
    ):
        raise ValueError("Invalid embedding shape or non-finite features")
    return x


def _split_groups(
    records: list[ConversationRecord],
) -> tuple[list[int], list[int], int, int]:
    """Keep connected sources, lineages and identical texts in the same split."""
    parents = list(range(len(records)))

    def root(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen: dict[tuple[str, str], int] = {}
    for i, record in enumerate(records):
        keys = [("text", input_hash(record))]
        keys.extend(
            ("lineage", value)
            for value in [record.id, record.lineage.root_id, *record.lineage.parent_ids]
            if value
        )
        if record.source.doc_id:
            keys.append(("document", record.source.doc_id))
        for key in keys:
            if key in seen:
                parents[root(i)] = root(seen[key])
            seen[key] = i
    groups: dict[int, list[int]] = {}
    for i in range(len(records)):
        groups.setdefault(root(i), []).append(i)
    if len(groups) < 2:
        raise ValueError("Need at least two independent source/lineage/text groups")
    # Hash ordering keeps the split deterministic even if input records are reordered.
    ordered = sorted(
        groups.values(),
        key=lambda group: hashlib.sha256(
            json.dumps(sorted(records[i].id for i in group)).encode()
        ).hexdigest(),
    )
    validation_groups = max(1, math.ceil(len(ordered) * 0.2))
    validation = sorted(i for group in ordered[:validation_groups] for i in group)
    training = sorted(i for group in ordered[validation_groups:] for i in group)
    if len(training) < 2:
        raise ValueError("Need at least two training records after group split")
    return training, validation, len(ordered) - validation_groups, validation_groups


def _spearman(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    def ranks(values):
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - (counts + 1) / 2)[inverse]

    a, b = ranks(actual), ranks(predicted)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def train_model(
    records: list[ConversationRecord],
    vectors: np.ndarray | None,
    embedding: EmbeddingIdentity,
    *,
    target: str = "quality",
    alpha: float = 1.0,
    feature_format: FeatureFormat = "instruction_response_json_v1",
    privacy_fingerprint: str | None = None,
) -> DistilledModel:
    """Fit on the training split only. The saved model never fits holdout labels."""
    if target != "quality" and not target.startswith("quality_per_dim."):
        raise ValueError("Target must be quality or quality_per_dim.<rubric dimension>")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if not records or any(not isinstance(r, ConversationRecord) for r in records):
        raise ValueError("Training requires judged conversation records")
    if len({r.id for r in records}) != len(records):
        raise ValueError("Duplicate training record IDs")
    if any(
        r.payload.turns or r.content_hash != content_hash(r.payload) for r in records
    ):
        raise ValueError(
            "Training requires unchanged single-turn conversation payloads"
        )
    if any(not r.scores.judge_model for r in records):
        raise ValueError("Training requires teacher model provenance")
    rubrics = {r.scores.rubric_hash for r in records}
    if len(rubrics) != 1 or not next(iter(rubrics)):
        raise ValueError(
            "Training requires one nonempty rubric hash across all records"
        )
    labels = [score_value(r, target) for r in records]
    if any(value is None or not math.isfinite(value) for value in labels):
        raise ValueError("Missing or non-finite rubric labels")
    y = np.asarray(labels, dtype=float)
    x = validate_features(vectors, count=len(records))
    training, validation, train_groups, validation_groups = _split_groups(records)
    if np.ptp(y[training]) == 0:
        raise ValueError("Training labels must contain at least two distinct scores")
    feature_mean, feature_scale = None, None
    if feature_format == "pair_quality_v2":
        if x.shape[1] != len(PAIR_FEATURE_NAMES):
            raise ValueError("pair_quality_v2 requires 12 features")
        feature_mean = x[training].mean(axis=0)
        feature_scale = x[training].std(axis=0)
        feature_scale[feature_scale < 1e-8] = 1
        x = (x - feature_mean) / feature_scale
    elif feature_format != "instruction_response_json_v1":
        raise ValueError(f"Unsupported distilled feature format: {feature_format}")
    x_mean, y_mean = x[training].mean(axis=0), y[training].mean()
    u, singular, vt = np.linalg.svd(x[training] - x_mean, full_matrices=False)
    coefficients = vt.T @ (
        (singular / (singular**2 + alpha)) * (u.T @ (y[training] - y_mean))
    )
    intercept = float(y_mean - x_mean @ coefficients)
    predicted = x[validation] @ coefficients + intercept
    errors = predicted - y[validation]
    return DistilledModel(
        version=2 if feature_format == "pair_quality_v2" else 1,
        privacy_fingerprint=privacy_fingerprint,
        text_format=feature_format,
        feature_names=PAIR_FEATURE_NAMES
        if feature_format == "pair_quality_v2"
        else None,
        feature_mean=feature_mean.tolist() if feature_mean is not None else None,
        feature_scale=feature_scale.tolist() if feature_scale is not None else None,
        embedding=embedding,
        target=target,
        rubric_hash=next(iter(rubrics)),
        alpha=alpha,
        coefficients=coefficients.tolist(),
        intercept=intercept,
        training_ids=[records[i].id for i in training],
        validation_ids=[records[i].id for i in validation],
        input_hashes={r.id: input_hash(r) for r in records},
        labels_hash=hashlib.sha256(
            json.dumps(
                sorted(zip([r.id for r in records], labels, strict=True))
            ).encode()
        ).hexdigest(),
        evaluation=Evaluation(
            train_count=len(training),
            validation_count=len(validation),
            train_groups=train_groups,
            validation_groups=validation_groups,
            mae=float(np.mean(np.abs(errors))),
            rmse=float(np.sqrt(np.mean(errors**2))),
            spearman=_spearman(y[validation], predicted),
        ),
    )
