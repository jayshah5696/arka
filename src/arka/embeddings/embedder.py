"""Embedder: produce vectors for texts, with provider routing + checkpoint cache.

Lifted out of ``PipelineRunner`` (~150 lines of private methods) so any Stage
that needs embeddings -- diversity scoring in the run report,
``SemanticSimilarityFilterStage``, future semantic dedup -- can share the
same caching seam instead of reaching into the runner.\n\nThe public surface is one method::

    Embedder(config).embed(texts, checkpoint_manager=...)
        -> np.ndarray | None

Provider selection (huggingface vs openai-compatible) and per-text caching
keyed on ``(provider, model, text)`` live behind the seam. The cache backs
to the run's :class:`CheckpointManager` -- the same persistence layer the
runner already uses for resume.

K-means cluster labelling lives here too because it is the only consumer
of embeddings outside the runner today; keeping it next to ``embed`` makes
``compute_diversity_score`` a small composition.
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

import numpy as np

from arka.config.models import LLMConfig, ResolvedConfig
from arka.llm.openai_client import build_openai_client

if TYPE_CHECKING:
    from arka.pipeline.checkpoint import CheckpointManager
    from arka.records.models import Record


class Embedder:
    """Provider-routing, cache-aware embedder for a single ResolvedConfig."""

    def __init__(self, config: ResolvedConfig) -> None:
        self._config = config

    def embed(
        self,
        texts: list[str],
        *,
        checkpoint_manager: CheckpointManager | None = None,
    ) -> np.ndarray | None:
        """Return one vector per text, or ``None`` if any text fails to embed.

        Cached per (provider, model, text) in the run's CheckpointManager so
        ``--resume`` runs do not re-call the embedding API or re-load the
        HuggingFace model. Cache key construction was previously inline in
        ``PipelineRunner._embed_texts``.
        """
        if not texts:
            return None

        vectors: list[np.ndarray | None] = [None] * len(texts)
        texts_to_embed: list[tuple[int, str, str]] = []

        embeddings_cfg = self._config.embeddings
        model_key = f"{embeddings_cfg.provider}:{embeddings_cfg.model}:"

        for i, text in enumerate(texts):
            text_hash = model_key + hashlib.sha256(text.encode("utf-8")).hexdigest()
            cached = (
                checkpoint_manager.load_embedding(text_hash)
                if checkpoint_manager
                else None
            )
            if cached is not None:
                vectors[i] = np.frombuffer(cached, dtype=float)
            else:
                texts_to_embed.append((i, text_hash, text))

        if texts_to_embed:
            new_texts = [item[2] for item in texts_to_embed]
            new_vectors = self._embed_uncached(new_texts)
            if new_vectors is not None and len(new_vectors) == len(new_texts):
                for (idx, text_hash, _), vec in zip(
                    texts_to_embed, new_vectors, strict=True
                ):
                    vectors[idx] = vec
                    if checkpoint_manager is not None:
                        checkpoint_manager.save_embedding(
                            text_hash, vec.astype(float).tobytes()
                        )

        if any(v is None for v in vectors):
            return None
        return np.array(vectors, dtype=float)

    def compute_diversity_score(
        self,
        *,
        records: list[Record],
        checkpoint_manager: CheckpointManager | None = None,
    ) -> float | None:
        """Normalised cluster entropy across record diversity texts.

        Used by the run report. Returns ``None`` when there are fewer than
        two embeddable texts (or when embedding fails).
        """
        instructions = [
            text
            for record in records
            if (text := record.text_for_diversity()) is not None
        ]
        if len(instructions) < 2:
            return None

        embeddings = self.embed(instructions, checkpoint_manager=checkpoint_manager)
        if embeddings is None or len(embeddings) < 2:
            return None

        cluster_count = min(50, len(embeddings))
        if cluster_count < 2:
            return None
        labels = _kmeans_labels(embeddings, cluster_count=cluster_count)
        counts = np.bincount(labels, minlength=cluster_count)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        score = float(entropy / math.log(cluster_count))
        return round(score, 4)

    # --- Private provider routing ---

    def _embed_uncached(self, texts: list[str]) -> np.ndarray | None:
        if self._config.embeddings.provider == "huggingface":
            return self._embed_huggingface(texts)
        return self._embed_openai(texts)

    def _embed_huggingface(self, texts: list[str]) -> np.ndarray | None:
        model_name = _resolved_huggingface_embedding_model(
            self._config.embeddings.model
        )
        try:
            from fastembed import TextEmbedding

            embedding_model = TextEmbedding(model_name=model_name)
            vectors = list(embedding_model.embed(texts))
        except Exception:
            return None
        if not vectors:
            return None
        return np.array(vectors, dtype=float)

    def _embed_openai(self, texts: list[str]) -> np.ndarray | None:
        llm_config = _embedding_llm_config(self._config)
        client = build_openai_client(llm_config)
        try:
            response = client.embeddings.create(
                model=self._config.embeddings.model,
                input=texts,
            )
        except Exception:
            return None
        vectors = [item.embedding for item in response.data]
        if not vectors:
            return None
        return np.array(vectors, dtype=float)


def _resolved_huggingface_embedding_model(model: str) -> str:
    if "/" in model:
        return model
    return f"sentence-transformers/{model}"


def _embedding_llm_config(config: ResolvedConfig) -> LLMConfig:
    """Project the embedding-specific config onto an LLMConfig for the OpenAI client.

    Falls back to the top-level ``llm`` block for any field the embeddings
    block leaves unset, matching the historical behaviour in
    ``PipelineRunner._embedding_llm_config``.
    """
    embedding_cfg = config.embeddings
    api_key = embedding_cfg.api_key or config.llm.api_key
    base_url = embedding_cfg.base_url or config.llm.base_url
    timeout_seconds = (
        embedding_cfg.timeout_seconds
        if embedding_cfg.timeout_seconds is not None
        else config.llm.timeout_seconds
    )
    max_retries = (
        embedding_cfg.max_retries
        if embedding_cfg.max_retries is not None
        else config.llm.max_retries
    )
    openai_compatible = embedding_cfg.openai_compatible or config.llm.openai_compatible
    return LLMConfig(
        provider="openai",
        model=embedding_cfg.model,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        openai_compatible=openai_compatible,
    )


def _kmeans_labels(embeddings: np.ndarray, cluster_count: int) -> np.ndarray:
    """Tiny k-means implementation used for diversity scoring.

    Pure NumPy, fixed seed, capped at 20 iterations -- intentionally minimal
    because diversity scoring is a single point summary, not a full
    clustering pipeline.
    """
    rng = np.random.default_rng(0)
    indices = rng.choice(len(embeddings), size=cluster_count, replace=False)
    centroids = embeddings[indices].copy()
    labels = np.zeros(len(embeddings), dtype=int)

    for _ in range(20):
        distances = np.linalg.norm(
            embeddings[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
        )
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster_index in range(cluster_count):
            members = embeddings[labels == cluster_index]
            if len(members) == 0:
                centroids[cluster_index] = embeddings[rng.integers(0, len(embeddings))]
            else:
                centroids[cluster_index] = members.mean(axis=0)
    return labels
