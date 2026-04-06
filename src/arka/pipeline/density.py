from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from arka.config.models import ResolvedConfig
from arka.core.paths import RunPaths
from arka.pipeline.output import OutputWriter
from arka.pipeline.runner import PipelineRunner

logger = logging.getLogger(__name__)


class DensityAnalyzer:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.output_writer = OutputWriter()

    def get_sparse_embeddings(
        self,
        config: ResolvedConfig,
    ) -> np.ndarray | None:
        reference_run_id = config.density_controller.reference_run_id
        if not reference_run_id:
            logger.warning("Density controller enabled but no reference_run_id provided.")
            return None

        run_paths = RunPaths.bootstrap(root_dir=self.project_root, run_id=reference_run_id)

        if not run_paths.manifest_path.exists():
            logger.warning("Reference run manifest not found at %s", run_paths.manifest_path)
            return None

        try:
            manifest = json.loads(run_paths.manifest_path.read_text())
            dataset_path_str = manifest.get("dataset_path")
            if not dataset_path_str:
                logger.warning("Reference run manifest does not contain a dataset_path.")
                return None
            dataset_path = Path(dataset_path_str)
            if not dataset_path.is_absolute():
                dataset_path = self.project_root / dataset_path
        except Exception as e:
            logger.warning("Failed to parse reference run manifest: %s", e)
            return None

        # Determine output format and read instructions from dataset
        # Wait, the dataset is usually written as jsonl by output writer. We can just read the final records from the last stage's parquet file
        # Actually it's easier to load the records from the last stage directly, or we can use OutputWriter to read the dataset?
        # The runner writes the final dataset in jsonl, but the parquet file of the final stage is a List[Record]. Let's find the last stage from manifest

        stage_names = manifest.get("stage_names", [])
        if not stage_names:
            logger.warning("No stages found in reference run manifest.")
            return None

        last_stage = stage_names[-1]
        last_stage_parquet = run_paths.stage_data_path(last_stage)

        if not last_stage_parquet.exists():
            logger.warning("Last stage parquet not found at %s", last_stage_parquet)
            return None

        try:
            records = self.output_writer.read_parquet(last_stage_parquet)
        except Exception as e:
            logger.warning("Failed to read last stage parquet for reference run: %s", e)
            return None

        instructions = [
            text
            for record in records
            if (text := record.text_for_diversity()) is not None
        ]

        if not instructions:
            logger.warning("No instructions extracted from reference run for density analysis.")
            return None

        runner = PipelineRunner(self.project_root)
        embeddings = runner._embed_texts(config=config, texts=instructions)

        if embeddings is None or len(embeddings) < 2:
            logger.warning("Failed to embed reference run instructions or not enough embeddings.")
            return None

        k_neighbors = config.density_controller.k_neighbors
        if len(embeddings) <= k_neighbors:
            k_neighbors = max(1, len(embeddings) - 1)

        # Compute distance to k-th nearest neighbor for each point
        # Since sklearn is not listed in dependencies (only numpy, fastembed, polars, etc), we compute it with numpy
        n_samples = len(embeddings)
        kth_distances = np.zeros(n_samples)

        for i in range(n_samples):
            # Compute distances from embeddings[i] to all other points
            diff = embeddings - embeddings[i]
            dist = np.linalg.norm(diff, axis=1)
            # Partition to find the k-th smallest distance (which is at index k, since index 0 is distance 0 to itself)
            # using argpartition for efficiency
            idx = np.argpartition(dist, k_neighbors)
            kth_distances[i] = dist[idx[k_neighbors]]

        threshold_percentile = config.density_controller.sparse_threshold_percentile

        # We want the sparsest regions, which means the largest k-th nearest neighbor distances
        # e.g. top 20% means percentile 80
        threshold_val = np.percentile(kth_distances, 100 - threshold_percentile)

        sparse_indices = np.where(kth_distances >= threshold_val)[0]
        if len(sparse_indices) == 0:
            sparse_indices = np.array([np.argmax(kth_distances)])

        sparse_embeddings = embeddings[sparse_indices]
        return sparse_embeddings
