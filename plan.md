Plan:
1. Update `CheckpointManager` in `src/arka/pipeline/checkpoint.py` to create `embeddings_cache` and provide `save_embedding`/`load_embedding` methods.
2. Update `_build_run_report` to take `checkpoint_manager` in `src/arka/pipeline/runner.py`.
3. Update calls to `_build_run_report` to pass `checkpoint_manager`.
4. Update `_compute_diversity_score` to take `checkpoint_manager`.
5. Update `_embed_texts` to take `checkpoint_manager` and add the caching logic.
6. Update `SemanticSimilarityFilterStage.run` to pass `ctx.checkpoint_manager` to `_embed_texts`.
7. Add an entry to `.jules/bolt.md`.
8. Complete pre-commit checks and submit.
