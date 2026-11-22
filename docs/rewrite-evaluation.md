# Evaluate Arka selection on a rewrite task

The question is whether choosing examples with Arka improves rewrites at the same training cost. A higher judge score on selected training examples cannot answer it.

1. Reserve real rewrite requests before generating examples. Include the original passage, edit requested, facts to preserve, and required output format. Keep each seed and its descendants together using `lineage.root_id`. Group PDF examples by `source.doc_id` to keep overlapping chunks together. Remove exact duplicates across partitions and inspect near matches.
2. Generate one candidate pool. Save its generator-stage Parquet artifact and the final selected artifact. Sample the baseline from the full pool with a recorded random seed. Draw both training sets only from training families. Save record IDs, root IDs, and the artifact hashes so another run can recover the selection.
3. Render both training sets with the same pinned model revision, tokenizer, and chat template. For completion-only SFT, split each conversation into a conversational `prompt` (all preceding messages) and `completion` (the final assistant message). Inspect the trainer's actual labels on a short example and a truncated example. Verify that the expected answer has supervised tokens. Count non-padding input tokens and supervised tokens separately.
4. Evaluate the base model before training. Fine-tune two copies with identical optimizer, adapter settings, seed, truncation, and batch size. Use the same number of training tokens in each arm. Record actual token counts, selection cost, and training time. Evaluate all three checkpoints on the reserved prompts with identical decoding settings.
5. Shuffle and blind the outputs. Check required format and named facts for each prompt. Ask reviewers which rewrite best follows the request, allowing ties. Report the per-prompt wins, ties, losses, format pass rate, and fact errors. Estimate uncertainty by resampling held-out seed families. Report the size of the holdout and unresolved reviewer disagreements.

A run is ready to interpret when it has a saved split, training manifests, model revisions, decoding settings, and blinded judgments. This recipe specifies an experiment. The CLI fixtures verify data handling only.

[Hugging Face TRL supervised fine-tuning](https://huggingface.co/docs/trl/en/sft_trainer)
