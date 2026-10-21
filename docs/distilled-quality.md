# Train and reuse a quality classifier

The distilled scorer learns pass/fail labels from existing judge scores. It embeds the system message, instruction and response, then fits a linear logistic classifier. Training uses scikit-learn. Inference uses saved numeric coefficients and NumPy. It makes no judge calls and never retrains.

Treat the scorer as experimental until you measure false rejections and total cost on your workload. A probability estimates agreement with the training labels. It does not establish factual correctness.

## Prepare labels

Use the existing `labeling_score` stage on representative inputs. It preserves passing and failing records. Both LLM and decision backends are supported.

```yaml
pipeline:
  - type: seed_source
    path: ./source-a.jsonl
  - type: normalize_conversation
  - type: labeling_score
    rubric_path: ../rubrics/sft_quality.yaml
    backend: llm
```

Repeat with independent data sources and run IDs. Keep the resulting `02s_label_score/data.parquet` files. Training reads those files directly. Final dataset exports do not contain the label provenance needed for training.

Each input must be a single-turn conversation with valid scores for every rubric dimension, a matching rubric hash and a judge model. LLM labels require a prompt hash. Decision labels without a prompt hash require matching decision answers with confidence of at least 0.5. Missing labels, invalid scales and inconsistent content hashes stop training. The scorer does not relabel records or silently discard bad training inputs.

Apply privacy protection before judging. If the training policy would change or reject the saved text, training stops. Protect and judge the records again under that policy before training.

## Train in a separate run

Use [the training example](../examples/11-distilled-quality-train.yaml) after replacing its `training_paths` with your labeled artifacts. That pipeline contains only `distilled_quality_train`.

```bash
uv run arka --config examples/11-distilled-quality-train.yaml --run-id quality-train
```

`min_scores` defines pass/fail labels. Every configured minimum must be met for a passing label. These values must fit the rubric. `holdout_fraction`, default 0.25, selects a fraction of whole independent groups for evaluation. `split_seed`, default 0, controls the deterministic split. `evaluation_threshold`, default 0.5, determines which holdout predictions count as selected.

Records sharing a source, lineage root or exact model input stay together, including relationships that connect several groups. Ungrouped records use lineage and text relationships. Each partition must contain passing and failing labels. A single seed file is one source group, so provide more independent sources when it cannot be split. Arka fails instead of splitting related records to satisfy a requested holdout fraction.

The stage writes `model.json`, `quality_identity.json` and `stats.json` under `runs/<run-id>/stages/03_distilled_quality_train/`. Training exports an empty dataset. The model artifact contains coefficients, the input format version, rubric and embedding identities, privacy policy identity, teacher model names, label identity and hashed population identifiers. It contains no training text or Python object serialization.

The holdout report includes useful-record recall, false rejections, false acceptances and retained precision. These metrics use teacher labels. The report explicitly marks independent factual quality as unmeasured. The holdout is useful for model assessment; use another representative population to assess the complete filtering workflow and choose a threshold.

## Score before filtering

[The scoring example](../examples/12-distilled-quality-score.yaml) loads the trained model in annotation mode.

```yaml
- type: distilled_quality_score
  model_path: ./runs/quality-train/stages/03_distilled_quality_train/model.json
  rubric_path: ../rubrics/sft_quality.yaml
  min_scores:
    instruction_clarity: 3
    response_quality: 3
```

Predictions are stored in `scores.distilled_quality`. Model, rubric and input hashes are stored in the `distilled_quality` annotation. Existing teacher scores and taxonomy metadata remain unchanged. Annotation preserves record order. Unsupported record types and conversations with turns pass through and are counted as unsupported.

Set `min_probability` explicitly to enable filtering. Rejected records appear in `dropped.parquet` with reason `distilled_quality_below_min`. Follow the scorer with the existing `labeling_engine` stage to judge and filter survivors. There is no automatic judge routing.

Set `evaluate: true` only when all supported input records already have matching teacher labels. The stage reports the same confusion counts and marks `evaluation_independent: false` when any evaluated text, source or lineage root appeared in model development. Evaluation makes no new judge calls. It does not silently judge missing labels.

The model must match the current rubric, minimum scores, embedding provider/model/endpoint and privacy policy. Missing, malformed or incompatible artifacts fail explicitly. Embedding failures, nonfinite vectors and dimension changes also stop the run. Use the same embedding configuration for training and scoring. A model name cannot detect a provider silently replacing its weights.

## Resume and evaluate cost

Use the same command and run ID with `--resume`. Arka checks external model, rubric and training files before reusing a completed stage. Changed dependencies require a new run ID. Compatible completed stages do not retrain, relabel or make embedding calls on resume. An incomplete training stage may train again; available embedding checkpoints can be reused.

Measure the existing direct LLM and decision baselines on the same workload before adopting a threshold. Preserve all labels in a separate evaluation run so you can count useful records rejected by the student. Review factual quality independently, and examine results for each relevant domain and language.

Compare total cost across training labels, holdout labels, embeddings, student inference, later judge calls and audits. Stage reports include record and embedding-text counts, while normal judge stages report their usage and known cost. Elapsed time is available in the run artifacts. Existing embedding integration does not report complete monetary usage, so scorer reports set `cost_complete: false`. Obtain missing costs separately. A lower downstream judge count alone is not proof of lower total cost.

Run `just verify` for local CLI proof, or `uv run python scripts/verify_quality_cli.py` for this recipe alone. It uses synthetic vectors and loopback providers. It verifies mechanics without proving representative model quality or savings.
