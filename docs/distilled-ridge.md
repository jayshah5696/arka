# Experimental ridge ranking

The `distilled_ridge` stage predicts a numeric rubric score. It is separate from
the [distilled quality classifier](distilled-quality.md), whose `distilled_quality`
score is a pass probability. Keep these scales separate in selection weights.

Train a small model from existing rubric judgments, then use its predictions to
rank records. Training uses NumPy ridge regression over frozen embeddings and
text measurements. No new dependencies are required.

The [first real-data evaluation](verification/issue-165/README.md) found weak
ranking agreement for the original combined-text embedding representation on 600
short English records. Treat this scorer as experimental and validate it on your
own data before using its ranking for selection.

The [fresh holdout evaluation of version 2](verification/issue-165-v2/README.md)
found higher point estimates for the revised features. The paired confidence
interval still included no improvement over the original representation, so
the scorer remains experimental.

## Train a model

Use an Arka `data.parquet` artifact containing judged conversation records. Every
record must have a finite target score, a nonempty `scores.judge_model`, an intact
content hash, and the same nonempty `scores.rubric_hash`. Training accepts only
single-turn conversations. With a privacy policy configured, training rejects
text that still needs protection before making embedding requests. Protect and
judge the records first. The model records the policy fingerprint, and inference
requires the same policy.
Both existing judge backends can supply rubric scores. Final exported JSONL files
contain payloads rather than the score and lineage metadata needed for training.

```sh
uv run arka-train-scorer \
  --config config.yaml \
  --records runs/judged/stages/03_labeling_engine/data.parquet \
  --output models/quality.json
```

Use the actual stage artifact path from your run. `--target` defaults to `quality`.
Use `--target quality_per_dim.response_quality` to predict one rubric dimension.
`--alpha` controls ridge regularization and defaults to `1.0`. It must be finite
and positive. The command refuses to overwrite an existing model.

`--features` defaults to `pair_quality_v2`. This representation embeds the
instruction and response separately, then computes their cosine similarity.
It also computes 11 text measurements covering length, word overlap, and
response structure. The linear head uses these 12 features. Feature means and
scales are fitted on the training partition and saved with the coefficients.
Constant features use a scale of one. Inference uses the saved means and scales.

Use `--features instruction_response_json_v1` to reproduce the original model
over one combined instruction-answer embedding. Existing version 1 artifacts
still load with their original representation. Version 2 artifacts declare
`pair_quality_v2` and include validated feature names and scaling metadata.

The command reads embedding settings from `--config`. It does not execute the
pipeline or make judge requests. Embedding texts can require new computation or
provider calls. Version 2 uses exact instruction and response texts, which can
reuse matching entries in the run's embedding cache. Version 1 embeds a combined
JSON string. Its features differ from the instruction embeddings used in a
diversity report. Measure actual cache reuse rather than assuming it.

The model file contains numeric coefficients, the target score path, rubric hash,
embedding identity, text format version, and training and validation record IDs.
It also contains input hashes and a hash of the labels. It does not contain raw
training text. The command prints the model's SHA-256 and evaluation results.

## Score and rank records

Copy the printed SHA-256 into the stage configuration. Use the same embedding
provider, model, and API endpoint as training. Add these stages after your
conversation source or generator:

```yaml
pipeline:
  # Add your conversation source or generator before these stages.
  - type: distilled_ridge
    model_path: models/quality.json
    model_sha256: "<64-character SHA-256 printed by training>"
  - type: select
    target_count: 100
    weights:
      distilled_ridge: 1.0
```

See [the complete configuration example](examples/distilled-ridge.yaml). Paths
in pipeline configuration are relative to the configuration file's directory.
Training command paths are relative to the current working directory.

Run inference with the usual `uv run arka --config config.yaml` command. The
scoring stage preserves record order and passes unsupported record types and
multi-turn conversations through. It writes `scores.distilled_ridge` and metadata
under `scores.annotations.distilled_ridge`, exposed through the typed
`scores.distilled_ridge_prediction` property. The metadata contains the model
SHA-256, input hash, target, and rubric hash. Existing judge
scores and taxonomy metadata remain intact. A later scoring stage replaces the
previous distilled prediction; this version supports one prediction per record.

The scoring stage does not drop records. The optional `select` stage applies the
configured ranking. Predictions use the target's original numeric scale and are
not clipped, so they can fall outside the rubric's range. Do not mix predictions
from different targets or rubrics in one ranking without checking their scales.

## Evaluate and reuse

Training uses a deterministic 80/20 split of independent groups. Records connected
by document ID, lineage IDs, or identical instruction-answer text stay together.
The split uses about 20 percent of groups for validation. Record counts can differ
from 80/20 when groups have different sizes. The saved model fits only the training
partition. It never fits validation labels.

The report includes the actual record and group counts, mean absolute error,
root mean squared error, and Spearman rank correlation with average ranks for
ties. Spearman is `null` if either ranking is constant or validation contains only
one record. Training fails if there are fewer than two independent groups, fewer
than two training records, or no variation in training scores. Missing labels,
mixed rubrics, invalid embeddings, and incompatible artifacts also produce errors.

Choose representative records before training. The initial evaluation assumption
is English instruction-answer data. The code does not detect language or prove
agreement in other domains. Teacher agreement measures imitation of the existing
judge. It does not establish factual correctness. No quality threshold or cost
savings claim is built into this version.

Loading a model does not train it again or request new labels. Inference uses the
existing embedding checkpoint cache. The model file must match its configured
SHA-256 before pipeline execution. Resume also checks a saved identity for the model,
configuration, privacy policy, and input records. To use a different model, start a new run ID. Composite
selection rejects a prediction if the instruction or answer changed after scoring.

Use a versioned embedding model and an encoder whose text limit fits your records.
Provider-side changes under the same model name cannot be detected by the local
artifact identity. Long records can be truncated by the embedding provider.
