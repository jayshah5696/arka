# Distilled scoring and source audits

Run the Doctor in the parent skill. Use `just verify`, or run `uv run python scripts/verify_quality_cli.py --output /tmp/arka-quality-proof` with a new output directory.

## Source map

- `src/arka/config/models.py` defines `DistilledQualityTrainConfig`, `DistilledQualityScoreConfig` and `SourceAuditConfig`.
- `src/arka/pipeline/registry.py` registers all three stages. `decision_stages.py` registers the audit's evidence-aware decision questions and excludes them from panels.
- `distilled_quality.py` loads teacher artifacts, separates connected source/lineage/text groups, trains a logistic classifier, validates model metadata and predicts without retraining.
- `source_audit.py` owns deterministic source samples, per-judgment progress and observed failure reports.
- `quality_judging.py` shares rubric validation, permitted chunk resolution and both judge backends with answer repair.
- `quality_artifacts.py` owns atomic operation-state writes, identity hashes and source grouping helpers.
- `records/models.py` stores `distilled_quality` separately from teacher scores.
- `scripts/verify_quality_cli.py` drives actual installed CLI processes against loopback chat, decision and embedding endpoints. `scripts/verify_local.py` includes this helper in default verification.

## Drive and inspect

The helper retains every command, elapsed time, exit code and log path in `result.json`. It saves HTTP requests in `requests.json` and stops only its own fixture server on exit.

Check these paths in the evidence directory:

- `labels-0/` through `labels-3/` contain 16 labels from independent sources, split across LLM and decision judges. Their `02s_label_score/data.parquet` artifacts retain label provenance.
- `train/runs/quality/stages/03_distilled_quality_train/model.json` contains coefficients and compatible metadata. Its training and holdout groups are disjoint. The report records 12 training records and four holdout records. Training calls embeddings, with no new judge calls.
- `score/` annotates four records without teacher scores. `prefilter/` rejects two of four records before exactly two direct judge calls. `evaluate/` scores a distinct labeled source and marks the evaluation independent of model development.
- Changed-model and changed-label logs show failure before HTTP requests on resume. The audit's changed-rubric log checks the same boundary.
- `audit/` samples all four records in a small source, preserves their payloads and existing scores, and reports two passing and two failing judgments.
- `grounded-audit/` supplies the exact PDF chunk to its judgment. `missing-evidence-audit/` reports missing evidence and makes no audit judgment call.
- `interrupt/` saves its first judgment before a fixture error. Resume makes three judgment calls for the remaining sample. A subsequent completed resume makes zero requests.
- `private-audit/` retains protected reasoning and reports privacy rejection without counting a quality failure. `uncertain-audit/` does the same for a low-confidence decision. Both report a null failure rate.

Completed resumes preserve dataset bytes and make zero requests. At the time of this recipe update, the complete helper uses 49 local requests. Its request counts and synthetic cost fields describe the fixture only.

## Limits

This recipe proves CLI behavior with deliberately separable synthetic vectors. It does not establish teacher correctness, useful-record recall on real data, total cost savings or native Hugging Face model behavior. No live credentials, model downloads or paid calls are needed. Evaluate those paths separately before adopting a filtering threshold.
