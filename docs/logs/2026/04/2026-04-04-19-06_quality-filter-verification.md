# Quality filter verification

Date: 2026-04-04 19:06

Ran a live verification of the hardened single-judge quality filter path using OpenRouter.

## Verification setup

Created temporary verification inputs under `scratch/`:

- `scratch/verify-seeds.jsonl`
- `scratch/config.verify-openrouter.yaml`

The seed file had 3 examples:
1. strong gravity explanation
2. intentionally weak `Tell me stuff` / `Stuff.` example
3. strong factorial code example

Ran:

```bash
uv run arka --config scratch/config.verify-openrouter.yaml --run-id verify-quality-filter
```

## Produced artifacts

Under `scratch/runs/verify-quality-filter/`:

- `manifest.json`
- `report/run_report.json`
- `stages/01_source/data.parquet`
- `stages/02_normalize/data.parquet`
- `stages/03_label_quality/data.parquet`
- `stages/03_label_quality/dropped.parquet`
- `stages/03_label_quality/stats.json`

Dataset output:

- `scratch/output/verify-dataset.jsonl`

## Observed results

### run_report.json

The report showed the new stats correctly:

- `03_label_quality.count_in = 3`
- `03_label_quality.count_out = 2`
- `03_label_quality.dropped_count = 1`
- `drop_reasons = {"low_quality_score": 1}`
- `quality_distribution = {"mean": 3.6667, "std": 1.8856, "min": 1.0, "max": 5.0}`

Top-level report fields also reflected the same aggregated values.

### dropped.parquet

Verified dropped row contents:

```python
[{
  'id': 'seed-2',
  'drop_stage': '03_label_quality',
  'drop_reason': 'low_quality_score',
  'drop_detail': 'overall_score=1.0 < min_overall_score=3.5'
}]
```

This confirms the dropped artifact is present and inspectable.

### final dataset

The final dataset contained the 2 strong examples and excluded the weak one.

### kept stage artifact

Verified the kept stage parquet contains score metadata in `scores_json`, including:

- `quality`
- `quality_per_dim`
- `rubric_hash`
- `rubric_version`
- `judge_model`
- `judge_prompt_hash`

## Conclusion

The hardening work is functioning in a real run:

- dropped records are persisted
- drop reason is recorded
- stage stats are written
- `run_report.json` surfaces stage yields, drop reasons, and quality distribution
- final dataset excludes low-quality examples as expected

## Note

Because the config file lived under `scratch/`, the run root was also created under `scratch/runs/` and the dataset under `scratch/output/`. That behavior is consistent with the current CLI/project-root logic (`project_root = config_path.parent`).

## Next step

Proceed to label-path error taxonomy so failed label attempts can also be classified and persisted with explicit reason codes rather than only hard-failing.
