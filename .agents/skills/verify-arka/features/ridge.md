# Numeric ridge ranking

Run `uv run python scripts/verify_ridge_cli.py` or the shared `just verify`
workflow. This drives the installed training and pipeline commands against a
local embedding HTTP fixture. No judge calls or credentials are required.

The recipe trains version 1 and version 2 models from 30 saved judged records,
scores them with `distilled_ridge`, and selects three records. It checks model
metadata, exact version 1 ranking, completed resume without extra requests,
changed-model rejection, and private training rejection before embedding calls.

Inspect `result.json`, `command-*.log`, both models, and the run Parquet files.
The local vectors prove transport and pipeline behavior. They do not establish
quality. The archived reports in `docs/verification/issue-165/` and
`docs/verification/issue-165-v2/` describe separate public-data evaluations.
The scorer remains experimental because the improvement interval includes zero.

Source ownership:

- `src/arka/cli.py` owns separate training and privacy preflight.
- `src/arka/distillation.py` owns features, group splits and versioned models.
- `src/arka/pipeline/distilled_ridge.py` owns annotation and resume identity.
- `src/arka/decisions/scores.py` rejects stale ranking predictions.
- `tests/unit/test_distilled_ridge.py` covers failure cases and compatibility.

The existing probability classifier and its quality verification recipe remain
separate. This recipe does not prove multilingual quality or total cost savings.
