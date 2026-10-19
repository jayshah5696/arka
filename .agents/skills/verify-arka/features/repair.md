# Bounded answer repair

## Sub-features

Rubric thresholds, explicit repair dimensions, one attempt per admitted record, a run limit, passing originals, replacement lineage, source evidence, privacy rejection, downstream deduplication, usage and resume.

## How to get to it

Use `examples/10-quality-repair.yaml` or [the configuration guide](../../../../docs/quality-repair.md). Run the CLI with `--resume` to reuse persisted judgments and replacements.

## Drive

Run `just verify` and inspect `repair/`, or run `uv run python scripts/verify_repair_cli.py --output /tmp/arka-repair-proof` with a new output directory.

The seven scenarios make 27 local requests in total. Inspect each scenario's `runs/repair/stages/03_quality_repair/` directory and logs.

- `bounded` judges six originals and attempts three repairs. It accepts two repairs before downstream deduplication. Three records leave repair, and two reach export. Drop reasons distinguish exhaustion, excluded dimensions and the attempt limit.
- `baseline` sets the repair limit to zero and exports no records.
- `pdf` includes the exact source chunk in generation, both judgments and repair. It exports one repaired answer.
- `missing-evidence` makes one ordinary generation request, then rejects the record without judging or repairing it.
- `decision` makes two decision requests and one repair request. Its earlier panel makes no requests for repair judgments.
- `privacy` records a protected PII rejection and usage, with no second judgment or export.
- `interrupted` persists generation, fails its second judgment, and resumes with exactly one judgment request. It never repeats the persisted generation.

Completed resume must preserve dataset bytes and make zero requests. Changed rubric contents and changed incoming stage artifacts must be rejected before provider calls. Inspect `changed-rubric.log` and `changed-input.log`.

## Limits

Fixture scores prove behavior and transport wiring. They do not establish independent answer quality or cost savings. `accepted_repairs` precedes downstream validation. Native language, Hugging Face and live-provider checks remain separate recipes. A crash before response persistence can repeat a provider call.
