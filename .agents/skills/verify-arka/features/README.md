# Arka verification paths

Run the shared skill's Doctor first. Use `just verify` for fresh default evidence. Config validation and stage preview cover all current top-level example YAML files, but do not execute those example pipelines.

| User path | Default local recipe | Optional extension |
| --- | --- | --- |
| [Generated experiments and safe resume](experiments.md) | Experiments helper | Other version 2 stage/backend combinations |
| [PDF ingestion and provenance](pdf.md) | Cleanup helper | Repeated-header removal |
| [Generated text cleanup](cleanup.md) | Cleanup helper | Representative data audits |
| [Selection and benchmark filtering](selection.md) | Cleanup and decisions helpers | Native stack benchmark case |
| [Decision scoring and resume](decisions.md) | Decisions helper | Authorized live provider check |
| [Bounded answer repair](repair.md) | Repair helper | Representative quality and cost evaluation |
| [Source audits and explicit rebuilding](recovery.md) | Recovery helper | Representative recovery yield and cost |
| [Distilled scoring and source audits](quality.md) | Quality helper | Representative quality, total cost and native embedding checks |
| [Numeric ridge ranking](ridge.md) | Ridge helper | Archived representative holdout evaluations |
| [Reasoning and exports](reasoning.md) | ChatML in both helpers | Native language and HF token checks |

Every default workflow runs the installed CLI and checks resume. This map is a bounded set of verification recipes, not exhaustive product coverage. Preserve evidence and name every optional path left unverified.
