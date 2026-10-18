---
type: research
status: complete
source: "Official model cards, Hugging Face model metadata and upstream source"
revision: "779040204b0f392465b7e6240bf9d9200816dc5d"
branch: "c-b/pii-redaction"
working_tree: "Separate PII implementation worktree; original workspace preserved"
research_date: "2026-10-17"
---
# Local detector options for user-defined PII coverage

The user requested that Arka users define language and entity coverage. This means selecting supported coverage through configuration, rather than fixing English or a universal category list in the implementation. An unsupported request must be rejected explicitly.

The selected protection boundary remains ingestion and generation before persistence or provider use. This document evaluates detector fit for that contract, not final anonymity guarantees.

## Evidence

The public [Hugging Face official benchmark dataset listing](https://huggingface.co/api/datasets?filter=benchmark:official&limit=500) returned 49 datasets during this investigation. None matched the PII, privacy, redaction or anonymization terms inspected. Therefore, this investigation did not obtain an official PII leaderboard ranking. Model-card benchmark results are reported by their authors, not measurements reproduced here.

| Option | User-configurable coverage | Operational implications |
| --- | --- | --- |
| Regex and validators | Only the identifiers and formats actually implemented by Arka. | No learned-model runtime is necessary, but patterns do not provide general contextual name and address recognition. |
| [Presidio](https://microsoft.github.io/presidio/analyzer/languages/) | Entity recognizers and language models can be configured. The default is English; additional languages need appropriate models and recognizers. | Optional analyzer/anonymizer and NLP dependencies. Per-language setup is more explicit than selecting arbitrary language codes. |
| [OpenAI Privacy Filter](https://github.com/openai/privacy-filter) | Eight native categories. Runtime filtering can select returned categories, but the upstream model does not learn new label policies from a config value. | Local CPU/GPU operation. The upstream package depends on PyTorch and uses supplied checkpoint files. It can download a missing checkpoint by default; Arka should require explicit local provisioning if selected. |
| [GLiNER2-PII](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi) | The model card explicitly supports selecting subsets of 42 labels at inference. Listed languages are English, French, Spanish, German, Italian, Portuguese and Dutch. | Local inference uses the GLiNER2 package with its local extra, including PyTorch and Transformers. A local checkpoint should be supplied explicitly. |

The model card for GLiNER2-PII reports limited precision on its SPY evaluation and warns about over-predicting names. Its evaluation does not establish quality on arbitrary PDFs or additional scripts. The OpenAI model card similarly describes distribution and language limits. Neither should be advertised as covering every user-supplied language or definition of PII.

## Metadata observed

- [openai/privacy-filter model metadata](https://huggingface.co/api/models/openai/privacy-filter): revision 7ffa9a043d54d1be65afb281eddf0ffbe629385b, 1,399,486,865 stored parameters, Apache 2.0.
- [fastino/gliner2-privacy-filter-PII-multi model metadata](https://huggingface.co/api/models/fastino/gliner2-privacy-filter-PII-multi): revision 1cb4166094dc58fa8d836429f060d6c95f62b495, 307,098,645 stored parameters, Apache 2.0.
- The GLiNER2 card also contains a 205M base-model description. The actual stored-parameter metadata is about 307M. This discrepancy is a reason to record the artifact revision and avoid sizing from prose alone.

Source dependencies were read from the upstream [OpenAI package](https://github.com/openai/privacy-filter/blob/main/pyproject.toml) and [GLiNER2 package](https://github.com/fastino-ai/GLiNER2/blob/main/pyproject.toml). The initial research did not install packages or download checkpoints. The later native verification, after user approval, is recorded below and in the verification document.

## Recommendation for discussion

Offer a narrow built-in regex backend and one optional local learned backend. GLiNER2 is the candidate for the latter because its configurable label subset matches the user's request and its current checkpoint is smaller than Privacy Filter's. This is a fit argument, not a claim of superior detection quality.

Do not add all candidate engines in the first version. Backend selection must be explicit. Optional model packages must not become mandatory core dependencies, and a missing model or unsupported language/entity must stop validation before the pipeline reads source text.

For learned detection, use returned character spans rather than looking up detected strings with str.find. Repeated values require their actual offsets. Long-input handling must detect or reject truncation; silently scanning a prefix is incompatible with the approved boundary.

Arka users should be able to set replacement tokens and drop thresholds. Detector configuration cannot imply support for categories or languages the backend cannot process.

The user subsequently selected GLiNER and authorized its optional dependencies, implementation and PR creation. The following update records the October 2026 research refresh. Production accuracy, memory and latency remain unmeasured.

## October 17, 2026 update

The [GLiNER2-PII paper](https://arxiv.org/html/2605.09973v1), published May 11, reports average exact-span F1 of 0.471 on SPY. It reports name over-prediction and evaluates two domains. This does not establish privacy recall for Arka datasets. Its table and the Hub card give slightly different headline scores, so keep the paper's figure tied to that table.

Fastino released [combined guardrails and PII](https://fastino.ai/blog/gliner2-guardrails-pii-multi-safety-moderation-privacy-filtering-small-language-model) on July 8. That checkpoint adds moderation tasks. Issue #166 requires PII protection, so the additional moderation behavior is not needed here.

[GLiNER2.5](https://fastino.ai/models/gliner2-5), released August 24, uses boundary prediction and supports extraction features beyond the earlier span model. The [multilingual checkpoint card](https://huggingface.co/fastino/gliner2.5-multi-v1) describes general schema extraction. Its newer architecture alone is not evidence that it is a better PII checkpoint for the agreed categories.

The September 29 study [Strong Multilingual Privacy Tagging at Encoder Speed](https://arxiv.org/abs/2609.38630v3), revised October 13, reports better privacy-redaction results than its GLiNER2 baseline on its own human-annotated evaluation. This is a useful counterexample to calling GLiNER universally best. Its dataset and metric differ from the SPY comparison, so the scores cannot be combined into one ranking. These results were not reproduced in Arka.

The current [GLiNER2 long-context documentation](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/12-long_context.md) explicitly warns that ordinary max_len extraction truncates. Its long API uses overlapping windows and returns global character offsets. Arka uses that API, checks offsets against original text and keeps inference local.

PyPI currently provides [gliner2 2.1.0](https://pypi.org/project/gliner2/2.1.0/). Its local extra adds PyTorch, Transformers, PEFT and safetensors. Arka pins this package version in an optional extra. The provisioned model uses the unchanged Hub revision recorded above. Runtime loading requires complete local files and requests local-only checkpoint access.

The selected initial implementation uses the dedicated GLiNER2-PII checkpoint and retains the narrow regex option. This follows the user's model-family choice and the documented configurable PII schema. It is not a leaderboard claim. Real local inference and CLI integration fixtures are recorded separately in [verification](03-verification.md).
