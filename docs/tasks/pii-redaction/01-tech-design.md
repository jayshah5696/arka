---
type: technical-design
status: draft
approval: "User approved ingestion and generation protection boundaries, requested user-configurable coverage, and authorized finishing the design, implementing #166 and opening a PR. Backend/dependency selection and remaining behavior are still under discussion."
source: "Issue #166 and 02-research-detector-options.md"
revision: "779040204b0f392465b7e6240bf9d9200816dc5d"
branch: "c-b/pii-redaction"
---
# Technical design for PII redaction

## Scope

Address [issue #166](https://github.com/jayshah5696/arka/issues/166) before designing the other open issues. The user selected this issue on October 17, 2026 and later instructed us to finish the design and implement the feature before opening a PR. Coverage must be configurable by Arka users. Backend and dependency choices still need agreement.

The [detector research](02-research-detector-options.md) records current public practices and available detector families. The [updated issue](https://github.com/jayshah5696/arka/issues/166) records options and their limitations. This document holds the agreed design and identifies pending decisions.

## Current system behavior

The local branch was fetched and verified equal to origin/main at the revision above.

Arka persists source records before downstream filters. PDF ingestion can call a decision provider within the source stage. Prompt-based generation persists raw responses before parsing and downstream filters. Therefore, placement of a later PII filter does not protect those earlier destinations.

A simplified current flow is:

~~~mermaid
flowchart TD
    A[Read seed file or PDF] --> B[Build source records]
    B --> C[Optional PDF decision request]
    C --> D[Save source records]
    D --> E[Generation request]
    E --> F[Save raw generated response]
    F --> G[Parse and build generated records]
    G --> H[Run judges and filters]
    H --> I[Export dataset]
~~~

The diagram summarizes the relevant boundaries. The optional decision request is specific to configured PDF ingestion. Individual generator strategies have additional internal calls that need to be covered if broad protection is selected.

Relevant code observations:

- [Source stages](../../../src/arka/pipeline/source_stages.py) own ingestion and the optional PDF decision check.
- [Runner](../../../src/arka/pipeline/runner.py) writes stage records and the final dataset.
- [Prompt and transform generators](../../../src/arka/pipeline/generator_stages.py) own their generation calls and prompt-generator raw-response persistence.
- [Evol-Instruct](../../../src/arka/pipeline/evol_generator_stage.py) has intermediate generation calls and its own raw-response artifact.
- [OutputWriter](../../../src/arka/pipeline/output.py) serializes payloads, scores and stage-event details. It currently performs no PII detection.

Evol-Instruct generates an intermediate instruction and sends it into another generation request before returning a completed record. Consequently, scanning only at the end of a generator stage would miss an approved protection boundary. A malformed prompt-generator response can also enter raw-response storage and drop details before a complete record exists.

Stages normally obtain LLM and decision clients through StageContext. These are separate construction paths. This is useful context for the later ownership decision, but no client wrapper or source integration API has been selected yet.

## System design

The user selected protection at ingestion and generation boundaries through the question tool on October 17, 2026. The selected contract checks source record text locally before saving it or sending it to a model, and checks generated text before saving it or using it in later model calls.

The intended flow is:

~~~mermaid
flowchart TD
    A[Read seed file or PDF] --> B[Local PII detection and agreed policy]
    B --> C[Protected source text]
    C --> D[Source artifacts and model requests]
    D --> E[Receive generated text]
    E --> F[Local PII detection and agreed policy]
    F --> G[Protected generated text]
    G --> H[Generation artifacts and later model requests]
    H --> I[Dataset export]
~~~

Replacement and drop behavior are still undecided. The diagram describes where that policy applies, not which records it accepts.

The alternatives considered were:

| Option | Contract | Architectural consequence |
| --- | --- | --- |
| Ingestion and generation boundaries | Check source record text locally before its first persistence or provider use. Check generated text before its first persistence or later model use. | Integrate the same detection and redaction policy with the relevant source and generation paths. Account for internal calls and raw responses. |
| Final export | Check record text when producing the exported dataset. | Keep the change near export. Earlier artifacts and requests remain outside this contract. |
| Explicit pipeline stage | Protection begins where the configured stage runs. | Reuse stage artifacts and placement controls. A later stage cannot protect earlier writes or requests. |

The first option is selected. It covers source privacy and generated-output artifacts, including earlier destinations that a later pipeline filter could not protect.

A contract should concern specified text fields and detector-supported categories. It must distinguish a detector miss from a failure to apply redaction before a protected destination. The design cannot promise that a detector discovers all personal information.

Users choose supported languages and entity categories through configuration. This does not mean every arbitrary language or entity is supported. Validation must compare the requested coverage with the selected backend and installed model before the pipeline reads source text.

The [focused detector research](02-research-detector-options.md) compares operational fit. The recommended initial backends are built-in regex with validators for implemented identifier formats, and optional local GLiNER2 for contextual entities. Users explicitly choose a backend. Presidio and OpenAI Privacy Filter remain alternatives. The initial backend set is pending agreement.

Language configuration declares the coverage users require; it does not silently add a language-identification stage or prove every record belongs to those languages.

### Proposed behavior for review

Use one run-level PII policy for sources and generated text. When the feature is enabled, require an explicit backend and nonempty requested coverage. Keep it disabled for existing configurations that omit it.

Use category placeholders as the initial replacement policy. Users can change the placeholder template, which must include the category if category-specific tokens are requested. A matched-byte drop ratio and blocked-category list are configurable. Omit the ratio to retain records after redaction; 15% is an example, not an enforced universal default. A blocked-category hit takes precedence over the ratio. These defaults and policies are proposals until review.

Compute the ratio using the union of original matched UTF-8 bytes across the text fields in the unit being checked, divided by the original bytes in those fields. Count overlaps once and decide before replacement. Source records and individual generated candidates are separate units. An intermediate generated instruction is checked before its next model call.

Keep original source files outside this operation. For record text, cover document chunks and conversation instruction, response, system, reasoning and textual turn content where present. Record metadata and filenames require a declared separate scope. Diagnostics for this feature contain counts, categories and offsets rather than matched text.

For valid structured model output, operate on text values while preserving the schema and metadata. For malformed raw output that cannot become a record, redact before any diagnostic or raw-response persistence. Never save an unredacted copy for debugging. Keep provider usage information so rejection does not hide spend.

Detection or policy errors stop processing the affected boundary with a payload-free error. A missing optional package or model fails startup. Do not substitute another detector or download a model during a run.

Record the effective policy and model identity for resume. Reject reuse of incompatible prior artifacts. Redacted payloads need updated content hashes; source offsets must continue to identify the original source coordinates. Identity and lineage rules will be specified during program design.

### Component interaction

The policy owns supported-coverage validation, span handling, replacement and the keep/drop decision. Detectors supply typed character spans. Sources and provider/generation boundaries apply the policy before the approved destinations. Artifact writers receive protected text; they must not silently write a different payload than the one that later stages use.

Model packages belong in an optional extra. A GLiNER2 extra adds PyTorch and Transformers for local inference, so its inclusion needs the dependency approval required by AGENTS.md. No package has been added.

## Program design

The separate managed worktree starts at the revision above. It uses branch `c-b/pii-redaction` and contains no changes from PR #217.

The shared policy code is implemented in `src/arka/privacy/policy.py`. Detectors return immutable character spans. The policy merges overlapping spans, computes original matched UTF-8 bytes and replaces matched text. It checks nested string values while preserving dictionary keys and non-text values. The result includes category counts and an optional rejection reason. Detector errors contain no input text.

`src/arka/privacy/detectors.py` contains the built-in detector for email addresses, explicit international phone numbers, payment cards with a Luhn check, IBANs with a mod-97 check and IP addresses validated with Python's standard library. These patterns do not detect names or postal addresses. They do not establish whether a detected identifier belongs to a private person.

Source and provider integrations are still pending. The code does not yet protect an Arka pipeline run. The optional learned backend awaits the dependency decision in the question tool. Configuration models and checkpoint identities must be completed before this feature is ready for a PR.

PR #217 is currently open and changes several of these integration points. It is not part of the main revision studied here. Fetch main again before implementation and refresh the relevant source reading if that PR has merged.

## Decisions

Approved scope:

- Work on #166 first.
- Research and align on the design before implementation.
- Protect source record text locally before saving it or sending it to a model.
- Protect generated text before saving it or using it in later model calls.
- Let Arka users configure the required language and entity coverage.
- Finish the design and implement #166 before opening its PR.

Recommendations under discussion:

- Select a detector against the actual supported data and deployment needs.
- Keep the contract explicit about fields, categories and destinations.

## Open decisions

1. Initial selectable backends and optional learned-model dependencies.
2. Review of the proposed replacement, dropping, field coverage and error behavior above.
3. Resume compatibility, content identity and retained source offsets after redaction.
4. Review of the full system design before program design. Implementation and PR creation have already been requested; remaining design choices must be resolved first.

These are design topics, not implementation phases.

## Verification approach

After the scope is agreed, enumerate the failure cases before implementing logic, as required by the repository's TDD instructions.

Separate detection-quality evaluation from integration checks. Evaluate detection on agreed representative categories and languages. Verify that known detected values do not reach the protected artifacts or model requests. Include Unicode span handling and overlapping detections. The selected boundary requires coverage of raw responses and internal generation calls, as well as dropped records and error/resume paths that could retain the protected text.

The shared policy and built-in detector have 22 focused tests. Each test file failed before its implementation was added. `just check` passed with 565 tests passed and one skipped. `uv build` produced the source package and wheel. These checks cover the current foundation, not the pending pipeline integrations. No learned-model benchmark or paid request has run. No optional dependency has been added.

Failure cases to cover before writing implementation logic:

- A source record is persisted or sent to the PDF decision provider before protection.
- An intermediate generated instruction reaches the next generation request without protection.
- Valid and malformed generated responses enter raw-response files or drop details with original detected values.
- Unsupported requested coverage, an absent model or a detector exception allows processing to continue.
- Long-text inference silently truncates a suffix containing an entity.
- Identical repeated values are replaced using the first string occurrence instead of their actual offsets.
- Overlapping spans count the same bytes twice, or Unicode character offsets are mistaken for byte offsets.
- Repeated checking alters already inserted category placeholders.
- A blocked-category or ratio rejection keeps an unredacted debug copy or loses provider usage.
- A policy/model change resumes an incompatible checkpoint, or a redacted payload retains its old content hash.

## PR description

The user requested the show-me skill for this PR's description. Use a compact Mermaid diagram of the implemented protection boundaries and a concrete before-and-after fixture example. Add a small control-flow diff only if it clarifies the final change. State actual verification results and remaining limits. Write the description from the completed branch diff and checked artifacts.
