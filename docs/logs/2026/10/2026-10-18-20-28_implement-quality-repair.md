# Implement quality repair

The user approved targeted answer repair for issue 173 and requested a pull request.

## Progress updates

I’ll implement targeted answer repair with one attempt per eligible record, then verify the CLI workflow and open a PR. I’ll use the existing PII protection and keep the pipeline linear.

I’ll make repair eligibility explicit in the config: minimum scores for rubric dimensions, which failed dimensions allow repair, and a maximum number of repair attempts. That lets users exclude failures such as an ambiguous instruction.

I’m writing the failure cases first. They cover interrupted attempts, missing PDF evidence, stale scores, PII rejection, and uncertain decision judgments. I’m also applying the Arka verification and PR description skills.

The 21 focused tests pass. The stage preserves passing records, links replacements to their originals, clears response-dependent scores, and resumes after a persisted generation without generating it again.

Next I’ll exercise the installed CLI, including downstream deduplication and PII protection. I’m also checking how repair interacts with decision panels, since grounded repair needs evidence that ordinary panel requests do not include.

The first CLI run passed all seven repair scenarios. It confirmed that downstream deduplication removes a replacement that duplicates a passing record, PDF repair uses the source chunk, PII-rejected output gets no second judgment, and interrupted repair reuses its saved generation.

The full test suite also passed: 662 tests passed and one optional native-language test skipped. I’ll add the repair recipe to `just verify`, then review the diff and prepare the PR.

`just verify` passed with the new repair recipe included, and the package build passed. The final suite reports 668 passed and one optional skip. All verification used local fixtures.

I’ve inspected the exported records, drop reasons, usage totals, and resume artifacts. I’m preparing two commits so the product change and verification workflow are easy to review separately, then I’ll push the branch and create the PR.

## Final response

Created [PR #220: Add bounded answer repair for failed quality checks](https://github.com/jayshah5696/arka/pull/220).

The new `quality_repair` stage:

- Judges answers against explicit rubric thresholds and attempts one repair for eligible failures, within a configured limit.
- Preserves instructions, links replacements to originals, and clears stale response scores and reasoning.
- Uses permitted PDF evidence and the existing PII protection.
- Saves judgments and generations so resume can reuse completed work.

Validation passed:

- `just check`: 668 passed, one optional native-language test skipped.
- `uv build`: source archive and wheel built.
- `just verify`: all default workflows passed, including seven repair scenarios with 27 local HTTP requests.

The CLI proof covers downstream deduplication, PII rejection, changed-input rejection and interrupted resume without repeating saved generation. Representative answer-quality gains and production cost savings remain unmeasured.

Configuration and limitations are documented in [docs/quality-repair.md](../../../../quality-repair.md).
