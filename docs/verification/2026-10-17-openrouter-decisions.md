# OpenRouter Decisions API verification

Arka passed local checks and a small live experiment on 2026-10-17.
The checked implementation is `33c198f`, on PR
[#216](https://github.com/jayshah5696/arka/pull/216).
The run used the configured OpenRouter key. No credentials are in this report
or the saved configs.

The requested model was `openai/gpt-6-luna-decisions`. The response identified
`openai/gpt-6-luna-decisions-20261006`. Requests went to
`https://openrouter.ai/api/alpha/decisions`, using the Jev question map described
in the [OpenRouter model documentation](https://openrouter.ai/openai/gpt-6-luna-decisions).
The direct OpenAI adapter uses a different wire format. This verification added
the OpenRouter adapter and provider-reported cost handling, with failing tests
written before the implementation.

## Live experiment

Six synthetic instruction and response pairs covered addition, multiplication,
and a capital city. Each question had one correct and one wrong answer.
Both runs used the same rubric and questions, with two workers. Filtering was
disabled for the comparison so every judge evaluated every record.

| Run | New HTTP requests | Input tokens | Elapsed seconds | Output records |
| --- | ---: | ---: | ---: | ---: |
| Shared panel and separate inverse critic | 12 | 7,002 | 2.273 | 6 |
| Separate judge calls | 30 | 7,596 | 3.426 | 6 |
| Filtering with cached panel answers | 0 | 0 | 0.052 | 3 |
| Resume of scored panel run | 0 | 0 | 0.016 | 6 |

The panel made 60% fewer requests. All 36 typed answers matched exactly across
the two runs, including full distributions and confidence values. Each inverse
critic used a separate request. All 42 comparison HTTP responses succeeded.
The request latency median was 208 ms and the measured p95 was 329 ms.
These timings come from one small run and include connection setup.

| Response | Expected | Quality, 1 to 5 | Reward, 0 to 9 | P(correct) | P(incorrect) |
| --- | --- | ---: | ---: | ---: | ---: |
| 2 + 2 = 4 | Correct | 4.524 | 8.07 | 1 | 0 |
| 2 + 2 = 5 | Wrong | 2.266 | 1.22 | 0 | 1 |
| France capital = Paris | Correct | 4.450 | 7.41 | 1 | 0 |
| France capital = London | Wrong | 2.120 | 1.07 | 0 | 1 |
| 7 × 8 = 56 | Correct | 4.698 | 8.60 | 1 | 0 |
| 7 × 8 = 63 | Wrong | 2.274 | 1.34 | 0 | 1 |

Filtering used a minimum rubric score of 3.5 and reward score of 6.
It kept all three correct answers and dropped all three wrong answers.
Cached answers supplied every filter. The instrumented pipeline runs fail
immediately if a stage calls `StageContext.llm_client`; no such calls occurred.
Resume also made no new HTTP requests. Its manifest retains the original
run's usage totals, so it still reports 12 historical requests.

A separate live probe checked yes/no, choice, and score answers together.
It classified a correct addition answer as correct, chose `math`, and returned
a score distribution with expected value 1.98 on a 0 to 2 scale.

The installed CLI also ran against OpenRouter with its production client
factory. A short Paris seed had P(usable) = 0.55. With an accept threshold of
0.60 and `on_uncertain: drop`, the seed gate dropped it. A longer water-cycle
example passed seed gating, English language detection, and identity checking,
then exported one record. That run made three requests. Both CLI runs resumed
from their stage checkpoints. Their LLM config pointed to an unused local port.

There were 47 live requests in total: one probe, 42 comparison requests, one
short-seed request, and three substantive-seed CLI requests. OpenRouter reported
`usage.cost: 0` for these runs. This is the reported charge for this experiment,
not a promise about future pricing. No further live requests were made.

## Local checks

| Check | Result |
| --- | --- |
| `just check` | Lint and formatting passed. 543 tests passed, one skipped. |
| `uv build` | Wheel and source distribution built. |
| `just validate-examples` | All nine examples passed. |
| `just dry-run-examples` | All nine examples passed. |
| `uv run python scripts/verify_decisions_cli.py` | Three records made nine loopback requests, selection exported two, and resume made zero extra requests. |
| Cleanup CLI helper | PDF and taxonomy exports passed, with 14 loopback calls and zero extra calls on resume. |

The skipped test needs an optional native fastText model, which is not present.
The build emitted an existing uv build-version warning. Pytest emitted existing
Pydantic deprecation warnings. Neither prevented the checks from passing.

## Reproduction and evidence

Set `OPENROUTER_API_KEY` in the environment, then use a new output directory:

```sh
uv run python scripts/verify_decisions_live.py --output /tmp/arka-decisions-live
```

This script makes paid requests. It limits execution to 48 HTTP requests,
uses one attempt per request, and saves synthetic request and response bodies
without headers. It runs the production pipeline and decision client with a
transport that forwards HTTP and records receipts. The separate CLI runs above
used the production factory without this instrumentation.

The [JSON summary](2026-10-17-openrouter-decisions.json) preserves the measured
results. Full local evidence remains at these paths:

- `/tmp/arka-openrouter-probe-grgxh3zo/receipts.json`
- `/tmp/arka-openrouter-live-20261017/result.json` and `receipts.json`
- `/tmp/arka-openrouter-live-20261017/panel/runs/` and `direct/runs/`
- `/tmp/arka-openrouter-cli-20261017/`, including CLI logs and Parquet artifacts
- `/tmp/arka-decisions-u9p0en63/`
- `/tmp/arka-cleanup-20261017-live-followup/`
- `/tmp/arka-decision-final-check.log` and `/tmp/arka-decision-final-build.log`

## PR and issue status

GitHub showed one open PR, #216. It was mergeable, with no reviews, comments,
or CI checks recorded. The earlier implementation PRs #184 through #191 are
merged. The three newer CLI PRs #182, #183, and #193 are merged. `origin/main`
was `5c251ef` and is already included in this branch.

All ten issues remain open. This PR covers core functionality across seven
decision-model issues. The six-example experiment does not replace their
representative dataset tests or the calibration-audit stage requested in #199.
Generation checks, semantic similarity, contamination, pairwise preference,
and taxonomy integrations have local automated coverage, but were not exercised
against the live provider in this experiment. Direct OpenAI, TypeSafe, and Clef
were tested through fixtures, not live credentials.
