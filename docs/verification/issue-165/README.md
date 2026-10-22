> These results evaluate the experimental ridge regression scorer, now named `distilled_ridge`. They do not evaluate the separate `distilled_quality` probability classifier.

# Real-data evaluation of issue 165

The CLI works with a real encoder and real judge APIs. The tested scorer did not
produce useful ranking agreement on this sample. Keep it experimental. These
results do not support using its predictions to reject records.

The [version 2 follow-up](../issue-165-v2/README.md) uses revised features and a
fresh holdout. This report retains the original representation's measurements.

## Data and method

The source was the public [OpenAssistant/oasst1 dataset](https://huggingface.co/datasets/OpenAssistant/oasst1).
It contains human quality ratings and conversation tree IDs. We selected 600
English, human-written, first-turn answers with at least two quality raters and
no positive PII annotation. Instruction-answer features had to fit within 240
tokens of the actual encoder, avoiding truncation. These filters yielded 5,731
eligible records. Selection used a fixed SHA-256 order, without stratifying by
human scores.

The 600 selected records came from 542 conversation trees. The scorer's existing
split assigned 478 records from 433 independent groups to training and 122
records from 109 groups to validation. We checked that tree IDs and identical
feature texts did not overlap between partitions. This is an internal split of
the public dataset's train partition, not its official validation partition.

The teacher was `google/gemini-3.1-flash-lite-preview` through Arka's existing
LLM judge and `rubrics/sft_quality.yaml`. The target was the weighted overall
quality score on the rubric's 1 to 5 scale. The encoder was the actual
`sentence-transformers/all-MiniLM-L6-v2` ONNX model, producing 384 features.
The ridge penalty stayed at the predeclared default of 1.0. We did not tune
settings against the holdout. Encoder file hashes and package versions are
recorded in [results.json](results.json).

The separate direct decision baseline used Arka's `DecisionClient`, the same
rubric dimensions, and `openai/gpt-6-luna-decisions`. The provider returned model
ID `openai/gpt-6-luna-decisions-20261006`.

## Quality results

Student rank agreement with the teacher was 0.113. The 95 percent interval was
negative 0.064 to positive 0.282. We used 1,000 bootstrap samples of the holdout
conversation trees. The interval includes zero, so this run does not establish
positive ranking agreement. The intervals describe holdout sampling uncertainty
for this fitted model. They do not include variation from retraining.

Student mean absolute error was 0.735 rubric points. Predicting the training
mean for every holdout record gave 0.764. Student root mean squared error was
0.864, compared with 0.843 for that constant baseline. The small reduction in
absolute error did not translate into useful ranking agreement.

Agreement with the existing human quality ratings was:

- Student Spearman correlation was 0.054, with a 95 percent interval of negative
  0.139 to positive 0.240.
- Direct LLM teacher correlation was 0.422, with an interval of 0.273 to 0.568.
- Direct decision correlation was 0.432 on the 121 holdout records it scored.
  One holdout record received no decision score. This value is not an exact
  comparison on all 122 records.

We defined a human-rated good record as one with a mean human quality rating of
at least 0.75. We fixed that threshold before judging. Selecting the top half by
student score retained 22 of 46 such records, or 47.8 percent. Selecting by the
teacher retained 27 of 46, or 58.7 percent. The decision baseline retained 29 of
45 on its scored subset, or 64.4 percent. These point estimates do not establish
that one direct judge is better than the other.

The student top half overlapped the teacher top half by 49.2 percent. Teacher
scores had ties; these selection comparisons break ties by original record
order. Human quality ratings are independent of the newly collected teacher
labels, but they are not proof of factual correctness.

## Error inspection

We inspected the four largest disagreements. The student assigned high scores
to responses that did not follow the instruction. For example, one record asked
for a dialogue and contained only the answer "Hi". The student predicted 4.49,
the teacher assigned 2.6, and human quality was 0.1. Another off-topic answer
received a student score of 4.43 despite a human quality score of zero.

The full holdout predictions, human ratings, decision scores where available,
and selected-record flags are in [holdout_scores.csv](holdout_scores.csv).
The report records the largest errors by original message and tree IDs without
copying the raw conversations into this repository.

## Runtime and actual CLI checks

We ran the installed `arka-train-scorer` command on the judged Parquet artifact.
The command embedded all 600 records, fit only the training partition, evaluated
the holdout, and saved a JSON model. It completed in 8.53 seconds, including
process startup. The encoder had already been downloaded.

We then ran the installed `arka` command on the 122 holdout instruction-answer
pairs. It loaded the saved model, annotated all records, and selected 61. We
checked that the actual exported records matched the computed top half. The
command completed in 2.86 seconds, including startup, model loading, artifacts,
selection, and reporting. The scoring stage itself took about 1.9 seconds.
Resume completed in 0.68 seconds and skipped all four completed stages.

The teacher collection took 113.75 seconds across the pilot, main batch, and
retry batch, using up to six concurrent requests. Its successful per-record
median time was 1.02 seconds. The decision baseline's successful per-record
median was 0.23 seconds. These timings cover different workloads and are not
a controlled speedup comparison. No full decision batch wall time is claimed
because the collection stopped on provider errors and was resumed.

## Cost and failed requests

The LLM teacher made 604 HTTP requests to obtain 600 complete labels. Four
initial responses omitted required rubric scores; one retry each succeeded.
All 604 returned HTTP 200, including the four incomplete labels. No dummy
labels were substituted. The collection used 244,372 prompt tokens and 31,998
completion tokens. Provider usage reported USD 0.10909 in upstream inference
cost. OpenRouter's own charge field was zero because it identified the requests
as BYOK. Zero in that field does not mean the inference was free.

The decision backend returned scores for 597 of 600 records. It refused three
harmful-content records with HTTP 502. One was in the holdout. These records
remain failures in the report. The collection, its retries, one diagnostic call,
and the two model probes made 611 HTTP attempts. The successful scores used
333,996 input tokens. The API reported a zero charge, but did not provide the
upstream cost. A full cost comparison with the decision backend is therefore
incomplete. See [collection_failures.json](collection_failures.json).

Local embedding and CPU time are measured as elapsed time, not assigned a
dollar cost. The total teacher acquisition count includes evaluation labels.
This experiment does not demonstrate an overall judge-cost reduction.

## Reproduce and inspect

The collection and analysis script is [evaluate.py](evaluate.py). It uses the
project's installed dependencies. It reads `OPENROUTER_API_KEY` from the
environment and makes real paid API calls. It stores public conversations and
raw responses under `/tmp/arka-165-real-eval`, outside the repository. Set
`ARKA_EVAL_DIR` to a fresh directory for an independent run. Run from the
repository root or set `ARKA_EVAL_REPO` explicitly.

```sh
uv run python docs/verification/issue-165/evaluate.py prepare
uv run python docs/verification/issue-165/evaluate.py pilot
uv run python docs/verification/issue-165/evaluate.py judge
uv run python docs/verification/issue-165/evaluate.py decision
uv run python docs/verification/issue-165/evaluate.py decision-judge
uv run python docs/verification/issue-165/evaluate.py fit
uv run python docs/verification/issue-165/evaluate.py analyze
```

Collection persists completed labels. If collection fails, inspect the error
before rerunning. A rerun requests only records without saved labels. The
decision collection can leave partial results when the provider refuses a
record. Analysis reports decision coverage rather than inventing missing scores.
The saved `collection_failures.json` documents this run's extra diagnostic
requests, which were made outside the script. A fresh run must record its own
diagnostics rather than reuse those counts.

The original model is `/tmp/arka-165-real-eval/model.json`. Its SHA-256 is
`e60e490562ab572dcc8d5dfb7483f1abd5b7cb2d52e62b7e37bda883fb801376`.
Training, inference, and resume logs are in that directory. Dataset byte hashes
and model hashes are saved in the results. Future provider output can differ
even with the same configuration.

These findings apply to this encoder, penalty, sample, and teacher. They do not
rule out distillation with another representation or more labels. Any further
model selection should use training data and a fresh final holdout. This sample
does not establish quality for longer conversations, other languages, or the
user's own production data.
