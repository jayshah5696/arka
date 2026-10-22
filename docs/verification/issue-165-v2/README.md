> These results evaluate the experimental ridge regression scorer, now named `distilled_ridge`. They do not evaluate the separate `distilled_quality` probability classifier.

# Revised scorer and fresh holdout

The revised scorer has higher point estimates than the original representation.
It did not pass the quality check fixed before evaluation. Keep issue 165 open
and keep the scorer experimental.

## What changed

Version 2 uses 12 features: cosine similarity between separate instruction and
answer embeddings, plus 11 measurements of length, word overlap, and response
structure. The ridge head standardizes features using training records only.
The artifact stores feature names, means, and scales. Version 1 models still
load with their original combined-text embedding representation.

Development comparisons on the original 600 records considered combined and
response embeddings, pair interactions, text measurements, and a small relevance
model. The relevance model added negligible benefit and was omitted. No Python
dependencies or runtime models were added. The original holdout was retired and
treated as development data for this round.

We selected 1,800 additional development records and reserved 600 fresh records
before choosing the final model. The new holdout contains 288 conversation
trees. Development and holdout tree IDs and exact feature texts do not overlap.
Eligibility stayed the same as the [first evaluation](../issue-165/README.md):
short English human-written first-turn answers from OpenAssistant, with at
least two human quality ratings and no positive PII annotation.

## Model selection and label coverage

Of 2,400 selected development records, 2,398 had valid teacher scores after one
retry for incomplete results. Two unresolved failures were excluded from model
fitting and remain identified in the saved results. No labels were fabricated.

Five development folds grouped by conversation tree selected alpha 100 from
0.01, 0.1, 1, 10, 100, and 1,000. Out-of-fold teacher Spearman was 0.302.
Both final artifacts then fit the same 1,920 records. Their other 478 records
were development validation data. Because development data informed model
selection, that internal validation is not the final quality estimate.

The chosen feature format, penalty, model hashes, and holdout identity hash
were locked before collecting fresh holdout teacher labels. Of 600 reserved
records, 599 received valid scores after one retry. That one failure remains
explicitly reported. Neither model was changed after observing these results.

## Fresh quality results

All following student comparisons use the same 599 records. The original
representation was retrained on the same fitting records, so the comparison
does not mix the first evaluation's sample with this sample.

- Teacher Spearman increased from 0.204 for the original representation to
  0.297 for version 2. Version 2's interval was 0.214 to 0.378.
- Human quality Spearman increased from 0.208 to 0.338. Version 2's interval
  was 0.264 to 0.416. The direct teacher's correlation with humans was 0.380.
- Mean absolute error decreased from 0.718 to 0.690 on the rubric's 1 to 5
  scale. Root mean squared error decreased from 0.859 to 0.811.
- Keeping the top half retained 61.9 percent of human-rated good records,
  compared with 57.8 percent for the original representation and 65.2 percent
  for the direct teacher. Good means a human quality rating of at least 0.75.

The bootstrap resampled conversation trees 1,000 times with seed 1652. The
paired interval for improvement in teacher Spearman was negative 0.021 to
positive 0.206. It includes zero. We cannot claim that the primary improvement
over the original representation is established by this run.

The predeclared check required a positive lower bound on that paired interval
and a student good-record recall no more than five percentage points below the
teacher's point estimate. The recall difference was negative 3.28 percentage
points, but the primary interval condition failed. The combined check failed.
The interval for the recall difference was negative 9.09 to positive 2.74
percentage points, so this is not a demonstrated noninferiority result either.

The separate decision baseline scored 597 of the 600 reserved records. Its
comparison with the LLM-labeled holdout covered 596 records. On that common
subset, human Spearman was 0.296 for the decision baseline, 0.332 for version 2,
and 0.376 for the LLM teacher. These point estimates do not establish that the
student is better than the decision judge. Human quality ratings also do not
establish independent factual correctness.

See [results.json](results.json) for intervals and coverage and
[holdout_scores.csv](holdout_scores.csv) for all 599 teacher and student scores.

## Actual CLI path and cost

We used the installed training command with `--features pair_quality_v2
--alpha 100`. Training took 38.52 seconds, including embeddings for all 2,398
valid development records and process startup. The original representation's
training took 33.59 seconds.

The installed pipeline scored 599 holdout records and exported the highest
299. We checked the actual exported payloads against the computed selections
for both models. Version 2 inference took 9.86 seconds, compared with 9.54
seconds for the original representation. Version 2 resume took 0.85 seconds and
skipped the completed stages. These measurements used the real MiniLM encoder;
they contain no substituted embeddings or judge responses.

New development labeling made 1,810 requests and reused the original 600 labels.
Fresh holdout labeling made 603 requests. Known upstream inference charges were
USD 0.327595 for development and USD 0.10910225 for the holdout. Four requests
across these collections lacked upstream cost information, so USD 0.43669725
is a known subtotal, not a complete total. The reused labels had their own
acquisition cost in the first evaluation.

The direct decision collection made 600 attempts, returned 597 scores, and took
34.26 seconds. Its upstream cost was unavailable. The fresh LLM collection took
100.57 seconds including retries, using up to six concurrent requests. Student
inference used no judge requests, but this evaluation itself did not save judge
calls because we acquired training and evaluation labels. No total-cost savings
claim is established. CPU time has not been assigned a dollar cost.

## Reproduce

The raw source data and first 600 judged records are the prerequisites from the
first evaluation. [collect.py](collect.py) reserves groups and collects new
labels through the existing judge clients. [evaluate.py](evaluate.py) performs
development cross-validation, freezes model choices, runs the installed CLIs,
and computes results. They use installed project dependencies and read the
existing API key from the environment. Collection makes paid API requests.

Run from the repository root. Use a fresh `ARKA_V2_EVAL_DIR` for an independent
run. `ARKA_BASE_EVAL_DIR` locates the first evaluation's raw artifacts. Preparation
refuses to overwrite an existing protocol. Inspect collection failures before
making an explicit retry; a retry requests only records without saved labels.

```sh
uv run python docs/verification/issue-165-v2/collect.py prepare
uv run python docs/verification/issue-165-v2/collect.py development
uv run python docs/verification/issue-165-v2/evaluate.py develop
uv run python docs/verification/issue-165-v2/collect.py final
uv run python docs/verification/issue-165-v2/collect.py decision
uv run python docs/verification/issue-165-v2/evaluate.py infer
uv run python docs/verification/issue-165-v2/evaluate.py analyze
```

This run's full protocol, labels, trained models, and CLI logs are under
`/tmp/arka-165-v2/`. Saved results replace the long selected-record ID lists with
their SHA-256 hashes; the full lists remain in the raw protocol. The saved model
hashes are in the locked choice metadata. The report can be recomputed without
additional API calls using the `analyze` operation. Future provider output may
differ. Further model selection must use development data and another untouched
holdout, rather than tuning on this one.
