# Learning-05: CPU news importance research iterations (foundation)

This is the **first opt-in research engine**, not a resident Mac service,
an autonomously improving production model, or a validated training success.
No launchd job is installed or started by these tools.

## What it actually does

Using only the existing immutable **news-only** historical cohort, Learning-02
self-attested human importance reviews, and Learning-04 time-split plan,
the engine:

1. Verifies the original cohort hashes, review integrity, review timestamp
   gate, holdout membership and all article IDs/counts.
2. Reads news text and human importance judgments from TRAIN **only if
   the review existed by the training cutoff**. Skips unknown, uncertain,
   late-labeled and quarantined TRAIN samples.
3. Trains a small CPU-only two-class multinomial Naive Bayes prototype in
   memory, using original article title and summary words. It cannot infer
   human labels; both classes need at least 3 TRAIN examples.
4. Evaluates importance *only on VALIDATION* human labels, with a minimum
   of 2 human-labeled examples per class. TEST labels are never selected
   into training or validation scoring.
5. Records only candidate algorithm/configuration fingerprints, cohort,
   plan and review hashes, sample counts, and descriptive validation log
   loss versus always-UNIMPORTANT baseline in **separate research SQLite**.
   Neither article text nor human gold is exported to the experiment ledger.
6. Selects the next unused fixed alpha value (0.25, 0.5, 1, 2, 4) on
   each explicit invocation. Restarting uses the existing SQLite journal;
   the same candidate is never rerun for the same immutable inputs.
   With unchanged data, after five candidates the status is
   `EXPERIMENT_GRID_EXHAUSTED`—not a busy self-learning loop.

If missing independent reviews or split coverage prevents a meaningful
experiment, status is `BLOCKED_INSUFFICIENT_HUMAN_GOLD`, not a
synthetic accuracy result. A new review file/cohort or split plan creates
a new candidate research identity; its provenance remains unverified.

A descriptive decrease in validation log loss **does not** establish a
generalization improvement, a calibrated model, or a profitable stock signal.
This prototype does not preserve a trained weight artifact, independently
authenticate reviewers, measure conditional direction/intensity/horizon,
use final TEST, or automatically promote/deploy a model.

## Mac workflow (only when genuine local news reviews exist)

Existing `scripts/news-learning-review.sh` has a new `research` action:

```sh
bash scripts/news-learning-review.sh --update research --once \
  --cohort /private/historical-news-cohort.json \
  --holdout /private/review-time-gated-news-holdout.json \
  --review /private/independently-reviewed-news.json \
  --journal "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-research.sqlite3"

bash scripts/news-learning-review.sh research --audit \
  --journal "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-research.sqlite3"
```

These are **examples only**. The journal must live in an existing private
directory, apart from production `news.sqlite3`. `--update` only works
on a clean main checkout. The command is single-iteration and opt-in:
do not add it to an existing collector launch agent yet.

## What still must be built before 24/7 operation

- Verified real human-gold coverage, stable source/corpus provenance,
  correction and cross-page near-duplicate controls.
- Candidate model weight persistence/versioning, a fair separate locked
  holdout process, and strictly validation-controlled acceptance policies.
- A resource-bounded stoppable macOS loop with restarts, backoff on
  insufficient data/exhausted search, and maintenance/health reporting.
- Separate news-pressure direction/intensity/horizon human-gold labels
  and evaluation before attempting those tasks.
- Independent approval before any shadow deployment or auto-promotion.

No stock prices, minute bars, OHLCV, volume, or realized returns may be
introduced to this module. Market forecasting remains DonghakStockVision's
sole responsibility.
