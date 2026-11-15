# Learning-05 — TRAIN-only research expansion on unchanged historical NEWS

This change returns to the original goal: use the Mac's existing historical
NEWS corpus to investigate **new, distinguishable modeling hypotheses even
when no new article has arrived**. It extends research beyond the original
five-alpha grid, but does not claim that keeping a Mac running indefinitely
will guarantee continual improvements.

## Actual research work

The existing `_validate_inputs` enforces pinned complete-news cohort,
human-only review provenance, time-gated TRAIN labels, and the original
TRAIN / VALIDATION / embargo / TEST / quarantine partitions.

A new **separate** journaled exploratory family performs stratified
three-fold (or fewer, if needed) evaluation **inside TRAIN only**.
Each new experiment actually trains a CPU multinomial Naive Bayes model
several times on different TRAIN folds and scores its held-out TRAIN
folds. Experiments vary:

- Source text features: unigram or unigram + adjacent token bigram
- Token cap per news document: 32 / 128 / 500
- Additive smoothing alpha: 0.25 / 1.0 / 4.0

The resulting **18 distinct configurations** each run a finite,
reproducible train-only CV experiment with a fresh SHA-256 research
identity, and metadata in `news_train_cv_experiments` inside the existing
**private research journal**. Restarting skips completed fingerprints.
It never touches the legacy `news_research_experiments` records, the
original news archive or production model. Running another identical
cycle after all 18 configurations are exhausted correctly reports
`TRAIN_CV_SEARCH_SPACE_EXHAUSTED`, not an invented learning result.
If either TRAIN class has fewer than three genuine by-cutoff human labels,
it reports `BLOCKED_INSUFFICIENT_TRAIN_HUMAN_GOLD`.

## Important protection against validation overfitting

The CV scores use **TRAIN labels only**. The protected VALIDATION set is
not scored, compared, ranked, or fed into candidate selection. TEST gold
is never scored or used for fitting. The original holdout validation file
and its provenance are still checked so the source boundaries remain
intact. Experiments on the same TRAIN cases are **correlated**:
18 lower CV losses are not 18 independent pieces of evidence. They must
not be used to certify that model quality improved, and this PR does not
create production model artifacts, push model weights, or auto-promote.

A fair independent assessment will require a policy for selecting exactly
one candidate from TRAIN-only results, strong original news provenance,
independently judged importance labels and a *locked* final evaluation
process. A larger, hypothesis-driven research generator and explicit
overfitting budget are also still necessary before truly open-ended 24/7
self-research.

## Optional Mac use (only with genuine private news review)

```sh
bash scripts/news-learning-review.sh --update train-cv --batch 18 \
  --cohort /private/news-cohort.json \
  --holdout /private/time-gated-news-holdout.json \
  --review /private/independently-reviewed-news.json \
  --journal "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-research.sqlite3"

bash scripts/news-learning-review.sh train-cv --audit \
  --journal "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-research.sqlite3"
```

These commands are **examples**, not instructions to run immediately.
No launchd agent is installed or started by this PR. CI tests use
synthetic news labels only and cannot verify genuine improvement.

No stock prices, minute/daily candles, trading volume, returns or
DonghakStockVision data enter this research engine. News-only research
remains strictly separate from actual market-price forecasting.
