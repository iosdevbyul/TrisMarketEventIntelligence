# Learning-02: human news importance and source-alignment review

This workflow trains **nothing** and never uses price, OHLCV, returns, or
market movements. It provides an independently **human-entered** set of
importance and evidence labels for the all-news cohort, retaining apparently
unimportant news and unclassified articles.

## Local Mac workflow (run only when human review becomes necessary)

Use the new script, which reuses the working Python environment automatically.
`--update` is optional and only works on clean `main`.

```sh
bash scripts/news-learning-review.sh --update cohort \
  --as-of 2026-11-15T23:59:59+09:00 \
  --limit 1000 \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/cohort-v2.json"

bash scripts/news-learning-review.sh prepare \
  --cohort "$HOME/Library/Application Support/TrisMarketEventIntelligence/cohort-v2.json" \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/review-unlabelled.json"
```

The output `review-unlabelled.json` starts with **every** judgment blank.
A person must read the original story, consider source context and then fill
only their genuinely reviewed items, setting:

- `importance`: `IMPORTANT`, `UNIMPORTANT`, or `UNCERTAIN`.
  `UNIMPORTANT` means human news significance assessment, **not** a
  claim about zero stock-price movement.
- `sourceAlignment`: `SUPPORTED`, `CONTRADICTED`, `UNRELATED`, or
  `INSUFFICIENT_CONTEXT`. Compare source text and the attributed
  story/claim; this is not proof of real-world factuality.
- `reviewer`: human-provided attribution; `reviewedAt`: timezone-aware
  ISO 8601 time after the first observed article time; `reason`: at least
  20 characters of meaningful review reasoning.

Unreviewed rows should remain entirely `null`. Do not generate a fictional
human label. Human reviewer identity is *self-attested*, not authenticated.
For a completed file:

```sh
bash scripts/news-learning-review.sh score \
  --cohort "$HOME/Library/Application Support/TrisMarketEventIntelligence/cohort-v2.json" \
  --score /path/to/privately-human-reviewed.json \
  --output /path/to/new-human-review-summary.json
```

Every output is **create-only**, never overwritten. Each labeled row is
pinned to the original archived article URL, capture hash, article text hash,
first observed time, optional existing story group, and complete cohort
SHA-256. The scorer rejects missing/duplicate articles, invalid timestamps,
source tampering, unsupported labels or incomplete partial judgments. It
reports importance counts by source, human verdict distribution, coverage,
missing labels and grouping diagnostics. A missing story group never
establishes uniqueness; human judgments on multiple articles in the same
story group are not independent.

No private archive, source URL, human note, copyrighted article text,
credentials or report should be uploaded to public CI.

## Status versus future autonomous training

`trainingDatasetEligible=false` **always**: human-entered values need
independent sampling, adjudicator agreement and source audits. This CLI
cannot show calibrated model quality or independently authenticate source
truth. New AI candidates, repeated news-only training, time-purged
validation and evidence-alignment metrics belong in later Learning phases.

Actual Mac news availability and real human labels are **not measured
by synthetic CI tests**. Never claim real training improvements from this
work alone.
