# Learning-04: news-only chronological partitions and leakage guard

This is a conservative, **read-only research partition planner**, not a
certified protected holdout or a model-training worker. It consumes only
the existing `historical_news_cohort_v1/v2` and
`news_human_importance_review_v1` artifacts. There is no market-price,
OHLCV, volume, return, stock prediction, market reaction, or model inference.

## What the partition planner does

- Validates the full cohort's immutable article hashes, first-observed times,
  all-news inclusion, and the human-review-to-cohort fingerprint using the
  existing Learning-01 and Learning-02 validation functions.
- Partitions by **first observed receipt in UTC**, with an explicit
  train-end and validation-end. An embargo around either boundary is
  `QUARANTINED`, not silently used to increase sample size.
- Uses transitive union of **existing storyGroupId, identical news text SHA**
  and **canonicalized original source URL** across the entire cohort page.
  If members of one group fall into different partitions or embargo windows,
  **all** group members are quarantined. Cross-group near duplicates not
  detectable through these identifiers remain a documented research gap.
- For training articles, distinguishes independent human importance reviews
  entered before the train cutoff versus reviews written afterward.
  No label value or human review reason is exported to the partition plan;
  validation/test labels remain held out.
- Leaves `trainingDatasetEligible=false`. Human adjudicator identity,
  prior model exposure to holdout labels, cross-page duplicate groups,
  corrected-article lineage and repeated holdout tuning have NOT been
  independently checked. The tool makes **no accuracy or improvement claim**.

## Mac usage, only when real-data testing is essential

The existing script now supports `split`. This code has synthetic CI tests;
you do not need to run it manually for normal PR development.

```sh
bash scripts/news-learning-review.sh --update split \
  --cohort /private/cohort-v2.json \
  --review /private/human-reviewed.json \
  --train-end 2026-09-01T00:00:00+09:00 \
  --validation-end 2026-10-01T00:00:00+09:00 \
  --embargo-hours 24 \
  --output /private/new-news-partitions.json
```

The dates are **examples**, not claimed audited dates. Output is create-only;
the original news SQLite DB and review JSON remain unchanged. This script
does not start any daemon, fetch articles or data, publish private notes,
or trigger automatic learning.

## Next safety gates before autonomous improvement

An independently locked evaluation/test set, human-audited group linkage
across pages, revision/near-duplicate detection, an experiment ledger that
prevents repeated holdout tuning, and a human source review corpus are
necessary before any promotion of a new candidate model. NEWS-only model
direction/intensity/horizon quality requires separately labeled news
hypotheses; none of those can be inferred from stock movement.
