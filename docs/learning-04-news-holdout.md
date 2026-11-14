# Learning-04: news-only temporal holdout and duplicate leakage gate

This is a read-only planning tool for future **news-only** training and
evaluation. It never uses price, OHLCV, volume, realized returns, market
causality or DonghakStockVision. It never trains a model or certifies a
historical news classification dataset.

## Why this gate exists

An apparently impressive score can simply be memorization when an earlier
story appears in training but the same report, updated version or copied
article appears in a later validation/test set. A time-only split is not
enough. A fixed test set must be reserved, not tuned against repeatedly.

The planner uses the existing Learning-01 archived news cohort and:
- Orders by **firstObservedAt**, not an unauthenticated publication date.
- Splits into TRAIN up to `--train-end`, a 24-hour default embargo,
  VALIDATION up to `--validation-end`, another embargo and reserved TEST.
- Reuses `story_index.canonical_url`, existing `storyGroupId` and the
  exact archived article text SHA-256 to group duplicate candidates,
  including cross-source exact text copies and URLs with tracking tags.
- **Quarantines all members** of a candidate duplicate group spanning
  different splits or touching the embargo boundary. An article cannot
  silently leak from training to validation/test through a later retelling.
- Emits article ID, source hash, observed time, partition, exclusion reason
  and counts only. It does not embed human gold labels or inferred
  important/unimportant or model confidence values.

## Mac invocation (only when real-data validation is needed)

The existing shell script includes the action:

```sh
bash scripts/news-learning-review.sh --update holdout \
  --cohort /private/historical-news-cohort-v2.json \
  --train-end 2026-09-01T00:00:00+09:00 \
  --validation-end 2026-10-01T00:00:00+09:00 \
  --embargo-hours 24 \
  --output /private/holdout-plan-new.json
```

The sample timestamps are **illustrative**; they do not represent actual
historical coverage or an approved model experiment. Output is create-only.
Only a clean main checkout can use optional `--update`.

## Important limits

This is a preliminary plan, not complete leakage prevention. Independent
semantic near-duplicates, translated reprints, articles with missing
story grouping, and retrospective human decisions require a separate
audit. Text hashes and grouping keys detect exact duplicates, not meaning
equivalence. `train_end` and `validation_end` are explicit UTC-normalized
cutoffs; if archives lack enough dates the partitions may be empty, but
the planner never invents articles or scores. Future candidate selection
must not repeatedly inspect or optimize against reserved TEST.

No source DB, journal, runtime launch agent or published signal is changed.
The output keeps `trainingDatasetEligible=false` and
`modelPerformanceVerified=false` until independently checked news gold,
train/test lineage and external human review exist.
