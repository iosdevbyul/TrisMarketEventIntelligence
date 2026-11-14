# Learning-04 — Cross-page news holdout integrity and duplicate audit

The existing Learning-01 cohort exporter produces bounded 1–5000 item pages.
A valid single-page Learning-04 split cannot reveal whether a matching news
story appears in another page's TRAIN, VALIDATION or reserved TEST period.

The cross-page audit uses **2–16 ordered, complete, contiguous page pairs**
(each original all-news cohort JSON plus its separately generated
`news_holdout_plan_v1` JSON). It checks that all page cursors connect, the
last page is terminal, the archive and as-of cutoff match, and that every
holdout is pinned to its exact cohort, has valid source text hashes,
first-observed times, counts and a consistent time-splitting policy.

Then it suggests candidates across *different pages AND partitions* using
existing canonical source URLs, exact text SHA and story group, plus the
existing conservative headline-token overlap method. All candidates remain
`HUMAN_REVIEW_REQUIRED` with `confirmedSameStory=null`. No original
split is modified, no gold labels are loaded, and no market or stock data
is used.

## Running on your Mac, when needed

The existing script has a `cross-page` action. Use files generated with
the same snapshot cutoff; **do not run this example until real full pages
and their original plans are prepared**:

```sh
bash scripts/news-learning-review.sh --update cross-page \
  --cohort /private/news-page-1.json --holdout /private/holdout-page-1.json \
  --cohort /private/news-page-2.json --holdout /private/holdout-page-2.json \
  --output /private/new-cross-page-review.json
```

Additional paired `--cohort` and `--holdout` arguments are supported.
By default headlines must have at least five shared relevant tokens and
0.78 Jaccard overlap within 30 days. A maximum of 20,000 archived news
records and 10,000 candidate pairs is accepted per bounded audit; an
overflow aborts without writing a misleading partial report.

This is only a **potential cross-page contamination audit**. Exact-match
grouping is not necessarily proof of editorial identity; unrelated
headlines can be similar and semantically identical Korean/English
translations may be missed. Semantic duplicates and correction lineage
still need human review. No candidates does **not** certify the news
holdout. Rechecking separate batches of pages does not substitute for an
independent, complete corpus-wide audit.

No production SQLite, existing news source, trained model, data scheduler,
DonghakStockVision or market-price input is changed. Real Mac runs and
human adjudications remain unverified until actually performed.
