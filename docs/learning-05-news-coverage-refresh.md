# Learning-05 — safely refresh actual private NEWS coverage

Learning-05 is NEWS ONLY. Manual human labeling is paused. The existing
RSS collector and its SQLite archive remain the sole source of news
observations; this helper neither collects new articles nor invokes AI.

## One command when you want to check for additional news

```sh
bash scripts/news-learning-review.sh --update coverage-refresh \
  --baseline "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-cohort-20261115.json" \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-cohort-refreshed-20261115.json"
```

The command defaults to the **current UTC time** and reads existing
`news.sqlite3` without writing to it. The complete current cohort is
checked against the original 482-news pinned snapshot using the
existing `coverage` implementation. If no newly observed articles
exist, it reports `NO_NEW_NEWS_SNAPSHOT_NOT_CREATED`, makes no new
cohort and retains the same requested output filename for a later run.

If there is at least one new article, and all original article IDs,
titles, summaries, source URLs, captures and first-observed timestamps
still match, the complete newer cohort is created as a **new private
file**, 0600 permissions, without replacing anything. That file can
be used as the baseline for future checks.

The refresh rejects:
- invalid or changed original records, missing previous articles,
  backdated newly inserted observations, or a cutoff not later than
  the earlier full snapshot;
- archives exceeding the 5000-article snapshot bound (no partial
  or quietly truncated learning dataset);
- output paths that already exist or are symlinks.

If no new news has been collected, no snapshot is generated and no
model is trained. Temporary internal data are deleted after validation.

The metadata summary reports new article count, additional observed
UTC days and remaining duration/source concentration warnings.
Satisfying thresholds is NOT an audited independent event sample,
a TRAIN/VALIDATION/TEST-ready dataset, or proof that AI inference
improved. No human gold is fabricated; TEST is not scored; no model,
stock price, OHLCV, volumes or returns are used.

For deterministic QA, optional `--as-of` accepts a timezone-aware ISO
timestamp. This is a **manual, one-shot diagnostic**: no background
schedule or new launchd service is installed.
