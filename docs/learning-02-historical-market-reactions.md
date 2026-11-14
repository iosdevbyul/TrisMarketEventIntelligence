# Learning-02: historical all-news to observed minute-price reactions

This is an **offline, read-only research join**, not an automated price
predictor. It reuses the existing `market-event-minute-source-audit`,
`study_intraday`, point-in-time event-ticker screening and explicitly
supplied exchange sessions. It does **not** add another market collector or
modify DonghakStockVision.

## Required real operator data

1. Export a Learning-01 `historical_news_cohort_v1` JSON page from the
   existing locally archived news DB. Every article in the page must retain
   the original source URL, first observed timestamp and UNKNOWN importance.
   Importantly, unclassified / apparently unimportant articles stay included.
2. Supply reviewed point-in-time ticker links in the existing 7-column
   `--links` CSV contract:
   `event_id,market,symbol,benchmark_symbol,source_url,evidence_seen_at,relationship`.
   Here `event_id` is the **original articleId**, not an economic_events ID.
   The link must match the archived article URL, and have been known by the
   article's first observed time. Otherwise it is blocked, not backfilled.
3. Supply an existing `minute_source_v1` hash-pinned manifest with stock
   and benchmark minute-price CSVs (including actual `observed_at` timestamps
   and `adjustment=adjusted`). Supply separately reviewed sessions/halts.
   `SCHEMA_COMPATIBLE_SOURCE_UNVERIFIED` does not mean the provider or
   corporate-action adjustment is independently authenticated.

Example usage, **only after those actual local files exist**:

```sh
bash scripts/verify.sh --update
market-event-historical-reactions \
  --cohort /path/to/reviewed-historical-news-cohort.json \
  --links /path/to/point-in-time-article-links.csv \
  --minute-manifest /path/to/verified-file-hashes-manifest.json \
  --sessions /path/to/reviewed-exchange-sessions.csv \
  --output /path/to/new-historical-reactions.json
```

Optional `--halts` supports a real halt CSV. The output is create-only and
never overwrites prior research. No source files or SQLite databases are
modified. Source URL/article title and market data can be licensed or
non-public: keep inputs and reports private.

## Exact interpretation of results

- The **available-at** instant for a news article is the first archived
  receipt time, not the publisher's unverified claimed publication time.
- Market bars whose `observed_at` exceeds the explicit historical cohort
  `asOf` cutoff are excluded; historical event study entry is the first
  strictly later completed minute close. Exit returns are measured over exact
  +1/+5/+15/+30/+60 elapsed-minute windows. Missing bars, halts and session
  boundaries remain missing; they are not compressed or guessed.
- Existing screening censors overlapping same-symbol news, and rejects
  duplicate article/symbol and incorrectly timed source links. The
  `marketReactionStatus=OBSERVED_RESEARCH_ONLY` status indicates one
  historical *candidate* excess-return observation, **not** verified
  causality, a positive/negative news importance label, calibrated confidence,
  or verified profitability.
- Articles with no valid link, no minute bars or no market reaction stay
  `UNMEASURED`. In particular, an ordinary/unimportant-looking article
  is **never** automatically assigned the negative training label.
- Source integrity via SHA-256 checks proves only that the local files
  match the explicitly supplied manifest. Real provider licensing,
  adjusted-price provenance, exchange calendars, issuer identity, and
  independently reviewed timestamps are **not** certified by the schema.
  Output always states `trainingDatasetEligible=false`.
- No real dataset, score, model, continuous retraining, or quality improvement
  is claimed merely because synthetic regression tests pass.

## Remaining work before genuine learning

- Run the Learning-01 cohort inventory against the user's actual Mac news
  archive, and measure complete pages/coverage and missing source days.
- Obtain separately authorized real KR/US minute stock/benchmark data from
  the market-data owner and independently review corporate adjustments,
  calendar/halts, exact `observed_at` and historical entity identities.
- Add explicit independent human materiality labels, including
  *non-material/quiet* examples; distinguish relative return motion from
  causal news impact and control for simultaneous news and other market
  factors.
- Apply time-purged rolling evaluation, independent out-of-period test sets
  and probability-calibration metrics before model training or auto-upgrade.
