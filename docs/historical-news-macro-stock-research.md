# Independent historic price / NEWS / economic-indicator research (Phase 1–2)

## Goal and existing code

TrisMarketEventIntelligence independently researches past market moves with
NEWS **and macroeconomic indicators**. It does not depend on another repository.
Reuse the existing `event_study`, `intraday_event_study`, source-pinned NEWS
cohort, LLM/Research worker, and local SQLite when subsequent validated
sources become available. This PR adds a **standalone, opt-in, offline data
boundary** for stock-versus-benchmark shock discovery and point-in-time
economic context; it does *not* run Gemma, change the deployed pipeline,
fetch external material without permission, or assert ten-year completeness.

## Required supplied inputs

- A target stock and a **distinct** market/sector benchmark, each with
  daily prices from a provider the operator is authorized to use, ideally
  covering 10 years. Two separate, UTF-8 CSVs must each have exactly
  `date,adjusted_close,volume`, with ascending unique YYYY-MM-DD session dates,
  positive finite **split/dividend-adjusted** close and non-negative integer
  volume. The contract validates shape, **not** independent source authenticity
  or whether the provider actually applied corporate actions.
- An existing source-hash verified historical NEWS cohort in the schema already
  used by `historical_news_cohort`. Every article contributes only when its
  `firstObservedAt` actually precedes the selected event cutoff. A headline
  whose historic *publication* date was inferred from a source but whose first
  observed timestamp is in 2026 is **not** legitimate 2016 PIT news.
  Existing archived 482 articles span only a few recent observation days and
  **cannot** provide 10 years' historical NEWS by themselves.
- A separate economic-indicator vintage CSV (may have header with zero rows):
  `series_id,observation_date,vintage_date,release_at,available_at,value,source`.
  Timestamps must be **explicit timezone-aware ISO 8601**. Here `release_at`
  is the verified public announcement timestamp, `available_at` is the
  earliest observed ingest timestamp, not a fictional retroactive lookup time.
  Revisions use their **own vintage_date and release_at**. No guessed release
  clock or revised latest value is allowed as if available historically.
  Import may be based on FRED/ALFRED vintages but ALFRED
  `realtime_start` alone is date-granular and does **not** prove the minute of
  public release.

Official details:
[FRED series observations](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)
and [FRED real-time periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html).

Economic series to prioritize: CPI inflation releases, unemployment/nonfarm
payrolls, Fed interest rate decisions, PCE, GDP, Treasury yields and
industrial production. Some series are daily market observations rather
than scheduled announcements; these require their own actual public
availability rules and should **not** be mislabeled as a scheduled release.
Cross-country series should preserve local release time and currency/context.

## Selection logic

- Calculate target close-to-close *stock return minus aligned benchmark return*,
  rank by absolute excess daily return and select TOP `--top 100`.
- Select `--controls 100` distinct small absolute-excess-return dates.
  These are **unmatched preliminary controls**; they are not yet balanced on
  regime, volatility, seasonal calendar or macro event exposure.
- Each event has market-open (09:30 America/New_York) cutoff; context separates
  first-observed pre-open NEWS from post-open NEWS. Per-source macro **latest
  vintage available at cutoff** is chosen without using later revisions.
- Stock/benchmark adjusted-close alignment is explicit, missing benchmark days
  are excluded rather than forward-filled.
- Output is an **unverified retrospective research cohort**: selection knows
  the same day's actual return and cannot count as a prospective prediction.
  It also cannot prove that a frequently mentioned news article **caused** a
  price move. Deduplication, sector controls, company-entity links and
  simultaneous-market-confounder review remain necessary.
- A historical record whose first observation is missing cannot be
  retroactively labeled known at event time. Any later-known statement,
  current-revised CPI, or later-posted article is isolated from `asOf` features.

## Run (no new model calls, no automatic downloads)

Inspect data without writing:

```sh
bash scripts/news-learning-review.sh --update historical-shocks --preview \
  --symbol NVDA --benchmark-symbol SPY \
  --stock-csv "$HOME/Library/Application Support/TrisMarketEventIntelligence/private-history/nvda-daily.csv" \
  --benchmark-csv "$HOME/Library/Application Support/TrisMarketEventIntelligence/private-history/spy-daily.csv" \
  --news-cohort "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-cohort-20261115.json" \
  --macro-csv "$HOME/Library/Application Support/TrisMarketEventIntelligence/private-history/macro-releases.csv" \
  --top 100 --controls 100
```

Only after obtaining real authorized source files, to create a private
0600 artifact, use the same arguments without `--preview` plus
`--output "$HOME/Library/Application Support/TrisMarketEventIntelligence/private-history/shock-study.json"`.
The file must not already exist. This is not yet a trained model.

## Follow-up to make genuine 10-year learning work

1. Verify authorized historical adjusted-price and benchmark availability,
   provenance, corporate actions, and time coverage on the user's Mac.
2. Verify historical as-published company and macro NEWS collection,
   original first-seen/available timestamps, feed archives, and source licences.
3. Source true ALFRED vintage historical values **and actual release
   timestamps**; check embargoes, major revisions and event timezone.
4. Build sector/market/low-move *matched* controls, entity matching and
   multi-news confounder clusters, and expose two different study modes:
   backward explanation (may know the outcome) versus forward replay
   (must seal predictions before future prices are revealed).
5. Invoke existing injected Gemma provider for bounded source-linked
   retrospective explanations **only** after source coverage is adequate;
   store experiment memory and use a frozen forward-replay holdout,
   not self-graded explanations, for progress.
6. Configure and validate resource-bounded crash-restartable Mac worker,
   with explicit operator permission before 24/7 activation.

Do not claim actual 10-year dataset completeness, training readiness,
verified forecast accuracy, identified causality, autonomous operation, or
model promotion from this initial data-packet implementation.
