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

## Phase 1: opt-in actual provider acquisition (PR #160)

The **previous** historical shock generator could only read supplied CSVs.
Now a source acquisition command requests actual historical data from real
providers and writes source-scoped, 0600, create-only private artifacts.
This capability is **not enabled by the Mac collector automatically**.
No API credentials, endpoints or raw responses appear in stdout or Git.

### Price: Tiingo EOD

Tiingo's official API documents adjusted close and volume at
https://www.tiingo.com/documentation/end-of-day . Its current plan page
https://www.tiingo.com/pricing lists 30+ years of price history under the
Starter plan, subject to endpoint access, rate limits and personal/internal
license constraints. The NEWS API has only three months of accessible news
history under ordinary plans; it cannot substitute for the required
ten-year historical *as-published* news archive.

Set your **own** token outside git, then run one limited symbol download
only after you have decided its terms suit your use:

```sh
export TRIS_TIINGO_API_TOKEN="YOUR_PRIVATE_TOKEN"
bash scripts/news-learning-review.sh --update historical-acquire tiingo \
  --symbol NVDA --start 2016-11-15 --end 2026-11-14 \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-eod-2016-2026.csv"
```

To compare to the market, export SPY separately to a **new** file. The
Tiingo adapter uses the official authenticated endpoint, checks numeric
rows and dates, preserves provider's actual `adjClose`, writes exactly the
`date,adjusted_close,volume` columns consumed by
`historical-shocks`, verifies the output with the existing data
contract, and prints SHA256 plus *actual* first/last dates and row count.
It refuses empty, partial, malformed or duplicate rows, and never
manufactures sessions. A successful response does **not** by itself
prove the adjustment/corporate-action quality or external data
authenticity. The token must never be included in output or committed.

### Macro: genuine FRED/ALFRED date-granular revision history

Official reference:
https://fred.stlouisfed.org/docs/api/fred/series_observations.html
and https://fred.stlouisfed.org/docs/api/fred/realtime_period.html .

```sh
export TRIS_FRED_API_KEY="YOUR_PRIVATE_FRED_KEY"
bash scripts/news-learning-review.sh --update historical-acquire fred \
  --series-id CPIAUCSL --start 2016-11-15 --end 2026-11-14 \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/cpi-vintages-2016-2026.json"
```

The acquisition requests an explicit historical real-time interval,
never the default latest revised series. It returns the provider's
`realtime_start`, `realtime_end`, observation date, and value as a
**date-only quarantine** report, with no imaginary `release_at` or
`available_at`. The output is **not** automatically passed into the
historical-shocks model: FRED's real-time/vintage date alone cannot
prove when the release became publicly tradable on that day. Confirm
actual BLS/BEA/FOMC announced release timestamps and original availability
evidence before promoting a row into the separate
`macro-releases.csv` contract. Data revisions are versioned;
identifying the revision date is not identifying the news impact.

Do not put raw FRED API keys inside GitHub code, shell command text shared
in chat, or unencrypted research artifacts.

### The remaining 10-year NEWS archive gap

GDELT GKG2/Events/Mentions raw historical archives and BigQuery can
supply **candidate** media source references since February 2015:
https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/ .
GDELT DOC's ordinary search API is not a substitute for ten years
of downloadable article text. GDELT news article publication time may
be unknown or inferred; a 15-minute archive timestamp is a collection
bound, not evidence of a prior public release. Before using GDELT in
training, explicitly vet historical raw-file license/attribution,
actual article text rights, company association, deduplicated story
units and observation-time precision. Never backdate a 2026 RSS record
to its 2016 supposed publication.

### Operational stop conditions

- No API token: **no network call**, no fallback to an unofficial price feed.
- API limits/access, invalid payload, insufficient history: explicit
  failure status, **not** synthetic fills.
- Missing macro release clock or old NEWS provenance: isolate the rows,
  **not** assign them a false PIT timestamp.
- No full 10-year price/news/macro audit: **not training-ready**.
- No human-independent forward evaluation: **no forecast skill claim**.
- No automatic model promotion, investment execution or Mac 24/7
  activation.

## Real NVDA + SPY 10-year price-first run (2026-11-15)

Operator reports successful **real Tiingo EOD responses** for NVDA and SPY,
both 2513 dates, from 2016-11-15 to 2026-11-13. Private files on the Mac:

- `nvda-eod-2016-2026.csv` SHA256
  `25585b28fd93f419a6d962f5db16b7aac6e38c7868de31b41435db64a4c40338`
- `spy-eod-2016-2026.csv` SHA256
  `7a5712013f3d57f6bac7b163d1821e0f55247eec10dfcd0b59bfc3aeb6d02064`

These are operator-reported acquisition summaries; the actual CSVs were
not uploaded and independent price-vendor/corporate-action audits were not
performed. Same row counts do not prove all session dates align.

**Real next action** uses the two existing price files, the already pinned
local recent NEWS cohort, and **does not invent macro input**. It produces
a 100-shock + 100-quiet-session retrospective private packet:

```sh
bash scripts/news-learning-review.sh --update historical-shocks \
  --symbol NVDA --benchmark-symbol SPY \
  --stock-csv "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-eod-2016-2026.csv" \
  --benchmark-csv "$HOME/Library/Application Support/TrisMarketEventIntelligence/spy-eod-2016-2026.csv" \
  --news-cohort "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-cohort-20261115.json" \
  --top 100 --controls 100 \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-spy-shocks-2016-2026.json"
```

Optional `--macro-csv FILE` is only for source-attested actual
`release_at` / `available_at` historical observations. The downloaded
FRED date-only vintage JSON is **NOT** eligible and must not be renamed
to a CSV or assigned fabricated release times.

The short stdout shows up to ten most extreme historical shock days with
their actual adjusted-close returns, plus source coverage counts. The
create-only private JSON has the full 100/100 candidates. No model calls,
automatic training, economic attribution or promotion occurs; news and
macro availability can legitimately be zero on 2016-2025 shock days.
Do not misrepresent these as 10-year complete news/macro cohorts.
