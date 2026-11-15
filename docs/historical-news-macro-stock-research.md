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

## Phase 3: PIT news-to-real-price observational outcome cohort (PR #172)

Reuses operator-confirmed **existing** private NVDA/SPY Tiingo exports
(`nvda-eod-2016-2026.csv`, `spy-eod-2016-2026.csv`, 2513 sessions
reported each, first 2016-11-15 and last 2026-11-13). No new stock API
calls, no download, and no price data committed to Git.

The new `historical-price-outcomes` CLI joins **all available daily
sessions**, not only the retrospective TOP100, to avoid automatically
training only on outcome-selected shock dates. Each row keeps strictly
separate:

- `features`: as of **09:30 America/New_York** on the event session,
  previous trading session's known adjusted closes, and articles first
  observed before the cutoff in the last three days. Source URLs/capture
  SHA and first-observed times are pinned; missing 10-year news coverage
  is recorded as missing, not fabricated. News candidate links to the
  issuer remain unverified.
- `outcome`: future-labeled adjusted close-to-close stock/benchmark
  return and benchmark-excess direction for 1/3/5 trading sessions
  beginning at the **previous trading session's close**. These are
  only observational outcome values, not a tradable 09:30 return:
  the return includes premarket movement before cutoff, and end-of-day
  values become observable only after market close. Neither label values
  nor TOP100 selection are inserted into `features`.
- Optional source-attested timestamped macro revision CSV uses the
  existing `read_macro_releases` and `_context` point-in-time logic.
  FRED *date-only* vintage JSON is not accepted as this CSV.

The source file hashes, covered/missing sessions and context truncation
are reported. The private generated JSON file is create-only mode 0600;
source inputs and production DB remain unchanged. This is a robust
**join and outcome-label foundation**, not a validated training corpus:
`trainingReady=false`, `modelPredictionGenerated=false`,
`priceVendorAuthenticityIndependentlyVerified=false`, no independent
causal attribution and no real forecasting accuracy claim. In particular,
a historical RSS item first captured in 2026 is never silently dated as
known in 2016.

First inspect local source coverage, without a model call or an output file:

```sh
bash scripts/news-learning-review.sh --update historical-price-outcomes \
  --symbol NVDA --benchmark-symbol SPY \
  --stock-csv "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-eod-2016-2026.csv" \
  --benchmark-csv "$HOME/Library/Application Support/TrisMarketEventIntelligence/spy-eod-2016-2026.csv" \
  --news-cohort "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-cohort-20261115.json" \
  --preview
```

Once the preview confirms source presence, run the same command without
`--preview` and add
`--output "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-spy-news-price-outcomes-v1.json"`.
Do not overwrite existing private reports.

**Next logical feature after confirming the real Mac output:**
a genuinely point-in-time company/sector news association and deduped
story-corpus enrichment; keep unverified issuer associations and missing
historic source timestamps out of supervised training. Then build a
time-split sealed prediction/actual-outcome evaluation, using the above
separated feature/outcome contract rather than copying outcome fields
into a model prompt. Using a settlement-based label as if it were a
09:30 executable forward return would be wrong; choose a properly
tradable return horizon or acquire authenticated interval/open prices
before model backtest.

## Historical GDELT coverage expansion after real Mac 0/2512 news result

On 2026-11-16 a real Mac `historical-price-outcomes --preview` run using
existing NVDA/SPY and `news-cohort-20261115.json` returned
`observations=2512`, `newsCoveredSessions=0`,
`newsMissingSessions=2512`, `trainingReady=false`.
The price join is operational; the **supplied 2026 RSS cohort** has no
usable first-observed-at-or-before-09:30 records in these historical
trade-date windows. This does not disprove the collector or other already
archived historical source references.

New opt-in `gdelt-replay` reuses the existing 100-rank NVDA/SPY
retrospective shock packet and the proven `gdelt-scout` collector. It can
fetch up to **three ranks in one explicitly invoked batch** and pins four
GKG archives per rank in independent, private `0600` receipts. It
re-validates any existing receipt against exact underlying study SHA, rank,
shock day, expected GDELT slots, archived ZIP SHA field, and strict
unverified issuer/article flags. Valid older rank-1 files are **reused**,
never re-fetched; a bad or mismatched receipt causes an explicit failure
rather than silently overwriting. If a later rank fails, completed earlier
ranks persist and can be replayed safely.

To inspect coverage without a network request:

```sh
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
bash scripts/news-learning-review.sh --update gdelt-replay \
  --study "$DATA/nvda-spy-shocks-2016-2026.json" \
  --data-dir "$DATA" --start-rank 1 --count 3 --preview
```

To run a *small* missing-rank acquisition (up to four real GKG ZIP downloads,
no LLM calls; reuse the pinned actual rank-1 result when supplied):

```sh
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
bash scripts/news-learning-review.sh gdelt-replay \
  --study "$DATA/nvda-spy-shocks-2016-2026.json" \
  --data-dir "$DATA" --start-rank 2 --count 1 --run
```

This is a **research source inventory** only, not a completed news-to-price
training cohort. A GKG archive slot, organization mention and article URL
do not attest first article publication/tradable availability, read the
original publisher's body, or prove market causality. Source rights,
duplicate story grouping, first-publication evidence, issuer matching,
controls, and independent as-of checks are needed before any historic
reference can enter forward-replay prediction inputs. Original 2026 RSS
capture times must not be rewritten or backdated. No user-approved
long-running loop, automatic worker install, model training or promotion
is performed here.

## Historical issuer primary-source corroboration: 2018-11-16 NVIDIA

Real Mac GDELT rank-2 collection confirmed 21 2018-11-16 archive URL
candidates, and the user ran `historical-story --prepare` to obtain
`referenceCount=21`, one possible repost group, 12 unclear categories,
3 sector/competitor, 3 company other, 1 earnings/guidance and 2 macro.
There was **no verified official primary source within that GKG URL-only
review** (`officialPrimarySource=false`). Article bodies and actual first
publisher times remain unknown.

Public research identified a separate first-party
[NVIDIA FY2019 Q3 earnings press release](https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-third-quarter-fiscal-2019-6747552),
dated 2018-11-15, preceding the 2018-11-16 shock trading day.
It confirms third-quarter revenue about USD 3.18bn, next-quarter outlook
USD 2.70bn +/- 2%, and management's discussion of excess channel
inventory after the cryptocurrency boom. These facts are pinned as short
paraphrases in
`research/nvda-2018-11-15-issuer-primary-evidence.json`.
A contemporaneous Reuters report syndicated on Yahoo Finance exists as
secondary timing context, but its original publication/first-tradable
clock is **not independently verified** in this local study.

New `historical-issuer-corroborate` strictly checks the original
historical-story review and curated issuer evidence contract, retains
hashes of both documents, counts source categories and original URL
matches, and emits a 0600 private comparative research packet with
distinct **verified issuer source relevance** versus **unverified GKG
organization/article association**. A document's calendar date does
not establish that it was already available at 09:30 the following
morning; even a plausible contemporaneous report is not automatically
proof that it caused the adjusted share return.

Use the already generated rank-2 private story review:

```sh
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
bash scripts/news-learning-review.sh --update historical-issuer-corroborate \
  --review "$DATA/nvda-story-review-rank2-v1.json" \
  --issuer research/nvda-2018-11-15-issuer-primary-evidence.json \
  --output "$DATA/nvda-2018-issuer-corroboration-v1.json"
```

This command is offline and does not call Gemma. It does **not** fetch a
copyrighted news article into training, fabricate an intraday release
timestamp, independently authenticate historic issuer PR original bytes,
promote any GKG URL to a PIT-valid model input or verify a causal
price impact. The current output keeps
`replayTrainingEligible=false`, `priceImpactCausalityVerified=false`,
`prospectiveForecastProduced=false`.

The next useful feature is independently verifiable first-publication
timing and issuer link/duplicate cluster validation so genuine
contemporaneous news can join the previously pinned NVDA/SPY price-outcome
calendar, followed by an independently scored time-forward prediction.

## SEC accepted-time anchor for 2018 NVIDIA earnings-to-price research

The real Mac issuer corroboration command for 2018-11-16 succeeded:
`OFFICIAL_DATE_AND_FACTS_CORROBORATED_CLOCK_UNVERIFIED`,
21 source-URL candidates, three NVIDIA issuer facts, zero exact GDELT
matches to the official issuer release and no verified publisher-first
clock. Existing private file:
`nvda-2018-issuer-corroboration-v1.json`.

**New independently checkable temporal evidence:** SEC EDGAR accession
[`0001045810-18-000148`](https://www.sec.gov/Archives/edgar/data/1045810/0001045810-18-000148-index.htm)
is NVIDIA's 2018-11-15 Item 2.02 8-K. EDGAR reports an **accepted**
timestamp of **2018-11-15 16:21:48 Eastern (21:21:48 UTC)** and
includes [Exhibit 99.1](https://www.sec.gov/Archives/edgar/data/1045810/000104581018000148/q3fy19pr.htm),
the 2018 Q3 earnings release. This demonstrates the submission was
accepted by EDGAR **before Friday Nov 16 09:30 New York open**.
It does **not** establish NVIDIA's first press-release publication
timestamp, when all traders first received it, or that this disclosure
caused the observed stock move.

`historical-sec-event` uses only pinned public SEC evidence, the
existing user-private source corroboration and the already acquired
NVDA/SPY adjusted-close CSVs. It produces a single **historically
observed** 2018-11-16 price/market outcome, while retaining a distinct
feature record with SEC receipt time and original issuer facts. The
label uses **2018-11-15 close → 2018-11-16 close**, NOT an
executable Friday 09:30 entry/forward return. The outcome is shown
only as retrospective research; modelPredictionMadeBeforeOutcome,
trainingReady and causalImpactVerified remain false. Source SHA and
both price-source SHA fields are pinned in a create-only private
mode-0600 output; neither original CSV nor production database changes.

Run on Mac with existing files (no new web request, no model call):

```sh
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
bash scripts/news-learning-review.sh --update historical-sec-event \
  --corroboration "$DATA/nvda-2018-issuer-corroboration-v1.json" \
  --sec research/nvda-2018-11-15-sec-8k-acceptance.json \
  --stock-csv "$DATA/nvda-eod-2016-2026.csv" \
  --benchmark-csv "$DATA/spy-eod-2016-2026.csv" \
  --output "$DATA/nvda-2018-sec-price-event-v1.json"
```

Next development after actual Mac evidence: generalize verified public
issuer/SEC available-at anchors to a **multi-event PIT-safe case
dataset**. Never promote GDELT URL references to issuer stories
automatically. Expand through events with real contemporaneous
documents and control days before modelling directional forecasts.
