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

## General SEC event-universe foundation: all aligned NVDA/SPY sessions (PR #176)

Actual Mac user run of PR #175 `historical-sec-event` succeeded:
`SEC_FILING_PREOPEN_PRICE_OUTCOME_OBSERVED_RETROSPECTIVE`;
NVIDIA `stockReturn=-0.18755867384873848`, SPY
`benchmarkReturn=0.0026005420826571424`,
`excessReturn=-0.19015921593139562` for 2018-11-15
adjusted close to 2018-11-16 adjusted close. SEC acceptance was at
2018-11-15 21:21:48 UTC and cutoff the following session 14:30 UTC.
Original private receipt is
`nvda-2018-sec-price-event-v1.json`. The actual price observation
does NOT establish news causality or prospective forecast performance.

`historical-sec-universe` generalizes the **data structure** from
one historical event to all actually aligned priced sessions. It accepts
one or more *already verified SEC accepted-at event packets* from PR #175,
re-checks source SHA-256 against the actual existing price CSVs,
pre-decision SEC timestamp, issuer/source evidence, and the realized
event-day outcome arithmetic. Duplicate event dates/IDs and missing
prices fail closed. Rows keep
`features.pinnedSecEvents` distinct from
`outcome.horizonsTradingSessions` with 1/3/5 end-of-day observed
benchmark-excess returns.

**IMPORTANT:** a non-event session is labeled
`NO_PINNED_SEC_EVIDENCE` and
`otherCompanyOrMacroNewsAbsenceVerified=false`.
This is not a verified no-news day. It must **never** be automatically
treated as a low-importance negative training example. The source
universe has only one verified SEC event so far. The other sessions
are a coverage calendar, not negative gold or a credible predictive
training set. The output remains `trainingReady=false` and
`forwardPredictionsSealed=false`; never automatically train a
direction classifier on rows with no verified news coverage.

Real-data preview (no source changes or network requests):

```sh
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
bash scripts/news-learning-review.sh --update historical-sec-universe \
  --stock-csv "$DATA/nvda-eod-2016-2026.csv" \
  --benchmark-csv "$DATA/spy-eod-2016-2026.csv" \
  --event "$DATA/nvda-2018-sec-price-event-v1.json" \
  --preview
```

For a private 0600 observational packet, use the same arguments with
`--output "$DATA/nvda-spy-sec-universe-v1.json"` instead of
`--preview`. It refuses overwriting. To expand event coverage, pass
additional *verified* SEC evidence packets as repeated `--event`
arguments. The current specific 2018 PR #175 verifier remains available
and should not be reimplemented. Future work: real public-company events
across additional eras and low-move controls with independently verified
news, sector and macro availability, tradable horizons and time-split
sealed forecasts. GDELT URL candidates and 2026 RSS observations cannot
be backdated into this event universe.

## SEC company-history inventory for actual multi-event source discovery

Actual Mac PR #176 `historical-sec-universe --preview` confirmed
`sessions=2512`, `pinnedEvents=1`,
`unknownEvidenceSessions=2511`,
`trueNoNewsControlDays=0`, `trainingReady=false`. The all-session
price/evidence join works, but **only one historical issuer source
is verified so far**. Do not interpret the other 2,511 dates as
actual no-news controls.

New `sec-history-inventory` fetches NVIDIA's SEC CIK 0001045810
company submissions metadata from the SEC's official
[data API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces).
The manifest's older pagination files are fetched only when they
overlap the requested range, at **less than two requests per second**.
Only HTTPS allowlisted `data.sec.gov/submissions/CIK0001045810*.json`
files are accessed. It supports 8-K/10-Q/10-K candidate metadata with
accession, filing calendar day, **RAW** acceptance-date-time text,
SEC index link, original SEC page hash and document path. All downloaded
sources are hashed; the resulting output is private 0600 and new-only.
No press release/Exhibit text is retrieved or mass-copied.

**Required operator step:** The SEC requires every programmatic
request to identify the app and an actual contact in the User-Agent.
Set `TRIS_SEC_USER_AGENT` locally with a real contact address; this
is NOT an API token and must not be checked into Git, printed into
reports, or invented by ChatGPT. No user-agent => fail before network.

```sh
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
export TRIS_SEC_USER_AGENT="Your Application RealContact@example.com"
bash scripts/news-learning-review.sh --update sec-history-inventory \
  --start 2016-11-15 --end 2026-11-13 \
  --output "$DATA/nvda-sec-history-candidates-2016-2026-v1.json"
```

Use the actual user identity in the User-Agent; the value above is only
an example. Source: [SEC developer fair access](https://www.sec.gov/about/developer-resources).

**Safeguards:**
- The SEC submissions JSON is a **candidate listing**, not a
  historical as-of article corpus. A submission acceptance timestamp is
  not necessarily the first press release or first public feed time.
  Neither source availability nor PIT eligibility is inferred from the
  formatted timestamp.
- All candidate items have
  `secAcceptanceTimeIndependentlyVerified=false`,
  `filingExhibitFactsVerified=false`,
  `eligibleAsNewsModelFeature=false`; the entire inventory has
  `trainingReady=false`. A later research phase must verify exact
  issuer disclosure, 8-K item relevance, official exhibit, actual SEC
  accepted-at time and target trading session before selecting
  defensible features.
- This deliberately does **not** request NVDA/SPY prices (already
  obtained), create future-looking labels, validate price causality,
  claim a positive/negative market reaction, or activate a daemon.
  It does not treat every 8-K as earnings news.
- 10-year range uses a bounded SEC manifest, no more than 8 historical
  pagination pages plus the base, fail closed if this would exceed the
  cap. If SEC denies automated access, report the error and respect SEC
  fair-access policy rather than evading it.

**Next after actual Mac inventory:** Select a meaningfully sized
historical set of relevant earnings/disclosure 8-Ks and low-move
non-earnings comparison windows; pin individual first-party SEC
acceptance/index/exhibit evidence and dedup per issuer event; then
import **verified** source-positive cases into PR #176 SEC universe
without treating every other date as a true no-news control. Genuine
ML forecasting requires more evidence-backed cases, a tradable horizon
aligned with the decision cutoff, and sealed time-forward tests.

## SEC event Mac input diagnosis (no new private files)

Actual 2026-11-16 Mac execution of the original `historical-sec-event`
failed with the generic `INVALID_SOURCE_OR_PIT_EVIDENCE` response.
This message hides the internal validation stage, so the incident **does
not establish** a broken Conda installation, lost price history,
missing SEC filing, or a specific case of date alignment. The same
Mac has previously processed the NVDA/SPY CSVs into 2,512 observed
outcome rows, and the issuer-GDELT corroboration completed separately.

The SEC event CLI now supports `--diagnose`: use exactly the original
private inputs and `--output` path, plus `--diagnose`. It validates
and calculates the outcome but **never writes the output file**.
If the run fails, it reports a bounded JSON `failureCode` and
`phase`, never private source contents or absolute local paths.
Possible named categories include `CORROBORATION_FILE_INVALID`,
`SEC_EVIDENCE_FILE_INVALID`,
`CORROBORATION_CONTRACT_MISMATCH`,
`SEC_EVIDENCE_CONTRACT_MISMATCH`,
`SEC_ACCEPTANCE_TIME_MISMATCH`,
`PRICE_EVENT_SESSION_MISSING`,
`PRICE_PREVIOUS_SESSION_MISMATCH` and
`PRICE_FILE_OR_EVENT_VALIDATION_FAILED`.
A successful diagnose prints retrospective actual returns while
`diagnosticOnly=true`, `outputCreated=false` and
`trainingReady=false`. The original restricted provenance,
cutoff, and price validation gates are **not weakened**.

```sh
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
bash scripts/news-learning-review.sh --update historical-sec-event \
  --corroboration "$DATA/nvda-2018-issuer-corroboration-v1.json" \
  --sec research/nvda-2018-11-15-sec-8k-acceptance.json \
  --stock-csv "$DATA/nvda-eod-2016-2026.csv" \
  --benchmark-csv "$DATA/spy-eod-2016-2026.csv" \
  --output "$DATA/nvda-2018-sec-price-event-v1.json" \
  --diagnose
```

Only after diagnosis identifies the correct issue and the evidence
validation is rechecked should the non-diagnostic command create the
private research packet. Do not overwrite existing immutable receipts
or re-download historical prices as a workaround.

## SEC historical inventory x all-session event universe coverage reconciliation

The 2018-11-16 NVIDIA SEC source-to-outcome packet is actually stored and
verified on the user's Mac: all 16 read-only SHA/field/return checks PASSED.
Do not recreate it. NVDA was -18.7559%, SPY +0.2601%, excess -19.0159
percentage points on **previous-close to event-date close**, not a
tradable 09:30 forward return.

Existing `sec-history-inventory` collects SEC historical **filing candidate
metadata** and deliberately does not verify exact first-publication market
availability or issuer exhibit facts. Existing `historical-sec-universe`
constructs a 1/3/5-session retrospective price-outcome universe from
separately verified event packets and does not turn unobserved days into
true "no news" controls.

New offline `sec-coverage-reconcile` combines these existing contracts
without a fresh SEC fetch, any model call, or a new price download.
A verified SEC event is matched **by accession number** to inventory
metadata, but inventory-only filings remain **unverified review candidates**.
For each unverified accession, the filing calendar day and first following
market session are *possible audit targets only*; they are not independently
established at-market publication windows. No outcome, direction, or
stock-return field enters this review queue. Missing/unknown filing
candidates never imply that a session had no company or macro news.
Submission acceptance clocks from the raw metadata are **not** promoted
to verified market decision timestamps.

Use the existing local private historical SEC inventory JSON, the
SHA-verified 2018 SEC event packet and the previously collected EOD CSVs.
Set `INVENTORY` to the **actual SEC historical inventory path** already
collected on your Mac. Do not invent or create an inventory file if one
does not exist:

```sh
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
INVENTORY="/path/to/existing-sec-historical-inventory.json"
bash scripts/news-learning-review.sh --update sec-coverage-reconcile \
  --inventory "$INVENTORY" \
  --stock-csv "$DATA/nvda-eod-2016-2026.csv" \
  --benchmark-csv "$DATA/spy-eod-2016-2026.csv" \
  --event "$DATA/nvda-2018-sec-price-event-v1.json" \
  --preview
```

A subsequent `--output "$DATA/nvda-sec-coverage-v1.json"` is
create-only (mode 0600) if preview and inventory validation pass.
The output captures per-source SHA, form counts, verified accession
matches, unverified candidate counts, a bounded and date-ordered
human review queue, and explicit `trainingReady=false`,
`backtestReady=false`, `finalTestTouched=false`. It does **not**
perform the independent SEC index/exhibit retrieval required to
verify additional filings or prove forecasting skill.

**Next research milestone:** manually or independently certify a
small, genuine spread of archived SEC exhibit/acceptance-time cases
from different market periods and ordinary-price sessions. Then feed
those new fully verified event packets to the existing universe and
repeat this coverage audit. Respect SEC fair-use identification,
copyright conditions and independent source/time evidence. Only
thereafter evaluate truly time-forward research with sealed outcomes.

## SEC history CIK normalization and safe root-cause categories

User's actual Mac `sec-history-inventory --start 2017-01-01
--end 2026-11-16` returned `SEC_SCOPE_OR_CIK` before producing
an inventory file. This error previously combined three different
conditions: invalid date span, CIK text normalization, and unexpected
SEC filings object. The requested interval is within the code's
4000-day limit. SEC submission payloads can represent company CIK as
a decimal numeric value or a zero-padded string. The previous check
used `str(cik) == str(int(CIK))`, rejecting legitimate
`"0001045810"` textual representations of NVIDIA's official CIK.

The downloader now validates exact numeric CIK equivalence while
rejecting nondecimal, oversized, boolean, or mismatched CIK values.
The date, CIK, and filings-object failures have distinct safe codes:
`SEC_DATE_SCOPE_INVALID`, `SEC_CIK_MISMATCH`,
`SEC_FILINGS_SCHEMA_INVALID`. SEC identity safeguards, hostname
restrictions, fair-use User-Agent, source-page hashes, filename
checks, max-download bytes, strict create-only private artifacts,
timestamp quarantine and model-readiness controls are unchanged.

Use the same existing download command with `--update` and genuine
operator-configured `TRIS_SEC_USER_AGENT`. Do not duplicate SEC
downloads if the original private inventory now exists and never
paste the identifying User-Agent into issue/PR output. Successful
inventory is **candidate filing metadata only**, not verified
issuer-exhibit content or independent earliest market-publication
time. Continue with `sec-coverage-reconcile --preview` only after
the original download output is confirmed.

## SEC inventory Mac failure: unsafe primaryDocument values quarantined

On 2026-11-16 the user's actual Mac retried `sec-history-inventory`
after PR #182, and got `SEC_PRIMARY_DOCUMENT_PATH`.
The downloader successfully reached source parsing, but one or more
8-K/10-Q/10-K candidate `primaryDocument` values did not satisfy
its strict relative-document path check. The exact rejected string
has not been observed or retained; do not infer which SEC filing,
attribute, or unusual path caused the error.

This inventory is **candidate accession metadata**, not a validated
first-publication text corpus. A potentially missing or unsafe
`primaryDocument` must **not** prevent discovery of otherwise valid
filing accessions and SEC filing indexes. The decoder now records the
`secPrimaryDocument` only if it is a bounded, safe relative path.
Empty, traversal, encoded, path-like URL, control/whitespace or query/
fragment values are replaced with `null` and
`secPrimaryDocumentPathStatus=QUARANTINED_MISSING_OR_UNSAFE`.
Each row keeps the SEC accession, filingDate, raw unverified acceptance
text, SEC index URL, source page ID and SHA. The inventory records
`quarantinedPrimaryDocumentPaths` and
`allPrimaryDocumentsHaveSafePaths`, so a successful collection still
**exposes** the unresolved source evidence gaps. No suspicious document
string is ever used as a network URL or publicly logged.

This does **not** repair a missing SEC filing, prove the authenticity
of an exhibit, verify earliest market availability, or make those
filings eligible as predictive news. `trainingReady=false`,
`eligibleAsNewsModelFeature=false`; SEC CIK and URL validation, input
bounds, duplicate accession consistency and create-only private output
remain enforced. Run the same guarded official SEC inventory command
and check whether `quarantinedPrimaryDocumentPaths` is nonzero.
Then use the prior `sec-coverage-reconcile --preview` as the next
offline evidence-gap audit.

## Multi-year 8-K original SEC byte-custody + actual price outcomes

As of Nov 16, 2026, real Mac SEC inventory (2017–2026: 136 candidates)
and source-to-market coverage (one independent 2018-11-15 8-K,
135 unverified candidates) are complete. User selected the **first
8-K in each year**, without using future price returns to pick shocks.

The existing source-review manifest
`research/nvda-multiyear-sec-eight-k-official-review-2018-2026.json`
contains seven public SEC index-review cases in 2018, 2020, 2021,
2023, 2024, 2025, 2026. Five are results-of-operations 8-Ks
(Item 2.02 + EX-99.1); two are *not* earnings (Item 8.01 and 5.02).
2017, 2019, 2022 remain explicitly unreviewed. A web-reviewed filing
URL or receipt time is **not** original-byte pinning. Before this
feature the manifest deliberately had
`rawIndexAndExhibitSha256Pinned=false`.

New opt-in `sec-multiyear-study` reuses official inventory, the
source-review manifest and preexisting NVDA/SPY **adjusted daily
close** CSVs. It validates:
- Exact NVDA SEC accession against a row in the actual private
  inventory; only preselected years/forms, never hindsight returns.
- Real SEC 8-K index HTML by accession, form, date, accepted-at time,
  and item category; for earnings, EX-99.1 must appear in the index
  and the exact referenced document must have NVIDIA/revenue content.
- Raw bytes SHA-256 for seven official index pages and five genuine
  EX-99.1 pages; local source files are create-only 0600 within the
  private existing data directory, with no unsafe redirects, query
  parameters, other domains, or symlinks. Repeated runs re-validate
  existing local source bytes; they don't redownload those pages.
- Verified SEC **acceptance** occurs after the filing-day 16:00 NY
  close and before the next **real, joined** stock/benchmark trading
  session NY 09:30 decision cutoff.
- For each of five earnings and two non-earnings 8-Ks, 1/3/5 trading
  session **previous-close-to-target-close** stock/benchmark/excess
  realized outcomes. They include the overnight move *before*
  preopen decision and are not executable post-09:30 trading returns.

No first-publisher-release clock or first market availability is
inferred from the EDGAR accepted time. No market impact causality,
sealed prospective prediction, true no-news control, learning-ready
training features or automatic model promotion is established.

Run once on the Mac, with the already configured genuine SEC
identification User-Agent. The command has no network operations
until explicitly provided `--run`; `--preview` only reads previously
verified local raw HTML evidence. SEC rate is kept below 2 requests/s.

```sh
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
OUT="$DATA/nvda-multiyear-sec-8k-study-v1.json"
if [[ -e "$OUT" || -L "$OUT" ]]; then
  echo OUTPUT_ALREADY_EXISTS
else
  bash scripts/news-learning-review.sh --update sec-multiyear-study \
    --review research/nvda-multiyear-sec-eight-k-official-review-2018-2026.json \
    --inventory "$DATA/nvda-sec-history-inventory-2017-2026-v1.json" \
    --stock-csv "$DATA/nvda-eod-2016-2026.csv" \
    --benchmark-csv "$DATA/spy-eod-2016-2026.csv" \
    --raw-dir "$DATA" --run --output "$OUT"
fi
```

For offline rereads, replace `--run --output "$OUT"` with `--preview`,
reusing stored SEC HTML sources. Safe stdout lists only aggregate
event/source counts, not private filenames or credentials.
If a genuine SEC index or exhibit cannot be fetched and verified,
the new research report is not created. A partial fetch may leave
source HTML files for a safe resumption; inspect the reported fixed
failure status and do not delete or overwrite files.

The next **distinct** step, after actual Mac capture passes, is a
cross-temporal dataset with a sufficiently broad set of independently
verified SEC filings (including ordinary/non-earnings days) and a
properly sealed, time-forward evaluation. Do not equate the new
retrospective returns with forecasting performance.

## 2017–2026 annual first 8-K expansion (no hindsight selection)

Preserve original user-Mac validated v1 seven-event study JSON and
12 cached raw SEC HTML files. Newly added
`research/nvda-multiyear-sec-eight-k-official-review-2017-2026-v2.json`
contains **ten first 8-Ks of each year 2017–2026**; no source
selection used realized returns.

- 2017-01-19: `0001045810-17-000006`, SEC 16:10:06 ET,
  Items 5.02/9.01, GOVERNANCE, next trading session
- 2019-01-28: `0001045810-19-000004`, SEC 09:04:54 ET,
  Items 2.02/9.01, revised guidance, EX-99.1 pressrelease.htm,
  **same** trading session
- 2022-02-08: `0001045810-22-000005`, SEC 08:09:34 ET,
  Item 1.02, CONTRACT_TERMINATION, **same** trading session;
  no earnings exhibit

The original all-after-close implementation would misdate
2019/2022 preopen receipts. The expanded builder now selects
the same session for receipts strictly before NY 09:30 and
the next session for receipts after NY 16:00, rejecting
intraday receipts from preopen feature inclusion.
The SEC receipt is **not** the original publisher clock nor
verified public dissemination time; SEC says availability may
lag EDGAR receipt. Outputs mark
`secReceiptBeforeDecisionCutoff=true`,
`sourceAvailableByDecisionCutoff=false`, and
`independentPublicAvailabilityAtCutoffVerified=false`.
Original preopen/overnight price movements remain embedded in
previous-close → target-close retrospective 1/3/5 day returns,
**not** an executable 09:30-entry return or predictive result.
Training, causality, model promotion and 24/7 remain disabled.

Run the existing `sec-multiyear-study` command on the Mac,
replacing `--review` with the new v2 manifest and using
`--output "$DATA/nvda-multiyear-sec-8k-study-2017-2026-v2.json"`.
Use the same existing SEC inventory, 10-year NVDA/SPY CSV and
private raw SEC cache directory; existing HTML source paths are
never overwritten. The report is create-only. With all sources
validated the expected count is **10 events, six earnings,
four other categories and 16 original source HTML documents**
(ten indexes and six EX-99.1 exhibits). These numbers are
EXPECTED, not yet verified by a real Mac execution; no newly
selected SEC HTML has been locally PINned as of this code PR.


## SEC PIT publication evidence gaps and unbiased multi-form research cohort

After PR #185 actual Mac success, 10 preselected first-8-K annual events
(2017–2026) have original SEC index/exhibit SHA-256 pins and retrospective
1/3/5-session outcomes. They do **not** have independently documented
first publisher release / SEC public-dissemination / first investor
availability timestamps. An SEC EDGAR *accepted-at* clock by itself is
not a historical dissemination timestamp. No SEC receipt is promoted
to a verified 09:30 market source.

New **offline** `sec-pit-cohort` uses only the **already obtained**
2017–2026 SEC submissions inventory (136 filings, 8-K/10-Q/10-K) and
existing 10-event SHA-pinned research JSON. It creates a full
accession-keyed evidence ledger:
- raw SEC index SHA-256 and EX-99.1 SHA when already pinned, but no
  fabricated source availability for the other metadata-only events;
- regulator receipt time and receipt-clock market session (where
  previously audited) separate from earliest *publicly disseminated*
  / independently observed / investor available timestamps;
- all currently unknown publication timestamps stay null and all
  `usableAsPreopenModelFeature` remain **false**;
- unobserved-company/market-news dates are never certified as
  genuine news-free controls;
- existing retrospective price outcomes are validated structurally
  but **never copied into the review queue or source features**.

The expansion queue is deterministic and **ex-ante**: by calendar
year and SEC form choose each group's first, midpoint and last filing
(as available), with optional 1/2/3 per group. Ranking **never** uses
realized NVDA/SPY price returns or selects only previous shocks. This
covers non-earnings and quarterly/annual disclosure types in
addition to annual first-8-Ks. A candidate in this queue is a
**research task**, not independently verified market-time news.

For a no-file read-only preview, execute on Mac:

```sh
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
bash scripts/news-learning-review.sh --update sec-pit-cohort \
  --inventory "$DATA/nvda-sec-history-inventory-2017-2026-v1.json" \
  --study "$DATA/nvda-multiyear-sec-8k-study-2017-2026-v2.json" \
  --preview
```

If the preview passes, persist once to a **NEW** private JSON:

```sh
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
OUT="$DATA/nvda-sec-pit-availability-cohort-2017-2026-v1.json"
if [[ -e "$OUT" || -L "$OUT" ]]; then
  echo OUTPUT_ALREADY_EXISTS
else
  bash scripts/news-learning-review.sh --update sec-pit-cohort \
    --inventory "$DATA/nvda-sec-history-inventory-2017-2026-v1.json" \
    --study "$DATA/nvda-multiyear-sec-8k-study-2017-2026-v2.json" \
    --output "$OUT"
fi
```

No network, SEC requests, price downloads, Ollama, model inference,
training, model promotion, production changes or 24/7 behavior.
The report is create-only 0600. The manifest is an auditable
calendar/form research plan, not a permissible historical feature
dataset or a valid predictive backtest. **No training is authorized
until independently verified public availability with actual
point-in-time provenance and a much broader event/control universe
can be established and a genuinely chronological sealed evaluation
is designed.**

## PR #188 — real SEC original-index custody, historical first-public clock remains unknown

The Mac has already created the PIT cohort: 136 filings, ten SHA-pinned SEC events, and 69 ex-ante form/year research tasks. PR #188 adds an opt-in SEC original HTML capture for an explicitly bounded subset of that fixed queue.

SEC's own Webmaster FAQ (https://www.sec.gov/about/webmaster-frequently-asked-questions) explains that SEC filing acceptance and first public web availability can differ, and it does not supply an EDGAR first-web-public timestamp. Hence SEC accepted time must not be falsely promoted to a historical market-usable availability time.

New `sec-publication-capture` checks accession, original NVIDIA issuer SEC archive URL, SEC form, filing day, source SHA and the complete existing PIT ledger, then reuses any cached SEC original `sec-<accession>-index.html` without downloading it. For uncached sources `--run` explicitly fetches only the missing official SEC filing-index HTML at less than 2 requests per second and saves originals with 0600 create-only protections. Source index metadata must show NVIDIA, accession, filing form/date and SEC accepted header. The separate immutable audit JSON contains source SHA, whether cached, and the accepted header text without assuming its timezone. It NEVER states when this historic document was first public: `historicalFirstPublicAvailabilityAt` and `firstIndependentHistoricalObservedAt` remain null, `trainingReady=false`.

The source content being retrievable today cannot independently prove a 2017–2026 publisher's first publication clock. The genuine future research step is to locate authentic contemporaneous timestamped third-party observation/official dissemination feeds, retain original bytes and authenticated observation timestamp, then compare them to decision cutoffs. Do not backdate modern discovery timestamps. No price outcomes, no model promotion or 24x7.

After merging this PR, user Mac command to capture first 12 preselected source tasks (not all 69) exactly once; preserve existing files:

```sh
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
OUT="$DATA/nvda-sec-publication-original-index-batch01-v1.json"
if [[ -e "$OUT" || -L "$OUT" ]]; then
  echo OUTPUT_ALREADY_EXISTS
elif [[ -z "${TRIS_SEC_USER_AGENT:-}" ]]; then
  echo SEC_USER_AGENT_NOT_CONFIGURED
else
  bash scripts/news-learning-review.sh --update sec-publication-capture \
    --cohort "$DATA/nvda-sec-pit-availability-cohort-2017-2026-v1.json" \
    --source-dir "$DATA" --limit 12 --run --output "$OUT"
fi
```

Preview is deliberately offline cache verification, and fails if any selected HTML has not yet been captured. An actual `--run` may fail due to SEC HTTP/HTML content differences; preserve any validated partial raw cache and do not recreate existing SEC inventory, NVDA/SPY prices, v1/v2 research or 2018 independent SEC event.

## PR #189 — SHA-original SEC 10-case versus batch01 12-case overlap and historical observation witnesses

After PR #188 the user's actual Mac produced the immutable source-capture batch01: twelve verified SEC filing index HTMLs, two previously cached and ten newly downloaded. Two cache hits do not necessarily mean precisely two are in the original ten-event study; exact intersection must be calculated by accession with original 10-event SHA and raw index bytes.

New offline `sec-source-lineage` accepts the already-created PIT cohort JSON (136 filings/69 predetermined review tasks), existing actual 10-case source/price JSON, already-created batch01 12-case raw index capture JSON, and the existing private SEC HTML directory. It verifies cross-report file SHA lineage, preserves exact fixed task order, validates issuer/form/day/SEC accepted-header facts, re-hashes EVERY locally cached batch index and requires equality with both prior original-index SHA pins when overlapping. Then it produces exact accession-based overlap, distinct batch cases, total unique SEC INDEX originals and a full per-case provenance ledger. No online SEC fetch, no price CSV access, no LLM and no training. This report does **not** alter 12 existing cached source files or any input data.

Optional `--witnesses` uses an independently acquired (outside this tool) local JSON manifest `sec_historical_first_observation_witnesses_v1` of original historical third-party archive files. Each witness requires a bound accession in the captured batch, a safe relative original filename under the already-private source directory, SHA-256 of existing original bytes, an HTTPS source URL, explicit timezone-aware claimed `firstObservedAt`, optional publisher `publisherClaimedPublishedAt`, and unique witness ID. The tool can check original byte integrity and chronology but it CANNOT independently prove that a self-reported archive timestamp was first observed at that historical clock. Thus each witness is marked `SOURCE_BYTES_PINNED_HISTORICAL_CLOCK_NOT_AUTHENTICATED`, with `usableAsPredecisionFeature=false`; independently verified first public count remains 0, trainingReady=false.

Run once on Mac after PR189 CI+merge. No network and no raw file rewriting:

```sh
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
OUT="$DATA/nvda-sec-original-overlap-and-witness-audit-2017-2026-v1.json"
if [[ -e "$OUT" || -L "$OUT" ]]; then
  echo OUTPUT_ALREADY_EXISTS
else
  bash scripts/news-learning-review.sh --update sec-source-lineage \
    --cohort "$DATA/nvda-sec-pit-availability-cohort-2017-2026-v1.json" \
    --study "$DATA/nvda-multiyear-sec-8k-study-2017-2026-v2.json" \
    --batch "$DATA/nvda-sec-publication-original-index-batch01-v1.json" \
    --source-dir "$DATA" --output "$OUT"
fi
```

The result stdout contains `studyOriginalCases`, `batchOriginalIndexes`, `batchOverlap`, `batchNewDistinctCases`, `totalUniqueIndexes` and `historicalFirstPublicTimesVerified=0`. Do not invent the exact overlap before real Mac run. No-news controls and 09:30 trade returns remain unverified. Additional historical contemporaneous observation evidence requires authentic source proof, not post-hoc date parsing.

## PR #190 — SEC primary filing document content research for the fixed first-12 cohort

The prior SEC 136-filing candidate inventory, ex-ante 69 research task queue, first 12 audited SHA-pinned SEC original index HTMLs, and exact 20 unique index overlap report are already preserved on the user's Mac. **Indexes are only document manifests**, not the actual issuer filing body. This feature resolves the official form-specific primary 8-K/10-Q/10-K HTML or text document from each already-cached index Document Format Files table. No source URL can escape the official NVIDIA SEC accession archive directory. It excludes EX-99.1 and unrelated exhibits from the primary document role. Ambiguous/missing primary links are quarantined, never replaced by arbitrary external pages.

Opt-in `sec-primary-content --run` retrieves ONLY missing primary source document originals, reuses existing SHA-stable `.source` cache, stores new originals create-only/0600, and produces separate create-only/0600 research JSON. The report pins full raw primary SHA-256; extracts bounded, verbatim official-source text excerpts and observed SEC Item headings (`1.01`, `1.02`, `2.02`, `5.02`, `7.01`, `8.01`, `9.01`) as **content evidence only**. It does not summarize imagined facts, connect market outcomes, call Gemma, build a price direction model, or claim the primary document was available at the historical decision cutoff. Every record has false market-time eligibility / independent public time proof and the report `trainingReady=false`.

Only when user asks for next Mac source step after PR CI and merge, one new file can be made. The first run uses existing SEC raw index cache, with bounded N=12 original primary documents:

```sh
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
OUT="$DATA/nvda-sec-primary-content-batch01-v1.json"
if [[ -e "$OUT" || -L "$OUT" ]]; then
  echo OUTPUT_ALREADY_EXISTS
else
  bash scripts/news-learning-review.sh --update sec-primary-content \
    --cohort "$DATA/nvda-sec-pit-availability-cohort-2017-2026-v1.json" \
    --batch "$DATA/nvda-sec-publication-original-index-batch01-v1.json" \
    --source-dir "$DATA" --limit 12 --run --output "$OUT"
fi
```

Possible unresolved document links in 8-K/10-K/10-Q indices are reported accurately, not treated as model data. The report is produced on real Mac only when actual runtime succeeds; CI synthetic checks cannot establish that the first 12 official documents are already available or that independent historical news publication times exist. All existing official SEC indexes/exhibits, 10-year NVDA/SPY CSV, original 10 study, PIT cohort, 12-index batch and 20-unique-index audit remain untouched.

## PR #191 — SHA-verified SEC primary body → model-neutral retrospective semantic hypotheses

The user already captured 12 first-batch SEC filing **primary documents** successfully on Mac, saved as immutable `nvda-sec-primary-content-batch01-v1.json`, alongside twelve `sec-<accession>-primary.source` raw original files. These are not the same as the original twenty distinct SEC filing *index* cases; no need to refetch indexes, document bodies, NVDA/SPY daily price CSV, or SEC metadata inventory.

The opt-in `sec-primary-semantic` works entirely from these existing original sources. It first verifies the exact SHA-256 of each primary document on disk and recomputes the bounded verbatim excerpt and observed SEC Item headings to compare them with the saved, source-bound primary capture report. It rejects attempts to promote historical market availability or replace original source with fabricated JSON excerpts. The deterministic `--prepare` mode is completely offline, calls no model, reads no outcomes, and creates a new private 0600 research input JSON. The `--run` mode also does that validation but then delegates model calls to existing provider-neutral `LLMSettings` + `build_runtime` (Ollama Gemma4 by default, future explicitly configured OpenAI-compatible models supported) and writes new immutable 0600 model results.

Each model response is strict-schema-checked, bounded and bound to the original accession/source SHA. Every observation **must reproduce a contiguous verbatim quote** from its SHA-verified original excerpt and cite the correct source evidence ID. The three hypotheses cover issuer fundamentals, an unverified sector alternative (which must not cite the issuer-only filing as sector proof), and missing context. All are labeled unverified interpretation; model output must explicitly deny proven historical first-publication time, proven price causality and forecast direction. Invalid citations or invented quoted facts cause the **entire model run to fail without a finalized output JSON**.

IMPORTANT: Model-proposed mechanisms are possible explanations, **not confirmed truth**; even original SEC text recovered today is not proven available at a past market cutoff. The tool never reads the price outcome study, issues trading commands, trains a model, creates Gold labels, changes production DB or promotes strategies. `trainingReady=false` and historical independent public clock evidence count 0 remain. A stock price direction model belongs to DonghakStockVision.

Safe one-time Mac source-preparation command after PR #191 merged (does NOT need Ollama):

```sh
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
OUT="$DATA/nvda-sec-primary-semantic-input-batch01-v1.json"
if [[ -e "$OUT" || -L "$OUT" ]]; then
  echo OUTPUT_ALREADY_EXISTS
else
  bash scripts/news-learning-review.sh --update sec-primary-semantic \
    --primary "$DATA/nvda-sec-primary-content-batch01-v1.json" \
    --source-dir "$DATA" --prepare --output "$OUT"
fi
```

After successful preparation the user may explicitly opt in to run at most two Gemma4 source interpretations in a distinct private output file; `--limit 2` is the initial conservative batch to avoid JSON truncation on local 16GB Intel Mac. If the model response is truncated or its quotes don't match the pinned source verbatim, fail closed and inspect the exact category without weakening validators. Do not change previously saved inputs or run 12 Gemma calls automatically.

```sh
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
OUT="$DATA/nvda-sec-primary-semantic-gemma4-batch01-v1.json"
if [[ -e "$OUT" || -L "$OUT" ]]; then
  echo OUTPUT_ALREADY_EXISTS
else
  bash scripts/news-learning-review.sh --update sec-primary-semantic \
    --primary "$DATA/nvda-sec-primary-content-batch01-v1.json" \
    --source-dir "$DATA" --run --model gemma4:e2b --limit 2 --output "$OUT"
fi
```

These commands use the existing central LLM settings/env and model factory. Outputs are for historical news semantics **research** and human review only, not PIT-eligible intraday features or verified trading performance.

## PR #192 — SEC original semantic Gemma runtime diagnostic (no validation relaxation)

Actual user Mac source grounded first-case `sec-primary-semantic --run --model gemma4:e2b --limit 1` exited `SOURCE_OR_MODEL_EVIDENCE_INVALID` after PR191 merged. That prior catch-all conceals whether original SHA validation, model inventory, JSON generation, strict quote matching, or private output creation failed. No model hypotheses were recorded, and the first-case output must not be assumed created. New CLI failure prints only `FAILED_STAGE=VALIDATE_INPUT|INITIALIZE_PROVIDER|GENERATE_AND_VALIDATE_MODEL|WRITE_NEW_OUTPUT` and stable internal `ERROR_CODE` (for example `SEC_SEMANTIC_SOURCE_SHA_MISMATCH`, `MODEL_GENERATION_TRUNCATED`, `SEC_SEMANTIC_QUOTE_NOT_IN_PINNED_SOURCE`). It NEVER prints original SEC source text, model JSON, local private filenames, credentials, or model responses. Underlying original 12 SEC primary HTML SHA pins, quote-and-citation validators, model-provider factory and training/first-publication gates stay unchanged. Read-only local triage need only rerun the original one-case command after merging PR192; do NOT re-download original SEC files or redo prior sources.


## PR #193 — one evidence-constrained Gemma hypothesis correction, original attribution unchanged

Real user Mac ran the SHA-audited SEC first-case Gemma4 interpretation with PR192 diagnostics and observed:
`FAILED_STAGE=GENERATE_AND_VALIDATE_MODEL ERROR_CODE=SEC_SEMANTIC_HYPOTHESIS_UNSUPPORTED`.
Thus the local model **generated** a response and the original SEC primary input/SHA preparation and local Ollama provider setup succeeded; a specific hypothesis failed category/evidence ID/length checks. We have NOT seen the raw model reply, so the violated subcondition cannot be determined.

PR193 tightens the initial prompt and structured hypothesis JSON schema: only the issuer fundamentals hypothesis may optionally cite the single real SEC original evidence ID; the independent **sector alternative and missing context must cite no fabricated source** and may only be tentative possibilities. Each mechanism and limitation must be a meaningful 15–1000-character sentence.

When (and only when) the initial model response has already passed the original SEC verbatim-quote/observation validation but fails a **hypothesis** check, a second and final opt-in same-runtime LLM call asks to regenerate **only three hypotheses** with a restricted hypotheses-only JSON schema. Verified source observations, original quote spans, original accession, SHA and source IDs are retained byte-for-byte from the first response. The corrected combined answer must pass the exact pre-existing strict _validate_response gate. If it still fails, no JSON report is created; there is no looping, inference of unsupported source claims, softened provenance or automatic model promotion. The report records `modelCalls` (1 or 2 for first-case) and `hypothesesCorrectionCount`. No price history read, no trading/24x7, and independent first-public historical clocks still count 0.

Retest only **one** real existing Mac SEC original source first-case after this PR is CI green and merged, using the same guarded `sec-primary-semantic --run --model gemma4:e2b --limit 1` command as before. Do not re-download original SEC index HTML, primary filing bodies, or price data.


### 2026-11-17: Actual Mac SEC Gemma hypotheses-only correction still rejected

After PR #193 squash merged, the user ran the original one-case SEC primary semantic Gemma4 command against existing original SEC primary source bytes on their Mac. It failed again with `FAILED_STAGE=GENERATE_AND_VALIDATE_MODEL ERROR_CODE=SEC_SEMANTIC_HYPOTHESIS_UNSUPPORTED`. This single code cannot tell whether the first model response failed or whether the one allowed hypothesis-only correction also failed. No successful final analysis output was observed; do not assume the original SEC source or model initialization failed.

The follow-up classification changes only **redacted diagnostics of the failed hypotheses-only correction**. The final error can now distinguish invalid category, duplicate category, unsupported citation/evidence ID, too-short/long mechanism or limitation, invalid fields, missing categories, or overall hypothesis count. It does not include source text, raw model output, source paths, sensitive configuration or unverified market timestamps. When input and observations are valid, the original full strict validator is still applied, and even a corrected-model response failing that validator remains rejected with no output. No extra model retries beyond PR #193, no SEC source re-download, no price retrieval, training, trading or 24/7 enablement. The next actual Mac single-case run is necessary only after the new PR CI passes and is merged.


### 2026-11-17: Real Gemma hypothesis citation violation and model-neutral correction design

After merged PR194, the actual user Mac SEC first-case Gemma4 call returned `FAILED_STAGE=GENERATE_AND_VALIDATE_MODEL ERROR_CODE=SEC_SEMANTIC_REPAIR_EVIDENCE_INVALID`. The initial SHA-pinned SEC source and verbatim quoted observations had passed validation; even the single hypotheses-only correction inserted evidence IDs not permitted by the strict research contract. No finalized AI analysis report exists yet.

The correction model is now asked for **only** category, mechanism and limitation for three hypotheses, using a JSON schema with no `evidence_ids` field. The application assigns `evidence_ids=[]` to these *corrected, unverified hypotheses*, which prevents the SEC issuer filing being silently represented as independent proof of sector or missing-context claims. If the model nevertheless outputs any extra field (including citation IDs), the attempt fails closed rather than stripping or relabeling the generated source claim. Original SEC-quoted observations, source accession/SHA and their exact validation are preserved; corrected hypotheses must still pass the original full response validator. This is not proof that any hypothesis is factual, causal or historically available; human review remains necessary. No stock returns or future outcomes enter the inference path, no model promotion or trading.

The already captured 12 real primary documents and historical SEC/price evidence remain immutable. The actual first-case Gemma4 Mac run is pending until this change passes CI and merges. Do not repeat primary SEC downloads or re-indexing.


### 2026-11-17: Reuse successful Gemma4 SEC first case; bounded 12-case research and offline human review

The user's first **real** source-backed Gemma4 semantic report succeeded after PR195 with `cases=1`, `modelCalls=2`, and `trainingReady=false`. Preserve that immutable case-0 output; it must not be generated again. The SHA-verified SEC primary-content packet contains twelve source bodies. Avoid a single long model inference invocation on the Intel Mac: the model CLI now supports explicit zero-based `--start-index`, retaining the three-cases-per-invocation ceiling.

With the existing private `$DATA/nvda-sec-primary-content-batch01-v1.json` and `$DATA/sec-<accession>-primary.source` files already saved, create **new, distinct** private output names for the rest of the fixed cohort:

- Case index 0: use the existing `nvda-sec-primary-semantic-gemma4-firstcase-v1.json`, no rerun.
- Indices 1–3: `sec-primary-semantic --run --model gemma4:e2b --start-index 1 --limit 3` and unique output name.
- Indices 4–6: same with `--start-index 4 --limit 3`.
- Indices 7–9: same with `--start-index 7 --limit 3`.
- Indices 10–11: `--start-index 10 --limit 3`; the final selection is bounded by available case count, so only 2 cases process.

Every case still validates its original source bytes SHA, source quotations and hypotheses against the previously fixed strict SEC contract. A batch aborts without writing output when any selected case fails. Outputs are create-only; do not overwrite or invent partial successes. Do not blindly rerun failed batches; classify the returned redacted error code first.

After one or more successful batches, create a **new private local** human-review packet with:

```bash
bash scripts/news-learning-review.sh --update sec-semantic-review \
  --primary "$DATA/nvda-sec-primary-content-batch01-v1.json" \
  --source-dir "$DATA" \
  --report "$DATA/nvda-sec-primary-semantic-gemma4-firstcase-v1.json" \
  --report "$DATA/<new-batch-report>.json" \
  --output "$DATA/<new-human-review-packet>.json"
```

`sec-semantic-review` is offline and model-free. It re-reads the already SHA-pinned local SEC original bytes, independently validates every report's matching primary-report hash, per-case source identity, exact source quotes, correction/model-call counts, response shape and absence of time/training/causal promotion claims, rejects duplicate accession IDs, then produces a local review template for a human to judge each observation, tentative hypothesis and summary. The packet includes original excerpts and the model's interpretations: **it must remain private and never be published to GitHub/Notion**. Missing case indices are reported without inventing coverage; no human ratings are prefilled. A validated JSON packet is NOT a completed human assessment or historically time-eligible training evidence. Human reviewer must evaluate actual meaning, possible cherry-picking and missing external context. No price outcomes or external network calls are permitted.


### 2026-11-17: Real SEC cases 1–3 blocked by non-verbatim model quotation; exact quote-anchor contract

After PR #196 merged, the user's Mac `sec-primary-semantic --run --model gemma4:e2b --start-index 1 --limit 3` attempt failed `FAILED_STAGE=GENERATE_AND_VALIDATE_MODEL ERROR_CODE=SEC_SEMANTIC_QUOTE_NOT_IN_PINNED_SOURCE`. The full three-case batch produced no output, and the downstream human-review command was not run. The index-0 real success JSON remains immutable. The failure is expected fail-closed behavior: a model-authored quote did not exactly match a contiguous span of the pinned source. It must **not** be repaired by fuzzy matching, string normalizing or assigning some other convenient source text to an unsupported interpretation.

The initial semantic model JSON contract now uses an explicit source-owned quote anchor selection: Python derives a bounded set of exact, contiguous, 12–200-character quotations from the already SHA-pinned 1100-character primary source excerpt. The model receives those short quote anchors with stable local IDs and returns only `quote_id` and `interpretation` for each observation; it no longer writes source citations or verbatim quote characters. The application rejects missing/unknown/repeated IDs and unexpected model-supplied quote or citation fields, attaches the selected **original exact source text and established evidence ID**, and applies the full preexisting `_validate_response` gate. This separates original evidence attribution from generative commentary, but does not establish semantic relevance or causation; human review remains mandatory. Evidence IDs for independent, unsupported hypotheses remain empty as in PR195. Existing case-index 0 report remains accepted by the offline review CLI, unchanged. Do not promote the retrieved 2026 SEC content as known to historical traders; first-public time verification remains zero.

After successful CI and squash merge of the quote-anchor change, retry **once** the same new, create-only original indices 1–3 batch output; do not re-download primary SEC documents or overwrite anything. Then use offline `sec-semantic-review` to recheck exact quotes and produce a human-review packet, and inspect answer relevance before proceeding to the remaining cases.


### 2026-11-17: Four real SEC cases passed strict source checks; offline semantic quality triage

After PR197 merged, the Mac user successfully generated a new case index 1–3 Gemma4 batch (`cases=3`, `modelCalls=6`, `trainingReady=false`) and combined it with the existing immutable index-0 report into `nvda-sec-primary-semantic-human-review-cases00-03-v1.json`. The existing review command reported `reviewCasesPrepared=4`, `totalPinnedCases=12`, `casesRemaining=8`, and `allSECQuotesReverifiedAgainstPinnedSources=true`. These results establish only source-text and JSON contract validity, **not semantic correctness**.

A new offline `sec-semantic-quality` command takes the original SHA-pinned SEC primary packet, the already saved semantic reports, the untouched offline human-review packet, and the source directory. It **regenerates and byte-for-byte compares the expected review packet** against the saved one, after independent source SHA + quotation checks, before any heuristic quality triage. It checks model-authored interpretations, limitations, mechanisms and summaries for *possible* unverified causal assertions, stock direction claims and assumed historical publication knowledge. Findings include ONLY safe code, source case index and JSON field path, never SEC source quotes, model text, private paths, or unsupported facts. A clean heuristic audit is NOT a human approval. No evaluation score, research gold, trained model, production update, or time-eligibility upgrade is created. If a person edits the unreviewed packet, this integrity-only command fails rather than approving edited decisions. Human judgment must be obtained in a separate, explicitly reviewed workflow.

One-time Mac command after branch merge (each artifact must be new, create-only):

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
bash scripts/news-learning-review.sh --update sec-semantic-quality \
  --primary "$DATA/nvda-sec-primary-content-batch01-v1.json" \
  --source-dir "$DATA" \
  --report "$DATA/nvda-sec-primary-semantic-gemma4-firstcase-v1.json" \
  --report "$DATA/nvda-sec-primary-semantic-gemma4-cases01-03-v1.json" \
  --review "$DATA/nvda-sec-primary-semantic-human-review-cases00-03-v1.json" \
  --output "$DATA/nvda-sec-primary-semantic-quality-cases00-03-v1.json"
```

If successful, preserve all existing files, keep the private audit JSON local and evaluate flagged *locations* against the original review text. Continue indices 4–6, 7–9 and 10–11 from already downloaded and SHA-pinned SEC primary sources with separate new filenames. Retrospective semantic claims are not price-outcome features; no refetch, historical first-public timestamp leap or stock-direction training.


### 2026-11-17: Case 4–6 real Mac failed on unsupported model-owned safety claim

After the original 4 SEC semantic cases passed strict SHA/quote verification and the redacted static triage found three non-conclusive potential causal-language findings, a guarded run for indices 4–6 failed with `SEC_SEMANTIC_UNSUPPORTED_MODEL_CLAIM`. The error combines checks for source accession/SHA identity and three model-generated safety booleans, so the redacted error alone does **not** identify which field disagreed. No 4–6 output and no later 7–11 output or 12-case review packet was created.

The initial Ollama schema previously asked the model to emit three booleans declaring `historical_public_time_verified`, `causal_impact_proven` and `stock_direction_predicted` as false. These are not model-authoritative assessments; they express limitations of the authenticated source and PIT corpus. The initial provider schema now omits these keys entirely, with `additionalProperties=false`, and rejects any model response containing them, even if the value is `false`. The validator deterministically fills the three fields as false **only after** an exact expected field-set check. The full final `_validate_response` remains unchanged, including source accession, SHA identity, exact quotes, citation and hypothesis checks. If the real error was actually model-echoed accession/hash mismatch, this change will not hide it: those still fail closed and require separate diagnosis. Existing first four reports remain immutable and accepted by offline review; no data re-download or production/training/market forecast change.

After CI success and merge, retry only the **failed, create-only** indices 4–6 batch once; do not repeat 0–3 or assume 7–11 were run. If the same or a new error occurs, stop and report the redacted error before further changes. On success, finish 7–9 then 10–11 and regenerate the 12-case human-review and heuristic triage packets in distinct new files.


### 2026-11-17: Real retry of indices 4–6 still blocked on echoed SEC identity

After PR199 merged (source/PIT safety-status boolean fields removed from the model's authoring contract), a second real Mac run against untouched SHA-pinned SEC indices 4–6 again returned `FAILED_STAGE=GENERATE_AND_VALIDATE_MODEL ERROR_CODE=SEC_SEMANTIC_UNSUPPORTED_MODEL_CLAIM`; no 4–6 report was created. Since the source identity values `accession` and `source_sha256` remained provider-generated, and all three safety flags were now validator-generated, the remaining unsupported claim was consistent with a model-echoed accession/source-SHA mismatch; the redacted code by itself does not reveal which one.

The initial model schema now removes `accession` and `source_sha256` **as well as** all three already validator-owned safety status fields. The LLM only produces quote anchor choices, tentative interpretations, three explicitly limited hypotheses, and a summary; it cannot author *any* source-identity, quotation or absence-of-verification fields. Any unrequested provider field, including a correct-looking `accession` or SHA, fails closed. Only **after** validating the exact provider-owned field set and SHA-derived contiguous quote anchor selection, code attaches the SEC accession and SHA already independently verified from the pinned original primary-source bytes. The unchanged full `_validate_response` still checks accession, SHA, full exact quote text, evidence, hypothesis and safety flags. Source identity is not copied from model commentary; no case-index changes or existing private output overwrite.

Tests cover forged and merely echoed identity rejection at the model boundary, independently source-derived identity in the stored output, and final downstream validator rejection of modified accession or SHA. The old successful 4/12 outputs remain immutable and offline-review compatible. No new first-publication evidence has been established and none of these retrospective interpretations are eligible for model training or automated stock-direction claims.

After CI and squash merge, do **one** guarded create-only Mac retry for `--start-index 4 --limit 3` into `nvda-sec-primary-semantic-gemma4-cases04-06-v1.json`. If it fails again, stop and investigate the precise safe error without weakening source provenance. If successful, extend from original pinned 7–9 and 10–11 and create separate 12-case offline review and heuristic quality triage JSONs.


### 2026-11-17: SEC unsupported narrative guard (new generations only)

The preliminary human reading of eight Gemma4 passages identified three limitation-oriented passages, two weakly supported alternative mechanisms, and three suspected severe over-inferences. These are *preliminary annotations*, not completed source-to-claim human judgments, a population hallucination rate, or validated ground-truth labels. Preserve all twelve original historical SEC outputs without rewriting them.

New opt-in semantic generations now instruct the model not to turn SEC form section labels into invented management actions, financial practices, sector events, or policy shocks. A conservative application-side gate refuses selected high-risk concrete narrative phrases when none of the corresponding phrase patterns appears in the SHA-pinned SEC excerpt. The gate checks observations, mechanism, limitation, and summary, and applies to both the first response and any hypotheses-only correction. It emits only a fixed safe error category, without private SEC text. It never edits the original source, citation anchors, historical public availability, or archived artifacts.

**Scope and limitation:** This is a bounded lexical *veto*, not an entailment engine or a meaning-based verification result. Other unsupported claims can evade it, and even a phrase present in the original excerpt does not prove the generated causal interpretation. Keep `HUMAN_SEMANTIC_REVIEW_REQUIRED`, `humanDecisionsRecorded=0`, and all training/prediction/promotion gates unchanged. Do not portray heuristic warnings as true hallucinations or their absence as verified quality.

After merge and CI, generate a new create-only comparison batch from existing privately SHA-pinned originals with a different filename. Do not overwrite baseline reports. A later separately reviewed evaluation should align each new model claim with exact primary-source spans, distinguish observation from hypothesis and independently sourced sector context, and compare baseline versus new output only after verified same-case pairing. No independent external sector evidence is supplied in this phase.


### 2026-11-17: Private claim-to-source review packet (unreviewed)

The offline `sec-claim-review` action consumes only existing SHA-pinned original SEC packet(s) and existing saved semantic reports. It calls the full original source and model-output validation used by `sec-semantic-review`, then extracts every observation interpretation, hypothesis mechanism, hypothesis limitation, and summary into stable per-case JSON field locations. Each private row contains the source excerpt, original exact source quote candidates (issuer-related fields only), the model statement, and explicit human-review placeholders, with `evidenceStatus=NOT_ASSESSED`. **A matching quote is merely a reviewer candidate and is NOT proof that the commentary is supported.** Sector and missing-context hypotheses receive no independent industry citations.

Example local invocation (new path, no overwrite):

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
bash scripts/news-learning-review.sh --update sec-claim-review \
  --primary "$DATA/nvda-sec-primary-content-batch01-v1.json" \
  --source-dir "$DATA" \
  --report "$DATA/nvda-sec-primary-semantic-gemma4-firstcase-v1.json" \
  --report "$DATA/nvda-sec-primary-semantic-gemma4-cases01-03-v1.json" \
  --output "$DATA/nvda-sec-claim-review-cases00-03-v1.json"
```

The resulting JSON contains PRIVATE SEC source text and model commentary: never commit, paste into logs, or expose over an API. The review packet itself does not record decisions or perform a semantic entailment verdict. `humanDecisionsRecorded=0`, first-public time verification 0, and all training/prediction/promotion gates remain false. Later human reviewers may use the stable field paths to compare supported claims, unsupported specifics, and legitimately uncertain scenarios; no automated success claim is allowed.


### 2026-11-17: Same-source SEC baseline/candidate comparison

The opt-in offline `sec-semantic-compare` command takes unchanged original SHA-pinned SEC primary content, one or more baseline semantic reports, and a separately generated candidate report set. Both sides pass the existing source-byte, quote and provenance checks through `build_review_packet`; case sets must match exactly with the same accession and source SHA. A distinct candidate semantic report SHA is mandatory for each case. Original reports are never modified.

The resulting create-only private comparison JSON contains the paired case indices, report SHA references, safe field paths, and `ADDED` / `REMOVED` / `CHANGED` / `UNCHANGED` markers; it intentionally excludes generated model prose and original SEC excerpts. Mechanisms and limitations align by hypothesis category, rather than relying on the model's array order. These metrics are *difference counts only*: lower differences do not imply better or worse semantic accuracy. `semanticQualityImprovementVerified=false` and `humanDecisionsRecorded=0` must stay explicit until independent original-source claim adjudication.

Example after producing a NEW candidate for exact original SEC indices 0-3, without overwriting any baseline:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
bash scripts/news-learning-review.sh --update sec-semantic-compare \
  --primary "$DATA/nvda-sec-primary-content-batch01-v1.json" \
  --source-dir "$DATA" \
  --baseline-report "$DATA/nvda-sec-primary-semantic-gemma4-firstcase-v1.json" \
  --baseline-report "$DATA/nvda-sec-primary-semantic-gemma4-cases01-03-v1.json" \
  --candidate-report "$DATA/NEW_SAME_CASE_CANDIDATE_00.json" \
  --candidate-report "$DATA/NEW_SAME_CASE_CANDIDATE_01_03.json" \
  --output "$DATA/nvda-sec-paired-comparison-00-03-v1.json"
```

The candidate paths above are placeholders, not created artifacts. Future evidence-based human scoring should read the separate private claim review packets, assess source entailment and unsupported detail per matched field, and compare adjudicated labels. No model self-approval, historical PIT upgrade, price training, or live deployment.


### 2026-11-17: Controlled Gemma4 candidate replay on immutable SEC cases

A new `sec-candidate-replay` command prepares the **next real experiment** using the exact existing SHA-pinned original SEC primary documents and the baseline semantic reports. It refuses missing baseline case indices, tampered original bytes, forged quotes, invalid offset windows and already-existing candidate output targets before calling the model. `--prepare` is offline, never calls Ollama, and prints only a safe summary of validated indices and status; `--run` is an explicit model opt-in, bounded to 1–3 cases and a new private create-only file.

The original baseline reports must remain untouched. Do not re-fetch the SEC originals, use stock price/outcome data, or equate generated differences with improvement. Current human claim review completions remain zero. The existing validator, unsupported-specific narrative gate, and original first-publication uncertainty remain in effect. All candidate reports are retrospective, human-unverified and excluded from prediction, training and promotion.

First example replays previously analyzed **index 0 only**, so the real Mac needs just one model invocation (plus an optional bounded hypothesis correction). Then repeat the same procedure for indices 1–3 with a distinct create-only output path.

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
bash scripts/news-learning-review.sh --update sec-candidate-replay \
  --primary "$DATA/nvda-sec-primary-content-batch01-v1.json" \
  --source-dir "$DATA" \
  --baseline-report "$DATA/nvda-sec-primary-semantic-gemma4-firstcase-v1.json" \
  --start-index 0 --limit 1 --prepare
```

After safe prepare succeeds, the corresponding run is:

```bash
bash scripts/news-learning-review.sh --update sec-candidate-replay \
  --primary "$DATA/nvda-sec-primary-content-batch01-v1.json" \
  --source-dir "$DATA" \
  --baseline-report "$DATA/nvda-sec-primary-semantic-gemma4-firstcase-v1.json" \
  --start-index 0 --limit 1 --run --model gemma4:e2b \
  --output "$DATA/nvda-sec-primary-semantic-gemma4-candidate-case00-v1.json"
```

Only after a successful new candidate output should the existing `sec-semantic-compare` compare baseline against that candidate. Then separately use `sec-claim-review` for human evaluation; field differences and reviewer candidates are not evidence of better quality. Do not publish the private SEC content, provider output or reviewer text.
