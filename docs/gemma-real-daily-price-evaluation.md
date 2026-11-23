# Gemma original-news results × observed adjusted daily prices

## Purpose and scope

This is the first **new feature PR after #264** to operationalize real,
historically observed prices already present in the user's local
`TrisMarketEventIntelligence` research directory.

It brings the preserved Stage 6 research prototypes into updated `main`,
without reopening #258, rewriting the existing Gemma JSON/price CSV, or
modifying production `news.sqlite3`. All market outcomes use observed
adjusted daily prices; none are fabricated or filled with zeros.

## Modules

- `market_outcome_validation_cli`: individual original-article Gemma
  target/direction/reason -> reviewed stock/ETF mapping -> 1/5 trading
  session observed absolute and benchmark-relative returns, including
  per-impact `CORRECT`/`INCORRECT`/`ABSTAINED`/`UNAVAILABLE`.
- `market_price_collection_cli`: explicit-symbol, optional yfinance
  research adapter for actual adjusted daily bars (not minute/quote),
  with source errors represented as unavailable.
- `market_price_recovery_cli`: recover *fully missing instruments* from
  a preserved older source export, never mix different provider snapshots
  within a given instrument. The recovery manifest binds output and
  mapping files to SHA256.
- `market_proxy_cohort_cli`: restrictive allowlist for exact named
  proxy targets, refusing to guess symbols for macroeconomic claims.

## Conservative evidence checks

The evaluator also supports two additional *optional* gates:

- `--news-db news.sqlite3` audits the actual on-device article ledger,
  requiring original URL, published timestamp, detected timestamp, body
  SHA256 and source document integrity to match the Gemma report.
  Incomplete, mismatched, duplicated or unverified source records become
  `NEWS_SOURCE_NOT_ADMITTED`; these do not contribute to the price
  comparison denominator. The DB is opened read-only.
- `--price-manifest` accepts the *preserved recovered-price* manifest
  which has `outputSha256` and `mappingsSha256`. Mismatched source
  bytes are a hard error; a matching hash does **not** establish a
  provider's exchange authenticity or original point-in-time availability.

Time cutoff and interpretation:

1. Parse the original article's time with an explicit UTC offset, convert
   to the mapped exchange's local date.
2. Treat the first **subsequent trading-date adjusted daily close** as
   the entry reference. Compare the next 1- and 5-session closes.
   This deliberately does not measure the initial intraday news shock,
   the market open-to-close gap, or executable realized returns.
3. Show the absolute stock return, benchmark return, benchmark-excess
   return and both corresponding direction verdicts. Missing sessions
   and benchmarks are NOT zero returns.
4. Report the denominator of scored impacts, number of distinct articles,
   publication dates and candidate story groups. Compare agreement with
   always-UP, always-DOWN and always-NEUTRAL baselines on the same
   scored cohort to reveal direction skew; these are descriptive
   comparisons, not learned model improvement.
5. All saved Gemma article predictions in the existing
   `main-validation-gemma` batch have
   `predictionSealedBeforeOutcome=false`. They remain
   `RETROSPECTIVE_UNSEALED` even if the direction happens to match.
   `prospectivePredictionAccuracy` is null and
   `trainingEligible=false`. Later verified target links are labeled
   `RETROSPECTIVE_POST_EVENT_MAPPING`.

## Run against existing real Mac price exports

Run in a fresh checkout containing the merged feature (or the new PR
head), with the user's existing files:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"

PYTHONPATH="$PWD/src" python -m \
  tris_market_event_intelligence.market_outcome_validation_cli \
  --gemma-dir "$DATA/main-validation-gemma" \
  --news-db "$DATA/news.sqlite3" \
  --mappings "$DATA/stage6-sector-target-map-v2.csv" \
  --prices "$DATA/stage6-sector-prices-v4-recovered.csv" \
  --price-manifest "$DATA/stage6-sector-prices-v4-recovered.csv.manifest.json" \
  --price-source "Yahoo Finance via yfinance historical adjusted daily; recovered source lineage" \
  --output "$DATA/gemma-real-daily-outcomes-pr265-v1.json"
```

This command reads actual historic source artifacts from the existing
machine. Do not run it with synthetic fixtures and present them as
historical findings. Output is create-only and no previous report is
overwritten.

If all articles are denied by the optional operational source gate,
inspect the per-item `sourceAuditStatus` first. This can occur if
existing Gemma JSON is missing a reliable source URL/body hash or was
generated from a different article. It must NOT be bypassed by inventing
field values or backdating the model's analysis.

If an old recovered-price manifest does not contain the expected
SHA256 fields, omit `--price-manifest` only to perform an explicitly
less-trusted descriptive investigation. Never fabricate a matching
manifest, claim the price source is independently attested, or
include it in training.

## Unmet real-market learning milestones

Actual source-verified first-public time, independently documented
target/benchmark membership, true price-provider authorization,
minute/hour release-window observations, immutable predictions sealed
**before** those outcomes, separate date-grouped and family-grouped
heldout testing, and empirical improvement over constant-direction
baselines remain unproven. Only successful independent evidence can
advance these gates. This PR does not automatically activate 24/7
collection, trading, learning, or source/provider API calls.
