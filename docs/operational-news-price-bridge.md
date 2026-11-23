# Operational News → Actual Minute-Price Outcome Bridge

## New PR scope and motivation

This work starts from the **updated main** following the #261 and #263
cleanup. It is not a continuation of an old draft branch. Its purpose is
to connect existing `news.sqlite3` records to explicit forecast and price
source artifacts **without generating fictitious article-to-issuer mappings,
invented timestamps, forecast seals, or fabricated returns**.

### Input requirements

- `--db`: actual, locally collected news SQLite with the existing
  `articles`, `captures`, `article_full_text`, and `story_members` schema.
  Every database access is **read-only**.
- `--mappings`: CSV with all columns defined by the existing
  `point_in_time_source_validation_cli.EVENT_COLUMNS`, plus
  `body_sha256` of the original captured source article. Each row is
  a separately supplied, predeclared target/benchmark/direction/horizon.
  Do not derive a ticker from an LLM's free-text target without an
  independently supported mapping.
- `--source-manifest`: the existing SHA-256-pinned `minute_source_v1`
  manifest referencing actual adjusted one-minute price data (not
  the existing unadjusted daily KRX feed). An operator-authored manifest
  proves file consistency, **not** original exchange provenance.
- `--sessions`: explicitly supplied trading calendar; optional
  `--halts` and `--evidence` inputs are passed to the existing
  original-publication and market-data validation contracts.
- `--output`: new diagnostic JSON path; create-only, never overwrite.

### Join policy

The real `news.sqlite3` audit runs first and verifies SHA-256 of stored
RSS and original article bytes, publisher-stated and collector-observed clocks,
and candidate story duplicates. Only source rows with no critical source
blockers proceed to market observations.

Each mapping must resolve to a real stored article with the same URL,
publisher-stated and detection times, and matching SHA256 against
the **actual stored original article body**. Reprinted or already-grouped
source articles are withheld from independent scoring until independently
reviewed. When a forecast is marked `LIVE`, the local original body
capture must be no later than the forecast seal.

`OPERATIONAL_ARTICLE_NOT_ADMITTED` is reported for incomplete, duplicate,
missing, or mismatched news records. No market-direction match is returned
on those rows. Approved **for descriptive evaluation** does not mean
that the release time, ticker connection, price provenance, or causal
effect have been independently certified.

The existing outcome calculator computes both publication-to-horizon
absolute/benchmark-relative returns and forecast-seal-to-horizon direction
on supplied adjusted minute prices, with explicit exchange dates,
missing bars and trading halt exclusions. If source audit, market
mapping, price coverage or exchange sessions fail, status is explicit
and returns do not become fictitious neutral labels.

### Usage

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
PYTHONPATH="$PWD/src" python -m tris_market_event_intelligence.operational_news_price_bridge_cli \
  --db "$DATA/news.sqlite3" \
  --mappings "$DATA/market-reviewed-forecast-mappings.csv" \
  --source-manifest "$DATA/minute-source-manifest.json" \
  --sessions "$DATA/exchange-sessions.csv" \
  --output "$DATA/operational-news-price-outcomes-v1.json"
```

Only use those mapping/price paths when the files actually exist.
This command **does not download market quotes or automatically author
forecast CSVs**. It reports how many real stored articles were linked,
admitted for descriptive research, blocked, or observed, and each
prediction's actual price return and explicit limitations.

### What remains unproven

Actual authenticated earliest-public time, source-origin attestation,
independently sourced symbol/sector/ETF mapping, genuine minute feed
availability and price-adjustment quality, prospective forecast sealing,
statistically independent holdout and improved learning performance.

For every output, `trainingEligible=false`,
`independentlyVerifiedMarketOutcome=false`, and
`sealedProspectiveForecastCertified=false` until separately established.
No rows are automatically promoted into the autonomous Learning Engine.

## Regression coverage

Integration tests use the real collector table shapes and **synthetic**
price fixtures. They cover intact real-article identity, separate
pre-release versus post-seal returns, tampered original body hash,
a live forecast occurring before original capture, a timestamp mismatch,
and nonexistent article IDs. These prove fail-closed behavior, not real
market forecasting quality.
