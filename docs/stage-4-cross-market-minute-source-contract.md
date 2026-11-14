# Stage 4 — Market-neutral KR/US minute-source readiness audit

## Permanent product objective

1. Detect material Korean and US news while markets are trading
2. Link articles to affected listed stocks from information available **at that time**
3. Learn minute-level price reactions relative to a relevant market benchmark
4. Predict future conditional impact only after source-verified backtests

This PR **does not complete any of the four stages**. It fills a source
compatibility boundary for stage 3 without pretending actual feeds exist.

## Confirmed current upstream state

Source: DonghakStockVision `docs/ohlcv-api.md`,
`src/donghak_stock_vision/api/ohlcv.py`, and
`src/donghak_stock_vision/data/schema.py` on the reviewed upstream
`main` branch.

The existing opt-in `GET /api/v1/stocks/{ticker}/ohlcv` returns up to
2,500 **KRX KOSPI daily, unadjusted** observations from a hash-pinned
SQLite snapshot. It does **not** provide authenticated 1-minute KR/US
quotes, a trusted KOSPI index benchmark, US stock minute prices,
corporate-action adjustments, or news publication timings. This
project MUST NOT convert that daily response into pseudo-minute data.
DonghakStockVision will later cover both markets, but work there
remains separate and this PR does not change that repository.

## Read-only dataset handoff

`market-event-minute-source-audit` reads a local JSON manifest adjacent
to 1–12 explicitly authorized, immutable minute CSV source files.
The manifest must have the following structure:

```json
{
  "schema_version": "minute_source_v1",
  "sources": [
    {
      "filename": "sample-stock.csv",
      "sha256": "ACTUAL_64_HEX_LOWERCASE_SHA256",
      "market": "XNAS",
      "provider": "example-provider",
      "source_kind": "STOCK"
    }
  ]
}
```

The SHA shown above is a placeholder, not a valid pin. Each CSV must
provide **exactly** these columns:

```csv
market,symbol,close_at,adjusted_close,adjustment,instrument_kind,provider,observed_at
XNAS,NVDA,2026-11-16T15:16:00+00:00,100.0,adjusted,STOCK,example-provider,2026-11-16T15:17:00+00:00
```

These are **synthetic schema examples**, not market observations.
`instrument_kind` is `STOCK` or `BENCHMARK`; both are only
source-declared assertions, never source-authenticated verification.
Rows require exact minute-end timestamps, timezone-aware observation
times at or after the close, valid finite positive prices and valid
`adjustment` metadata. Stocks and benchmarks may be provided across
`XKRX`, `XNYS` and `XNAS`.

The audit validates every file's SHA-256 **before and after** reading,
rejects filename path traversal, symlinks, oversized files, duplicate
market/symbol/minute records, contradictory instrument kinds, invalid
prices and malformed timestamps. It never repairs, synthesizes, fetches
or writes source data. The output includes per-source symbol coverage,
first/last minute, row count, adjustment status and declared origin.

### Status interpretation

- `SCHEMA_COMPATIBLE_SOURCE_UNVERIFIED`: metadata syntax and hash are
  consistent; **not proof** that prices were adjusted correctly or
  originated from that provider.
- `ADJUSTED_PRICE_PROVENANCE_REQUIRED`: source includes unadjusted or
  unknown adjustment. Cannot qualify as adjusted-return labels.
- `NO_MINUTE_BARS`: no observations to label.

The **overall report deliberately keeps**
`source_origin_authenticated=false`,
`adjusted_prices_independently_verified=false`,
`exchange_calendar_verified=false`,
`benchmark_comparability_verified=false`,
`news_first_seen_verified=false`,
`real_market_training_ready=false` and
`prediction_model_trained=false`, even when CSV syntax is compatible.
A bare `adjusted` CSV string is not corporate-action proof.

### Optional audit command

```sh
market-event-minute-source-audit \
  --manifest /path/to/reviewed/minute-source-manifest.json \
  --output /path/to/new-audit.json
```

Only explicitly supplied sources are read. The report is create-only;
an existing output is not overwritten. Credentials and non-public
DonghakStockVision production paths are never discovered.

No local Mac test is required for this code-only work: existing GitHub
Actions verifies Python 3.11, 3.12 and 3.13. **Actual** KR/US source
availability and provenance remain unverified until a human-authorized
offline dataset exists and its original capture evidence is reviewed.
Once that happens, connect audited adjusted minute data to the existing
intraday event-study data types without changing the label contracts.

## Remaining blockers, not silently finished

Trusted KR and US one-minute price capture, historical adjustment
proof, appropriate same-timestamp market benchmarks, symbol identity,
real first-publication/first-seen news events, early closures and halts,
overlap-safe labels, rolling out-of-period verification, and eventual
live latency monitoring. No production prediction or trading advice.
