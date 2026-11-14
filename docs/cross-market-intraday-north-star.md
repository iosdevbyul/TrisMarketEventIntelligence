# Product direction (authoritative): cross-market news-to-price reaction AI

This is the long-term north star for **TrisMarketEventIntelligence**.
Do not replace it with an unrelated dashboard, broad market summary, or
KRX-only daily-return model. The product must **detect material news while
KR and US exchanges are trading**, preserve exactly when information first
became available, learn what happened to stock and market prices after similar
past events, and estimate conditional short-horizon **direction and magnitude**
with calibrated uncertainty and explicit evidence.

## Project boundaries

- **DonghakStockVision** will support BOTH KRX and US markets. It owns
  market-data collection, price integrity and historical/intraday bar sources.
  Keep its existing authorized KRX operations intact. Do not modify that
  repository from this workstream. Its current research and operational
  datasets do **not** necessarily include approved US minute bars.
- **TrisMarketEventIntelligence** owns cross-market news ingestion, first
  publication/detection/availability timestamps, event normalization,
  source-grounded interpretation with local Gemma 2, historical event
  alignment, training labels, CPU-based probabilistic models, and eventual
  live inference. Event identification MUST NOT be confused with causal truth.
- **TrisMarketLens** will eventually display those results, not own models.
- Intel Mac CPU is the target. No assumptions about Apple Silicon, MLX,
  GPU availability or model fine-tuning.
- Existing daily Event Study stays as a longer-horizon research tool.
  This separate **intraday** study is the primary immediate-news path.

## Current first slice: intraday market reaction labels

A new pure-Python module supports `XKRX`, `XNYS` and `XNAS`.
It converts timezone-aware timestamps into absolute UTC instants, using
Asia/Seoul for KRX and America/New_York for NYSE/NASDAQ. US daylight
saving transitions are handled by IANA zone data.

Explicitly distinct news times:
- `published_at`: original source publication timestamp, if trusted;
- `received_at`: when the collection pipeline first observed the event;
- `available_at`: no earlier than receipt, when the processing pipeline
  had the information available for a prediction.

`published_at <= received_at <= available_at` is enforced.
The first entry is the close of a whole minute **strictly after**
`available_at`, not the publication minute or the collection minute if
the info was not yet received. The sample reports the elapsed information
latency. This intentionally discards any earlier market reaction; no
'lookahead' to a previous price is permitted.

Returns are computed over precisely +1, +5, +15, +30 and +60 *elapsed*
minutes after the entry close. Both stock and benchmark must provide
fully timestamp-aligned, adjusted one-minute closes for the entire
interval. Missing bars are explicit `MISSING_MINUTE_BAR`, never silently
compressed. Boundaries crossing a regular session are marked
`SESSION_BOUNDARY`, and no normal-session labels are generated from
premarket/after-hours events in this initial version.

NYSE/NASDAQ/ KR regular hours are only **static baseline windows**.
Exchange holidays, special closures, early closes, halts, auction
microstructure, trades/quotes, vendor delay, publication timestamp
trust, symbol mapping, and corporate-action validity are NOT
automatically verified here. **No prediction or trade execution**
should be based on labels until those checks are implemented and
historical evaluation is complete.

## Local test contract

The existing Mac one-liner automatically runs all regression tests,
including intraday tests. It never invokes real inference without an
explicit `--live` / `--live-next` flag:

```sh
bash scripts/verify.sh --update
```

For *reviewed, authorized* offline data already held by the user, the
optional independent command is:

```sh
market-event-intraday-study \
  --news /path/to/known-events.csv \
  --prices /path/to/adjusted-minute-closes.csv \
  --output /path/to/new-result.json
```

Input CSV contract (headers, with example **only**; not real market data):

```csv
event_id,market,symbol,benchmark_symbol,published_at,received_at,available_at
1,XNAS,NVDA,SPY,2026-11-16T10:15:00-05:00,2026-11-16T10:15:30-05:00,2026-11-16T10:15:45-05:00
```

```csv
market,symbol,close_at,adjusted_close,adjustment
XNAS,NVDA,2026-11-16T10:16:00-05:00,100.0,adjusted
XNAS,NVDA,2026-11-16T10:17:00-05:00,101.0,adjusted
XNAS,SPY,2026-11-16T10:16:00-05:00,100.0,adjusted
XNAS,SPY,2026-11-16T10:17:00-05:00,100.5,adjusted
```

CSV files must originate from sources with actual adjusted minute-bar
provenance: setting `adjustment=adjusted` without a real adjustment
process does not make unadjusted prices valid. The reader rejects
`unadjusted` and `unknown`. The output is only historical
`stock_return`, `benchmark_return`, `excess_return`, or a missing-label
status. It explicitly records `prediction_model_trained=false`,
`quality_verified=false`, and `causal_effect_estimated=false`.


## Subsequent safety slice: explicitly supplied exchange sessions

The optional `--sessions` and `--halts` inputs now let intraday labels
fail closed for documented OPEN/CLOSED trading dates, early closing and
stock/benchmark halt windows. Unknown dates remain unknown; static weekday
fallback is flagged as unverified, **not** a production calendar.
See `docs/stage-4-explicit-exchange-sessions.md`. The schedule's content
and source authenticity must still be independently checked before training.

## Next milestones (do not skip)

1. Source-verified, market-neutral minute-bar and benchmark integration
   with authorized DonghakStockVision US/KR data when available.
2. Calendar/halts/corporate actions and news timestamp provenance checks,
   preserving session-specific after-hours and premarket labels separately.
3. Historical event-to-ticker mapping, deduplication, censoring,
   overlapping-event filters, and point-in-time feature contracts.
4. Intel CPU baseline first (simple historical distribution), then
   LightGBM / scikit-learn only once enough validated observations exist;
   walk-forward testing and calibration must beat naive baselines.
5. Real-time news ingestion to market-move estimates with confidence
   intervals, traceability, latency and drift monitoring; no autonomous
   orders, no unfounded prediction-accuracy claims.

Never treat future labels as inference-time features or a single
`NO_HYPOTHESIS` / `COMPLETED` Gemma decision as ground truth.
