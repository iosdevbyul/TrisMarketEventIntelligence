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

## Subsequent slice: point-in-time ticker links and overlapping news

The optional `--links` CSV now gates a news-to-ticker assertion by
the evidence's **actual observed timestamp** (no future discovered
link may be used at prior inference time), source-URL duplicate feeds,
and overlapping independent news at the same exchange and symbol.
Contaminated future return labels are nulled rather than treated as
training targets. These link assertions are **not yet independently
source authenticated** and actual corporate ticker identity validity
periods still need verification. See
`docs/stage-4-event-ticker-point-in-time.md`.

## Research-only chronological holdout construction

The optional `--split-at` plus `--embargo-minutes` path in
`market-event-intraday-study` builds strictly point-in-time feature rows and
matured historical excess-return targets from screened source-linked events.
Only reviewed session and ticker link inputs can enable it; rejected labels
remain excluded, and no training or verified prediction is claimed.
See `docs/stage-4-walkforward-dataset.md`.

## CPU historical baseline: no live model yet

An explicit offline `market-event-cpu-baseline` now consumes the
existing purged `walkforward_dataset` and compares an exchange+horizon
training-only historical mean return against a zero-excess-return
prediction on a held-out period. The output remains unverified
research-only evidence, **not** a deployable prediction model or
stage-completion certificate. See
`docs/stage-4-cpu-baseline-evaluation.md`.

## Rolling out-of-period CPU baseline checks

The opt-in `--rolling-windows` mode reconstructs each embargoed
training/holdout fold from original screened news, links and minute
labels, rather than recycling an earlier single-cutoff dataset.
Every holdout window is disjoint; model-free historical mean versus zero
return baseline is evaluated independently per exchange and minute
horizon. This remains synthetic-test-verified research infrastructure,
**not proof of a trained or deployable predictor**.
See `docs/stage-4-rolling-walkforward-evaluation.md`.

## Real-data handoff boundary: no fabricated minute prices

The separate read-only `market-event-minute-source-audit` now accepts
explicit hash-pinned KR/US minute-price and benchmark CSV manifest files
for a **provenance-unverified compatibility audit**. This does not
upgrade DonghakStockVision's currently documented daily/unadjusted KRX
OHLCV API into an intraday feed, nor does it prove an independently
authenticated 1-minute US or Korean source. Details and blockers:
`docs/stage-4-cross-market-minute-source-contract.md`.

## Stage 1: HTTP-response first-seen time, not request-start time

The RSS collector now timestamps each article's first known receipt
**after** the HTTP response arrives, records individual fetch-attempt
durations/outcomes, and offers a read-only source latency audit. The
feed-declared publication date is not independently authenticated;
RSS feed coverage and market-breaking-news freshness are not yet
verified. See `docs/stage-1-news-first-seen-latency.md`.

## Stage 1: full local news-to-model checkpoint latency

The read-only `market-event-lifecycle-latency` audit connects existing
`articles.detected_at`, `economic_events.processed_at`, and
`impact_runs.processed_at` into receipt-to-extraction,
extraction-to-checkpoint, and end-to-end durations. It excludes impossible
clock order and keeps missing analysis runs visibly missing, not zero.
The Gemma checkpoint does **not** prove first usable inference, publisher
breaking time or real-time KR/US coverage.
See `docs/stage-1-news-to-gemma-lifecycle-latency.md`.

## Stage 1: Naver/GDELT first-response receipt measurements

The Naver and GDELT discovery adapters now timestamp first receipt
after successful HTTP response rather than at request start, and persist
per-query success, failure and duration to the existing fetch-attempts
table. The existing latency audit can report all news source categories.
The GDELT `seendate` is an index-seen timestamp, **not independently
authenticated original publisher breaking time**. See
`docs/stage-1-discovery-first-receipt.md`.

## Stage 1 integrated operational source health and live gates

The read-only `market-event-source-health` command audits failed/stale
polls, rate limiting, missing source telemetry, and per-source recovery.
`market-event-analyze --live-freshness-minutes` can opt in to
blocking Gemma invocation for stale or source-unhealthy events without
discarding them or altering the historical research path. This is not
verification of market-wide news coverage or global publication times.
See `docs/stage-1-source-health-and-live-gates.md`.

## Stage 1 integrated operator pass (bounded, opt-in Gemma)

The existing approved RSS operational cycle is now combined with
economic event extraction, source-health checks, and an explicit,
maximum-three-event *optional* local Ollama analysis. A nonblocking
pipeline lock, first-inserted article boundary, stale-news deferral,
atomic status report and zero-side-effect dry run prevent unintended
backlog model calls or duplicate runs. No background service, actual
KR/US breaking-news timing certification or market predictor is
claimed. See `docs/stage-1-integrated-local-pipeline.md`.

## Stage 1 operator-controlled KR/US market-session readiness

The integrated RSS→extraction→opt-in local Gemma pass can now consult
an explicitly supplied KR/US trading calendar before it makes any
request. It skips known out-of-session windows, blocks unknown dates
rather than inventing holidays, and reports source-health readiness.
The separate read-only `market-event-ops-readiness` command checks
per-source polling evidence. These are **operator-supplied sessions**,
not independent proof of actual market calendars, headline coverage,
or low-latency Gemma inference. After-hours news collection remains
a separate supported path. See
`docs/stage-1-kr-us-market-operations-readiness.md`.

## Stage 1: unified offline real-data evidence packet

The read-only `market-event-operational-evidence` command now combines
source-specific collection timing, news→extraction→Gemma persisted-result
timing, a scoped recent-news freshness breakdown, persisted source
health and optional operator-reviewed KR/US market-session readiness.
Missing/private runtime data and unknown market calendar dates are
reported as unavailable, never as zero latency or verified success.
Actual Mac runtime verification is **still outstanding**.
See `docs/stage-1-operational-evidence-packet.md`.

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
