# Stage 4 — explicit KR/US exchange sessions and trading halts

## Intent

The central goal remains: detect US and Korean market-moving news **while
markets trade**, align the first time the information was usable to
timestamped market bars, learn historical direction/magnitude, and later
predict short-horizon reactions with uncertainty on an Intel Mac.

Prior intraday event-study code relied on **static weekday regular hours**.
That is insufficient for holidays, exchange-specific closure days, scheduled
US early closes, special sessions and stock/benchmark trading halts. This
change introduces an **optional fail-closed schedule mode** without altering
historical daily labels, Gemma output or DonghakStockVision.

## Usage and trust boundary

For local QA, the existing one-line command runs all of these new tests:

```sh
bash scripts/verify.sh --update
```

For an explicitly reviewed, authorized offline dataset, use:

```sh
market-event-intraday-study \
  --news /path/to/news.csv \
  --prices /path/to/adjusted-minute-prices.csv \
  --sessions /path/to/exchange-days.csv \
  --halts /path/to/halts.csv \
  --output /path/to/new-report.json
```

`--halts` is optional but requires `--sessions`. No network call,
credential, extra Python dependency, GPU, or write into original inputs.

A **synthetic schedule example** (not authoritative holidays):

```csv
market,date,status,open_at,close_at
XNYS,2026-11-26,CLOSED,,
XNYS,2026-11-27,OPEN,2026-11-27T09:30:00-05:00,2026-11-27T13:00:00-05:00
XKRX,2026-11-27,OPEN,2026-11-27T09:00:00+09:00,2026-11-27T15:30:00+09:00
```

A synthetic halt example (intervals are `[start_at, end_at)`):

```csv
market,symbol,start_at,end_at
XNYS,NVDA,2026-11-27T12:58:00-05:00,2026-11-27T12:59:00-05:00
```

**Every covered date must be listed** as OPEN or CLOSED for each exchange.
Unlisted dates become `CALENDAR_UNAVAILABLE`, **not** automatically
holidays. OPEN sessions require timezone-aware opening and closing
instants on the corresponding exchange-local date. CLOSED sessions
cannot have opening or closing times. Duplicate exchange/date rows,
invalid timestamps, reversed/overlapping halts and non-positive prices
are rejected.

When the calendar is supplied, any price bars outside declared sessions
are rejected. News available on an explicitly CLOSED day yields
`MARKET_CLOSED`. News available on an unknown date yields
`CALENDAR_UNAVAILABLE`. Session closing boundaries prevent later
horizon labels. A halted stock or benchmark at entry produces
`TRADING_HALTED_AT_ENTRY`; any time window intersecting a known halt
produces `TRADING_HALTED_WINDOW` rather than a fabricated return.
Halt detection runs even when a data vendor has incorrectly carried
forward a stale price.

The resulting JSON reports `session_calendar_supplied`,
`trading_halts_supplied` and **`session_calendar_source_verified=false`**.
Providing a CSV does **not** prove its dates or halts reflect actual exchange
records. Obtain and cross-check calendars and halt events from licensed,
trusted sources before production. Corporate-action adjustment,
regular-session auction semantics, cross-venue matching, time-of-publication
authenticity and market feed latency remain distinct prerequisites.

The legacy invocation without `--sessions` still uses weekday
regular-session windows for backward-compatible offline research, but
its report identifies `holiday_calendar_unverified`. **Do not treat
unverified legacy labels as model-training ground truth.**

## Next priority

Establish explicit, source-verified calendar provenance and a dependable
KR/US one-minute price ingestion contract from DonghakStockVision when
its authorized data is available. Then develop overlap-safe event-to-ticker
alignment, point-in-time features, CPU baseline modeling and walk-forward
evaluation. No unverified predictive accuracy claims.
