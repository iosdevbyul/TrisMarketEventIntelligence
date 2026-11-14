# Stage 4 — Intel Mac event-study foundation (historical labels)

This is the first CPU-only research step toward a private, locally trained
market reaction model. It computes **historical** forward stock and benchmark
returns. It does not train a model, make future predictions or establish that
an event *caused* a price move.

## Read-only integration boundary

DonghakStockVision is **not** modified. When a historical export from that
project is available, convert a **copy** to two local CSV files. Do not insert
made-up observations and do not hand edit original KRX price data.

`events.csv`:

```csv
event_id,symbol,published_at
13,005930,2026-11-13T09:00:00+09:00
```

`prices.csv`:

```csv
symbol,date,adjusted_close
005930,2026-11-16,90000
005930,2026-11-17,91000
KOSPI,2026-11-16,3000
KOSPI,2026-11-17,3010
```

The rows above are **synthetic examples**, not measured observations. For
real use, `adjusted_close` must reflect corporate actions consistently
for both the stock and its benchmark, and both must use the Korean exchange
calendar. Do not silently rename raw close to adjusted close.

Example explicit, read-only analysis, only after preparing actual files:

```sh
market-event-study --events /path/to/events.csv \
  --prices /path/to/prices.csv --benchmark KOSPI \
  --output /path/to/new-event-study-report.json
```

The CLI reads both inputs without writing to them. The optional output is
created once and will never overwrite an existing file. It does not connect
to external services or trigger Ollama. The existing command
`bash scripts/verify.sh --update` runs all new engine and CLI tests.

## Conservative label contract

- `published_at` is a timezone-aware ISO 8601 timestamp; the timestamp
  is normalized to `Asia/Seoul` before selecting Korean trading sessions.
  A publish time with no timezone is rejected.
- Entry uses **the first available equity close strictly after the Korean
  publication date**. Same-day reactions are discarded, whether the news
  arrived before or after the close. This is conservative and does not claim
  that the entry close is an attainable execution price.
- Forward labels use stock trading sessions +1, +5, +20 **after the entry
  close**. Benchmark prices must be recorded for the exact entry and exit
  dates. Missing benchmark bars are errors, not compressed horizons.
- Return = adjusted exit close / adjusted entry close - 1. Excess return =
  stock forward return - same-period benchmark forward return. No beta,
  sector or factor adjustment is yet applied.
- `INSUFFICIENT_FUTURE`, `NO_ENTRY_SESSION`, and
  `BENCHMARK_ENTRY_MISSING` / `BENCHMARK_EXIT_MISSING` are explicit
  missing-label states, never zeros or fabricated future observations.
- Output `quality_verified=false` and `causal_effect_estimated=false`.
  It contains **ex-post labels**, not scores or recommendations.
- Real study will additionally require adjusted-price provenance,
  exchange-day calendar validation, identifying the specific impacted stock
  *before* publication, and train/validation splits by time with a gap to
  prevent overlapping-horizon leakage.

## Intel Mac implementation

Pure standard-library Python 3.11+. No MLX, Apple Silicon, GPU, torch,
LightGBM or extra runtime package. Future training will consume *validated*
event-study labels from this boundary. Historical labels are never permitted
as contemporaneous inference features.

## Do not confuse with current AI evaluation

The Gemma2 v8 analysis and its `NO_HYPOTHESIS` / `COMPLETED` checkpoints
remain unchanged. The first model accepted a tentative geopolitical claim
on event 38 but its semantic justification still needs source review.
The Event Study module does **not** infer a numeric price effect from
the current seven-plus examples or the limited currently available KRX data.
