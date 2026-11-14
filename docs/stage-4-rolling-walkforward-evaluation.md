# Stage 4 — Rolling walk-forward baseline research (KR/US)

The fixed four-step product roadmap is:

1. Detect important Korean and US market news in real time
2. Link each news event to affected stocks using information actually known
3. Learn minute-level stock and benchmark reactions after news arrives
4. Predict conditional future price impact with calibrated uncertainty

**This PR advances only step 3's offline research/evaluation infrastructure.**
No whole step is production-complete, and no real predictive skill is claimed.

## What changed

The existing read-only `market-event-intraday-study` now also supports
`--rolling-windows /path/to/reviewed-windows.csv` alongside the already
supported `--news`, `--prices`, `--sessions`, `--links` and optional
`--halts` inputs. This mode is mutually exclusive with `--split-at`.

A synthetic *example format*, NOT actual trading-date assertions:

```csv
cutoff_at,holdout_end_at
2026-10-01T00:00:00+00:00,2026-10-08T00:00:00+00:00
2026-10-09T00:00:00+00:00,2026-10-16T00:00:00+00:00
```

Each cutoff must be timezone-aware. Windows must be ordered, bounded,
nonoverlapping and span more than the minimum 60-minute embargo, with
2–24 windows supplied. The first holdout observation must have been
available by **cutoff + embargo**; a fold's test observations are only
those first available **before holdout_end_at** (exclusive). Historical
training labels must be fully mature **before cutoff - embargo**.

Crucially, **every fold is independently reconstructed from the original
screened news, ticker links and minute-label results** — not from a
previous split with permanently removed boundary observations. Earlier
holdout events may legitimately enter a later fold's training history
once their labels are known and the next embargo has elapsed. Later
holdout labels never affect earlier baseline predictions.

The existing Intel CPU empirical mean baseline is re-fit only on
that fold's permitted training observations, compared on that fold's
strictly disjoint holdout against a zero excess-return predictor.
Evaluation is independent for every KR/US exchange and for 1, 5, 15,
30 and 60 minute horizons. Sparse fold/exchange/horizon groups are
reported as `INSUFFICIENT_INDEPENDENT_EVENTS`; no score is invented.

The `rolling_backtest` section of the JSON report has each fold's
time bounds, train/holdout sample sizes, per-market metrics and
a descriptive mean of *scored fold* errors. A group with just one
scored fold is labelled `INSUFFICIENT_SCORED_FOLDS`. Aggregated
means are **not** calibrated accuracy estimates or evidence that
news caused the price move.

## Local command (only after real reviewed data exists)

```sh
market-event-intraday-study \
  --news /path/to/historical-news.csv \
  --prices /path/to/adjusted-minute-prices.csv \
  --sessions /path/to/reviewed-sessions.csv \
  --links /path/to/reviewed-ticker-links.csv \
  --rolling-windows /path/to/reviewed-windows.csv \
  --embargo-minutes 60 \
  --output /path/to/new-rolling-report.json
```

There is no need for a Mac run of this synthetic test suite. GitHub
Actions checks Python 3.11, 3.12 and 3.13. The CPU-only code uses the
standard library and does not modify DonghakStockVision or download
news or market data.

All reports retain `trained_deployable_model=false`,
`real_market_validated=false`, `predictive_accuracy_verified=false`
and `quality_verified=false`. Before training or predicting, source-
verified real Korean/US minute history, timestamps, exchange calendars,
halts, ticker identity, corporate-action adjustment, overlapping cross-
symbol incidents and calibration are still required.
