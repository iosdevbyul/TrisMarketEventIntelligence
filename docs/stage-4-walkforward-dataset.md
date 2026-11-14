# Stage 4 — Research-only point-in-time walk-forward dataset

The project's fixed direction is **KR/US real-time market-moving news detection
→ news-to-ticker mapping → minute price-response learning → calibrated
direction/magnitude forecasts** on an Intel Mac CPU. DonghakStockVision will
supply KR and US market history when it is source-verified; it is not changed
by this work.

This slice converts *already screened* one-minute event studies into a
chronological train/holdout research dataset. It **does not train a model** or
make an accuracy claim.

## Data boundaries and anti-leakage

A split has an explicit timezone-aware `--split-at` timestamp and an embargo
of at least the longest label horizon (default 60 minutes). For each
event/ticker/horizon:

- The **only** allowed inputs are known by `available_at`: exchange,
  symbol, pre-existing link relationship, exchange-local hour/weekday,
  source-to-inference latency, and requested horizon. These are an
  intentionally small baseline feature schema. Event ID, news article URL,
  source text and ex-post prices are not inference features.
- The **target** is the later observed market-adjusted excess return, never
  inserted into the input feature object.
- `TRAIN` requires an observed label exit strictly **before cutoff minus
  embargo**. `HOLDOUT` requires event information availability **at or
  after cutoff plus embargo**. Everything between is `PURGED_TIME_BOUNDARY`.
  There are no random splits, and labeled intervals must have an exact
  first post-availability close and true 1/5/15/30/60-minute exit.
- `OVERLAPPING_NEWS`, duplicates, failed time/session screens,
  unsupported links, missing prices and nonfinite returns never become
  training targets. Article duplicates are selected by earliest observed
  availability rather than input CSV ordering.
- If a source-link timestamp is later than the news's information
  availability, the dataset fails instead of pretending the link was known.
  Even accepted links have **no independent verification of their source**.

The payload explicitly keeps:
`prediction_model_trained=false`,
`training_dataset_eligible=false`,
`point_in_time_verified=false`.
These remain false until real source and adjustment provenance, licensed
minute bars, benchmark comparability, full session/halts and entity-identity
validation are available.

## Runbook

The existing GitHub CI runs all tests on Python 3.11, 3.12 and 3.13.
No separate Mac test is needed for this change.

For an independently reviewed historical dataset, the existing offline
CLI can optionally compute the split. **All CSV files are supplied by
the operator; this does not fetch or fabricate news or price data:**

```sh
market-event-intraday-study \
  --news /path/to/news.csv \
  --prices /path/to/adjusted-minute-closes.csv \
  --sessions /path/to/approved-exchange-days.csv \
  --links /path/to/point-in-time-links.csv \
  --split-at 2026-11-01T00:00:00+00:00 \
  --embargo-minutes 60 \
  --output /path/to/new-research-report.json
```

The schedule must list every covered local exchange day; unknown dates remain
unknown. The output file is create-only and will not overwrite old results.
The `walkforward_dataset` section includes samples with clear `features`
and `target_excess_return` separation plus exclusions/counts. If no
sufficient source-verified data exists, an empty split is a legitimate result,
not a benchmark score.

## What comes next

1. Verify real KR/US 1-minute prices, benchmark and exchange-calendar provenance.
2. Bind point-in-time company identities and source evidence to incidents;
   audit simultaneous cross-symbol and cross-market events.
3. Establish multiple rolling, purged evaluation windows and calibrated
   CPU-only statistical baselines before training larger models.
4. Deploy inference only after out-of-period measurement and basic drift
   monitoring. No trades are triggered from this code.
