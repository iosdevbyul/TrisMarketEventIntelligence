# Stage 4 — Intel Mac CPU historical statistical baseline

The permanent four-part product objective remains:

1. Real-time Korean and US material news detection
2. News-to-affected-stock mapping with point-in-time source evidence
3. Minute-by-minute historical market reaction learning
4. Future reaction-direction/magnitude predictions, **only after validation**

This is part (3)'s next small research component, **not** a trained
deployable model and not a completed product stage.

## Research-only comparison

The `market-event-cpu-baseline` command reads the
`walkforward_dataset` section from the existing
`market-event-intraday-study --split-at ...` JSON output.

It calculates, separately for each (exchange, horizon), the simplest
possible *training-only* historical prediction: mean market-adjusted
return for that exchange+horizon. It compares this **without fitting or
tuning on the holdout period** to a zero-excess-return baseline.

Metrics for historical holdout observations include mean absolute error,
root mean squared error, and empirical-direction accuracy where the
training mean is nonzero. Both comparison baselines are intentionally
weak. This cannot establish signal, calibrated probabilities, or
profitable future trades.

## Conservative validity rules

- Require at least three independent training incidents and two
  independent holdout incidents **per exchange+horizon** for metrics;
  otherwise report `INSUFFICIENT_INDEPENDENT_EVENTS`.
- Reject any target in input features, nonfinite targets, duplicate
  event/horizon pairs, mislabeled time periods, ambiguous event identifiers,
  conflicting stock/market/horizon fields and missing verified input schema.
- Recheck strict train/holdout split: training labels ended before cutoff
  minus embargo; holdout events were observable after cutoff plus embargo.
- Never use the holdout returns to choose a model parameter or tune.
- An empty dataset produces no groups and no made-up performance.
- Report `trained_deployable_model=false`,
  `real_market_validated=false`,
  `predictive_accuracy_verified=false`, and `quality_verified=false`.

## Offline workflow

No additional Mac verification is requested for this PR. GitHub Actions
runs the full test matrix. When and **only when** real reviewed data
is available, the optional command is:

```sh
market-event-cpu-baseline \
  --dataset /path/to/an-existing-intraday-study-report.json \
  --output /path/to/new-cpu-baseline-report.json
```

The input is never modified; output is create-only. Intel x86 CPU,
standard-library Python only; no GPU, MLX or downloads. Synthetic
tests merely prove the code behaves as specified.

## Before genuine prediction

Require authentic, lawful and timestamped KR/US minute bars, adjusted
price provenance, exchange calendar and halt evidence, event-to-ticker
point-in-time mappings, independence/incident grouping across symbols,
multiple rolling out-of-period folds and uncertainty/calibration evidence.
Even if one retrospective group beats zero return, **do not** mark the
prediction stage done. No live model or order handling is introduced.
