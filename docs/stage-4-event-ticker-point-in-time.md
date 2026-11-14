# Stage 4 — Point-in-time event-to-ticker links and overlapping-news screening

The enduring product objective is **KR/US live news detection, exact timing,
historical minute-level reactions and calibrated future price-impact estimates**
on an Intel Mac. DonghakStockVision supplies market data for both exchanges;
TrisMarketEventIntelligence owns event-to-ticker mapping, event studies, local
learning and predictions. No training or predictive accuracy is claimed here.

## What this slice adds

The existing `market-event-intraday-study` can now accept an optional
`--links /path/to/reviewed-links.csv` alongside the existing `--news`,
`--prices`, `--sessions` and `--halts` arguments.

This is a separate, *read-only* evidence file. Example records below are
**synthetic**, not verified investment facts:

```csv
event_id,market,symbol,benchmark_symbol,source_url,evidence_seen_at,relationship
7,XNAS,NVDA,SPY,https://example.org/story/7,2026-11-16T10:15:30-05:00,DIRECT
```

- The event ID, exchange, stock and benchmark must match exactly one linkage.
  Missing or duplicate mappings are refused rather than silently interpreted.
- `evidence_seen_at` is the time the claimed ticker relationship was
  actually observable; it must not be later than news `available_at`.
  Future-discovered ticker mappings are excluded from research labels.
- Source URLs are compared after **only known tracking parameters** are
  removed, retaining meaningful URL IDs. The earliest available event
  per exchange + symbol + canonical article wins; later duplicate-feed
  copies have no training labels.
- When a second, distinct accepted event hits the same exchange + symbol,
  the earlier event's +1/+5/+15/+30/+60-minute labels are marked
  `OVERLAPPING_NEWS` if their exit times include or follow the second
  event's availability. Two simultaneous independent events on the same
  symbol are likewise considered contaminated. Existing price returns are
  never reinterpreted as uncontaminated causality.
- `DIRECT`, `SECTOR` and `MARKET` describe the proposed link type but
  **do not** prove a financial causal mechanism. Wrong or unsupported
  benchmarks are refused for a specific event.

The report includes a per-event `link_screening` summary, flags
`point_in_time_links_supplied` and
`source_links_independently_verified=false`, and explicitly keeps
`training_dataset_eligible=false`. Excluded/censored labels have
`stock_return`, `benchmark_return` and `excess_return` set to null,
with a specific status; there are no imputed price changes.

Absent `--links`, the existing legacy offline calculation remains
unchanged, but is **not** assumed to be ready for prediction model training.
No news stream, actual stock mapping, minute-price source or US trading data
is fabricated.

## Test and integration

All new tests run automatically inside GitHub Actions and the existing
`bash scripts/verify.sh --update` local command. A separate Mac invocation
is **not required** for this GitHub-only logic.

When a fully reviewed KR/US dataset exists, the optional offline command is:

```sh
market-event-intraday-study \
  --news /path/to/news.csv \
  --prices /path/to/adjusted-minute-closes.csv \
  --sessions /path/to/reviewed-sessions.csv \
  --links /path/to/reviewed-links.csv \
  --output /path/to/new-screened-report.json
```

CSV source authenticity and the correctness of `evidence_seen_at` cannot
be proven by syntax validation alone. Before training, require real
stock-identifier validity periods, audited first-seen metadata, a trusted
KR/US minute-price feed, corporate-action adjustments, exchange calendar
and halt source verification, and walk-forward testing that prevents
same-incident leakage across train/test splits.

These are safeguards for future learning, **not** a trained predictor
or a guarantee the model will detect bad news in real time.
