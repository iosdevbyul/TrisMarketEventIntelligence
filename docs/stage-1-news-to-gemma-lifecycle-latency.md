# Stage 1 — News receipt to economic event to Gemma checkpoint latency

The persistent product goal is:

1. Detect material Korean/US financial news during trading hours.
2. Identify affected tickers using only information already known.
3. Learn minute-by-minute price reactions after first usable news.
4. Predict conditional market impact with calibrated uncertainty.

This audit advances **stage 1 instrumentation only**. It does not establish
production real-time coverage, finish any of the four goals, train a market
predictor or invoke Ollama.

## Existing SQLite provenance

The collector records `articles.detected_at` (first known RSS receipt)
and provider-claimed `articles.published_at`. Stage 1's prior PR #58
corrected new RSS captures from request-start to HTTP-response time, without
rewriting legacy records.

`economic_events.detected_at` copies the original receipt; the event
processor writes `economic_events.processed_at` when it stores an extracted
event. `impact_runs.processed_at` records the **successful end-of-analysis
database checkpoint**, once the model output was validated and persisted.
`NO_HYPOTHESIS` is a legitimate completed model checkpoint, not a price
forecast. Analysis errors without a successful `impact_runs` checkpoint
remain missing from the measured completion distribution.

The read-only lifecycle audit joins these existing records using
`economic_events.article_id` and `impact_runs.event_id`. It reports
per-source median/p95 seconds for:

- **Receipt → extraction checkpoint**
- **Extraction checkpoint → successful model checkpoint**
- **Receipt → successful model checkpoint**
- **Provider-declared publication → successful model checkpoint**, only when
  the claimed publication timestamp is present and not in the future of receipt

Each distribution includes its sample count. Events without successful
model checkpoints are counted separately and **never given an artificial
zero-second latency**. Model versions may each have independent checkpoints;
they are counted as separate outcomes, never as independent news events.

Timezone-naive inputs, impossible chronology or mismatched article/event
receipt times do not silently enter performance summaries. Source titles,
full article URLs, story text, API credentials and model reasoning are
not included in the report. No network access, migrations or writes.

## Optional local audit, only if actual runtime diagnosis is requested

```sh
market-event-lifecycle-latency \
  --db /path/to/known-existing-news.sqlite3 \
  --output /path/to/new-audit.json
```

The tool requires an **existing** database; it opens in SQLite read-only
mode and does not create missing tables or model records. A legacy database
without `impact_runs` is reported as not yet instrumented for model
completion, not silently upgraded.

GitHub CI verifies synthetic persisted timing flows on Python 3.11,
3.12 and 3.13. No terminal command is needed from the user for this
GitHub-only implementation.

## What this does not measure

- Actual global news breaking time: publisher timestamps are source claims.
- Response latency for other discovery providers: their collectors still
  require equivalent instrumentation.
- First generated Gemma token or model computation start: the existing
  `impact_runs.processed_at` marks a **post-completion checkpoint**.
- Whether an inference was *usable* before a stock price moved, whether
  any headline was omitted, or how well source coverage follows KRX and
  NYSE/NASDAQ sessions. Actual market-session experiments are required.

The report explicitly keeps `news_to_usable_inference_verified=false`,
`real_time_detection_verified=false`,
`publisher_timestamp_authenticated=false`,
`gemma_token_start_observed=false`, and
`live_market_coverage_verified=false`.

The **next highest-impact development** is to extend timestamp telemetry
to Naver/GDELT discovery and to measure actual collection→Gemma latency
during a supervised market-session run without inventing data.
