# Stage 1 — RSS source arrival timing and ingestion delay audit

## Fixed four-stage goal

1. Detect materially important news in Korea and the United States in real time
2. Link news to affected stocks using information available at that time
3. Learn stock/benchmark minute-price reactions after the news becomes available
4. Predict future conditional direction and magnitude with measured uncertainty

This change advances stage 1 collection observability; **it does not complete
production KR/US breaking-news detection**.

## Correction: never backdate arrival to HTTP request start

The original collector assigned `detected_at` and `captures.fetched_at` from
the clock **before** issuing an HTTP GET. That wrongly suggested RSS content
had been available at the beginning of network transfer. Now:

- `request_started_at`: timestamp just before the HTTP fetch
- `response_at`: timestamp immediately after receiving the HTTP response,
  used for first-seen `articles.detected_at` and capture `fetched_at`
- `completed_at`: timestamp after parsing and database persistence
- `request_elapsed_ms`: monotonic duration of the HTTP request
- `status`: `ok`, `not_modified` or `error`
- `returned` and `inserted`: fetched article count and newly stored article count

A new append-only `fetch_attempts` table is created by the existing collector
initialization, without rewriting previous captures or changing source IDs.
Existing deduplication retains the earliest article `detected_at` on repeats,
and unchanged feed checks remain recorded. Error reports still omit secrets
and request URLs.

This correctly measures response-time news *receipt*, but **not**
authentication of the original publisher's actual first publication.
The time an economic-event classifier finishes is distinct and is stored
separately as `economic_events.processed_at` in the existing pipeline.

## Read-only audit

```sh
market-event-news-latency \
  --db /path/to/known-local-news.sqlite3 \
  --output /path/to/new-latency-audit.json
```

The audit opens an **existing** DB read-only and never initializes or
migrates it. It reports per source:

- Number of stored first-seen articles
- Missing RSS publication timestamps
- Provider-declared publication timestamps *after* detected_at
- Valid publication-to-first-receipt median and p95 seconds
- Monotonic HTTP request p95 milliseconds when the new telemetry exists
- Fetch attempt counts, errors, 304 responses and last-check recency

Legacy DBs without the new `fetch_attempts` table are readable,
but `fetch_attempt_telemetry_available=false`. The report does not
silently manufacture old network timing measurements.

The collected RSS `pubDate` and Atom dates are **source-supplied fields**.
Some may be stale, updated after the original publication, absent or
coarse-grained. The publication-to-receipt lag is therefore descriptive
and not evidence of true breaking-news latency. A feed can publish far
later than a headline broke elsewhere. An RSS interval of 300 seconds
(default) is also not a guarantee of real-time market coverage.

All reports explicitly preserve
`real_time_detection_verified=false`,
`publisher_timestamp_authenticated=false`,
`first_global_publication_verified=false`.

## Remaining before stage 1 completion

Review actual KR/US source availability, publication provenance,
network polling delay, market session coverage, stale/failure rates,
news-to-stock first-seen timestamps and the elapsed time until
**Gemma 2 analysis finishes**. Then run actual local integration and
monitoring in market sessions. This PR does not touch
DonghakStockVision, its KRX API keys or any US pricing provider.

GitHub CI covers synthetic tests on Python 3.11/3.12/3.13. The user
does **not** need to run a Mac command for this code-only change.
