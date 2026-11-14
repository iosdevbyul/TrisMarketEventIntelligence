# Stage 1 — Naver/GDELT first receipt and network latency

## Long-term product goal

**KR/US real-time material news detection → affected stock identity →
minute-price response learning → future calibrated impact prediction.**

This work improves the first stage's instrumentation only. It does not
prove that Naver, GDELT or RSS deliver breaking market news in time for
minute-level decisions, and it does not complete any full product stage.

## What changed

The Naver and GDELT discovery collector previously assigned an article's
`detected_at` from a clock captured **before** sending the HTTP request.
With this change, both sources use the timestamp captured immediately
**after the HTTP response arrives**. The response stamp is passed to
`parse_naver`/`parse_gdelt` and persisted as first receipt in
`articles.detected_at` and `captures.fetched_at`.

Every actually attempted discovery API request now persists an append-only
`fetch_attempts` record in the same table used by RSS:

- UTC request-start, HTTP response-arrival and persistence-complete timestamps
- Monotonic HTTP request elapsed milliseconds
- `ok` or `error` outcome, article counts returned and newly inserted
- HTTP 429 rate limit and 401/403 denied requests as failed attempts before
  stopping the rest of that source's queries

This preserves existing optional Naver credentials, topic selection,
GDELT source behavior, rate-limit protections, article deduplication,
raw-capture SHA-256 and `fetch_state` error codes. Repeated discovery
of an article never overwrites its original `detected_at`. Missing Naver
credentials still cause **no** external request and no false fetch attempt.

The existing read-only command `market-event-news-latency` can now
report Naver/GDELT as well as RSS per-source request-duration and
publication-to-first-receipt distributions on a local database containing
these collection observations. The user does not need to run this command
for CI validation.

## Important interpretation

- Naver `pubDate` is provider-supplied article date, not independently
  authenticated global breaking-news time.
- GDELT `seendate` is a discovery/index observation timestamp, and **must
  not** be misrepresented as original publisher publication time.
- HTTP response receipt is the earliest time this particular collector
  can claim it had the article; it is not necessarily the source's first
  release or the earliest possible trading decision.
- Duration uses a monotonic clock; timestamps use UTC wall time. These
  are distinct measurements.
- The API may return previously published or stale news, and polling
  intervals plus API quotas can make coverage incomplete.
- The full news→event extraction→Gemma completion audit from PR #59
  stays separate. No latency figures are invented without observed runtime.

## Verification and next tasks

Mock-HTTP tests cover Naver and GDELT receipts, duplicate articles,
blocked source handling, missing credentials and the shared read-only
latency report. GitHub Actions runs Python 3.11, 3.12 and 3.13.

Before stage 1 may be called complete, run authorized KR/US market-session
observations, establish per-source discovery lag and source coverage,
assess actual response-to-Gemma completion time, and track missed
headlines, stale responses and model turnaround relative to market moves.
No source credentials or live market data are added here.
