# Stage 1 — Source health, news freshness and live analysis gates

**Fixed direction:** detect important Korean and US market news in real
time → link affected listed stocks → learn post-news minute-price reactions
→ forecast conditional direction/magnitude after genuine validation.

This is a single integrated Stage 1 operations feature, not a new
dashboard or a claim that live news detection is production-ready.

## Implemented together

1. **Source health:** `audit_source_health` reads local SQLite
   `fetch_attempts` and `fetch_state` read-only. It reports each
   configured source's last successful poll, last attempt age,
   consecutive failed requests, missing telemetry, explicit 429
   rate limits, 401/403 access denials and stale polls.
   Missing sources/telemetry are blocked, not assumed healthy.
   304 Not Modified is a successful poll: no new articles is not
   automatically a failure.
2. **News freshness:** `assess_live_event` distinguishes fresh
   receipt, stale receipt, absent publisher timestamp, publisher
   timestamp later than first receipt and overly old declared
   publication. A recent HTTP response carrying a month-old
   headline does not become breaking news merely because it
   was newly fetched.
3. **Pre-inference safety gate:** `ImpactProcessor.process_pending`
   optionally refuses to call Gemma when the event is old or its
   source is unhealthy. The event remains **deferred**, not
   checkpointed as successful or permanently discarded.
   The result now reports `deferred` alongside existing
   `processed`, `impacts` and `failed`. Historical research
   remains unchanged unless the live guard is explicitly enabled.
4. **Operations command:** `market-event-source-health` writes a
   create-only health report, or prints to stdout. Exit status 2
   denotes blocked/missing sources so macOS monitoring or shell
   automation can detect a problem without parsing story content.
5. **Tests:** Source recovery, failed/stale/unknown sources, provider
   429, article anomalies, model skip and checkpoint preservation,
   plus backwards-compatible non-live analysis all run under
   existing Python 3.11/3.12/3.13 CI.

## Examples — optional local operation only

Read-only source health (defaults to the existing bounded operational
RSS source IDs and a maximum 15-minute successful-poll age):

```sh
market-event-source-health --db /path/to/known/news.sqlite3
```

Explicit **opt-in** live gate; existing provider/model configuration
and the original `market-event-analyze` behavior are preserved:

```sh
market-event-analyze --live-freshness-minutes 15 --source-success-minutes 15
```

With the gate on, publisher-declared timestamps and source health must
both be present and within the supplied bounds before Gemma is
invoked. This is intentionally conservative; it may defer genuine
breaking stories with missing/unreliable publication metadata.
**Do not enable in a historical backfill or call this a reliable
headline truth detector.** Deferred items are kept for explicit
review; an old headline must not automatically be reclassified as a
new market-moving event when source health recovers.

The existing `--show-errors`, `--pilot-preview`, event-ID
selection, analysis eligibility rules and local Ollama model
configuration remain unchanged.

## Limits that still block Stage 1 completion

- A successful RSS/Naver/GDELT poll is NOT proof of complete news
  coverage; stale or missing headlines may never arrive.
- Naver/GDELT provider dates are not independently verified
  publisher breaking times.
- SQLite checkpoint health is not an independent monitoring service
  and does not yet deliver user notifications or audit actual market
  sessions. This command performs no background monitoring.
- A timestamp check alone cannot verify whether a tradeable
  stock price has already reacted to a headline.
- Full exchange-calendar, ticker identity and authenticated minute
  bars remain necessary for reaction learning.

No Mac terminal run is needed for this code-only CI verification.
