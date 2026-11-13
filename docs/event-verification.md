# Policy event extraction (Phase 3)

This phase builds a conservative, deterministic event-extraction stage over the RSS records persisted by Phase 2. It does **not** call an LLM or make price predictions.

## Execution

```bash
market-event-collect --once
market-event-extract --db data/news.sqlite3
```

The processor reads the `articles` table and writes `economic_events` and `extraction_runs`. Restarting the process does not duplicate prior article/version pairs. Increasing the `EXTRACTOR_VERSION` enables explicit reprocessing without changing older outputs.

Each event contains source, source_url, article_id, capture_sha256, published_at, detected_at, processed_at, classification, status, direction, and optional basis-point change. No historical event timestamp is replaced by processing time.

## States

- `VERIFIED`: official monetary-policy feed *and* clear decision wording with unambiguous basis-point delta, or explicit unchanged-rate decision. This is **textual evidence verification**, not independent authentication of issuer documents.
- `CANDIDATE`: rate-related official-feed language with missing magnitude, competing directions, or forecast-related language; needs stronger verification.
- `UNVERIFIED`: rate-related item from an untrusted source, never marked verified by text alone.

It is safer to miss a rate decision than to mislabel a forecast as a completed official decision. Phase 3 intentionally does not follow article links to verify full policy statements, calculate policy rate target ranges, compare market expectations, or authenticate documents independently. Before automation triggers trade-related downstream processing, add full-document validation, historical reconciliation, and stricter event contracts.

Event text originates from external feeds. Treat evidence as untrusted input; do not execute it or use it as instructions. These statuses do not claim the event has a causal impact on securities.
