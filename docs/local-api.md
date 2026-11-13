# Phase 5 — Loopback event API contract

The service listens on **127.0.0.1:8100** by default, with no public network binding. This is a local-only research interface. Any untrusted process running under the same macOS user may reach the API; loopback is not authentication.

Start with `market-event-api`. The database location is shared with the collector via `MARKET_EVENT_DB` (default: `data/news.sqlite3`). The `GET /ready` endpoint returns 503 until extraction and impact tables exist; `GET /health` only confirms that the process is alive.

| Endpoint | Purpose |
| --- | --- |
| GET /health | Liveness |
| GET /ready | Local database readiness |
| GET /api/v1/events?after_id=0&limit=100 | Event stream |
| GET /api/v1/events?status=VERIFIED | Filtered event stream |
| GET /api/v1/impacts?after_id=0&limit=100 | Analysis impacts, independent cursor |
| GET /api/v1/analysis-runs?after_id=0&limit=100 | Completed analysis runs including NO_HYPOTHESIS |

Responses have `schemaVersion`, `items`, `nextCursor`, `hasMore`. All streams are ordered by ascending ID. Clients persist a cursor **only after their own durable transaction succeeds**. Re-reading the last page is safe for clients deduplicating by ID. Limit is 1–500.

**Crucial:** The event ID does not tell you whether AI inference has finished. Poll `/impacts` separately, because an impact may arrive after the original event. Poll `/analysis-runs` for explicit completion including no-impact results. Do not rely on simultaneous event/impact visibility.

The API never triggers AI calls or mutates data. Existing event statuses reflect Phase 3 rule-supported RSS text, not independent original-document authentication. Impact direction is a research hypothesis, never a measured return. Client timestamps must not be interpreted as historical availability if generated later.

## Versioning and failure behavior

No history or timestamp-based filtering is required for v1. Cursors are SQLite row IDs, scoped to one persistent database. **Replacing the database invalidates cursors; clients must reset and reconcile.** Cursor identity and stable database instance IDs are candidates for v1.1. Invalid query parameters return HTTP 422; inaccessible or incomplete database returns 503, not fabricated empty data. No client write or acknowledgement endpoint is exposed; acknowledgement happens in the consumer's own store.
