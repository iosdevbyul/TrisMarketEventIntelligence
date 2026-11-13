# Phase 5-A — Expanded news ingestion

The existing RSS/Atom collector can now ingest multiple news categories from a single local SQLite database. Its existing `fed_monetary` and `bok_monetary` IDs remain unchanged to preserve historical fetch checkpoints.

## Built-in sources

| Source ID | Coverage | Publisher source |
|---|---|---|
| fed_monetary | US rate decisions / monetary policy | Federal Reserve |
| bok_monetary | Korean monetary-policy announcements | Bank of Korea |
| fed_all | US Fed press releases / banking regulation | Federal Reserve |
| bls_cpi | US consumer price index | US Bureau of Labor Statistics |
| bls_ppi | US producer price index | US Bureau of Labor Statistics |
| bls_employment | US employment situation | US Bureau of Labor Statistics |
| sec_press | US securities enforcement / regulation | SEC |
| bbc_business | International business and corporate stories | BBC News |
| bbc_world | Geopolitics and world news | BBC News |
| bbc_technology | AI / technology / industry news | BBC News |

The BLS, SEC and Federal Reserve feeds appear in publisher RSS catalogs. BBC news section feeds are independently indexed and may have usage restrictions. All network access remains contingent on the publisher making the feed available and allowing automated consumption. No live-source success is asserted by mocked CI tests.

**Korean general business / companies / geopolitical news is not yet comprehensively covered.** To add permitted RSS feeds from Korean publishers, supply a local JSON configuration with unique feed IDs. Validate terms before subscribing. Do not include paid article content or redistribute syndicated RSS descriptions.

## Run

```bash
market-event-collect --list-sources
market-event-collect --once --only bls_cpi,bls_ppi,sec_press
market-event-collect --once --feeds-file /absolute/path/to/private-feeds.json
market-event-collect --interval 300 --feeds-file /absolute/path/to/private-feeds.json
```

JSON config example: `config/feeds.example.json`. Example domains are placeholders, **not valid working publishers**. The collector requires HTTPS, limits config size and source count, and disallows reassignment of built-in source IDs. `--only` limits network calls. Database records persist the source ID, original article URL, raw feed bytes, SHA-256, publication and first-detection timestamps. Errors are isolated by source; conditional requests, duplicate rejection and restart persistence remain unchanged.

## Explicit limitations

- New RSS headlines are **not** yet mapped to economic event types. The Phase 3 extractor only recognizes rate-related terms and cannot classify CEO departures, earnings, tariffs or crises. That is Phase 5-B.
- The Phase 4 LLM currently only accepts rule-verified monetary-policy records. General market impact inference is Phase 5-C.
- An article appearing in multiple source feeds may create multiple source-scoped records; cross-publisher story clustering is future work.
- Polling does not guarantee breaking-news latency. Actual endpoint status, permissions, response format and capture quality require a live integration run on the user's Mac before claiming operational coverage.
- Feed bodies and descriptions may be copyrighted; keep them private, limit retention, and follow publisher terms.
