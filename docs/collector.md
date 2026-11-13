# Official RSS Collector (Phase 2)

Sources, from official RSS catalogs:

- Federal Reserve monetary policy releases: https://www.federalreserve.gov/feeds/press_monetary.xml
- Bank of Korea monetary-policy press releases: https://www.bok.or.kr/portal/bbs/P0000559/news.rss?menuNo=200690

## Run

```bash
python -m pip install -e ".[dev]"
market-event-collect --once
market-event-collect --interval 300
MARKET_EVENT_DB=/absolute/path/news.sqlite3 market-event-collect --once
```

The loop checks feeds sequentially and recovers from ordinary HTTP, invalid XML and oversize response errors at the next interval. HTTP 304 and ETag/Last-Modified conditional requests limit unnecessary downloads. Captures are stored as raw HTTP **response bodies**, with SHA-256 digest. Captures do not include full HTTP headers or linked article pages. Article summary is taken from the feed and may contain HTML; clients must escape it.

SQLite stores append-only first-seen article records keyed by (source, external_id) and deduplicated raw captures keyed by (source, sha256). Each successful fetch and its articles commit atomically. A duplicate feed entry retains its original detected_at timestamp. Fetch errors are recorded per source, and a failed feed doesn't stop other feeds. WAL and busy timeout are enabled.

The collector does not infer actual economic events or perform AI impact analysis. The feed's publication date might be absent or invalid; detected_at always records when the collector saw the entry.

Network and publisher restrictions can prevent fetching live feeds in CI. Automated tests use offline mock transports; live feed validation and Mac launchd operational checks are separate release gates. Do not rely on RSS polling for guaranteed real-time delivery.
