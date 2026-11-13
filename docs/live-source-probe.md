# Live collection E2E probe (5-D5, not yet accepted)

An explicit bounded live probe runs a bounded set of official and mainstream RSS requests and one GDELT
query on a disposable SQLite database. It then runs the actual claim extractor
and story index and compares collected article URLs/titles against the
source-backed November 13 benchmark. Reports contain only counts/statuses:
no raw news payloads, scraped page text or credentials are uploaded.

Run via GitHub Actions **Live Source Probe** (manual workflow dispatch) or
`python -m tris_market_event_intelligence.live_probe --output live-probe.json`.

The PR that adds this workflow performs a one-time initial probe. If HTTP
requests are rejected, fail or return no articles, the outcome is recorded;
a skipped/broken remote source is not reclassified as a successful observation.

**A GitHub runner is not the user's Intel Mac.** This test does NOT
establish that the Mac can reach the source, that Naver/Daum/Nate HTML
crawling is permitted or works, or that all 10 historical stories were
discovered during normal unattended operation. There is no scheduler enabled by this probe and
no AI inference. Phase 6 remains BLOCKED until sustained independent
source collection and event recall/precision are validated.

## Korean discovery coverage

Three separate Bank of Korea official RSS feeds (all press releases,
economic statistics, and press corrections) complement the existing policy
feed. These official HTTPS feeds are listed on the Bank of Korea's published
RSS directory. They improve *potential coverage*, not proven recall.
BBC World RSS adds international-news coverage to the real network probe.
The BLS 403 and GDELT rate-limit/timeout statuses continue to appear as
failures. No provider block is bypassed.
