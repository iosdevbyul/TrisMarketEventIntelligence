# Optional Korean portal web scraping

**Opt-in only.** This is an additional HTML source alongside RSS, Naver Search API
and GDELT. It does not replace those collectors. Supported landing-page
candidates are Naver news economy, Daum news economy and Nate news economy.

Run manually on a locally installed package:

```bash
market-event-scrape --portals daum,nate --max-articles 5 --delay 5
market-event-scrape --portals naver,daum,nate --max-articles 10 --delay 3
market-event-claims
```

No periodic scheduler is enabled by this change. A future macOS schedule must
respect each host's current robots rules, provider conditions and retry limits.
This collector uses a stable identifying User-Agent, at most 30 article requests
per portal per run, at least one second between HTTP requests, no JS execution,
no login/captcha bypass, no cross-origin redirects, and fail-closed robots checks
on landing and article hosts. A robots 404/410 is interpreted as no declared
robots restrictions; this is **not** publisher permission to reuse copyrighted
article contents, so check the portal and original publisher's terms.

Articles are stored via the existing SQLite NewsStore with scraped HTML capture
and extracted text summary (at most 12,000 chars). This is intended for local,
non-redistributed research only. A future retention/clean-up policy should be
added before sustained collection; do not publish captured HTML or article
content via external APIs.

**Operational limitations**: HTML and selectors change. Naver/Daum/Nate may
block automation, provide no readable article body, or prohibit access through
robots/terms. Requests can therefore produce blocked_by_robots, error or
partial outcomes. No live portal success or headline recall is asserted by
offline mock CI. Only known portal article hostnames and path shapes are
eligible; sites linking out to original publisher hosts are not crawled.

Not a guarantee of coverage or legality for any particular host.
