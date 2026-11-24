# Publisher expansion, phase 2

Seven additional RSS candidates: Asia Economy, Seoul Economic Daily, Edaily,
ETNews economy, ZDNet Korea, TechCrunch and MarketWatch. HTTPS candidate
endpoints are registered for explicit discovery only. Edaily HTTPS and
FeedBurner article host compatibility require actual local verification.
Blocked and subscription-only content must not be bypassed.

The registry now drives all four full-text SQL source checks, including
pending and retry counts. Existing operational defaults are unchanged.
No automatic daemon activation, no historical prediction backfill.
Use the existing media batch CLI with --discover --sources followed by
--run after confirming the discovery results. Source rights, semantic
completeness and price outcomes remain separate from processing coverage.

## Local diagnostic repair

Actual Mac discovery: six sources returned 260 rows, Edaily HTTPS reset.
Four sources completed 220 new analyses, plus 12 BBC; 930 total.
ZDNet 30 extraction failures; actual HTML contains articleBody and content-ID.
MarketWatch 10 failures; representative request returned HTTP 401 challenge.
Use exact cleartext Edaily RSS exception (unauthenticated discovery; no redirects),
retain HTTPS-only original articles, recognize ZDNet article body, record MarketWatch
401/403/451 as access-restricted and exclude from automatic and explicit repair.
No access-control bypass. Real recovery remains pending. HTTP RSS is not authenticated
and must not establish independently verified first-public availability.
