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

Mac recovery at ed8d86c: 30/30 ZDNet captures and analyses completed, all ten
MarketWatch articles classified ACCESS_RESTRICTED_HTTP_401, 960 analyzed rows,
878 unique versions, 82 duplicate rows, 93 audio/video exclusions. Edaily
collector received HTTP 406 despite direct HTTP 200 with default Accept.
Use the standard wildcard Accept only on the exact Edaily RSS endpoint;
actual collector acceptance remains pending. No authentication bypass.

Mac at 080bd8b: Edaily RSS healthy, 46 new records, all 46 body extraction
failures; one additional BBC analysis completed (961 analyzed, 879 unique).
Representative Edaily original returned HTTP 200, UTF-8 HTML, 126848 bytes,
with news_body and itemprop=articleBody plus ad/recommendation regions.
Require the observed semantic body, exclude known page furniture, preserve
raw bytes and fail closed if the body is absent. Real recovery pending.
