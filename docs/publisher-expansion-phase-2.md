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
