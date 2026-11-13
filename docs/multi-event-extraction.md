# Phase 5-B — Multi-category financial event discovery

Supports MONETARY_POLICY, ECONOMIC_INDICATOR, EARNINGS, CORPORATE_EVENT, REGULATION, GEOPOLITICAL, SUPPLY_CHAIN, INDUSTRY, SERVICE_INCIDENT, MARKET_MOVEMENT. One primary category per article plus bounded additional tags.

Claim kind is REPORTED, ATTRIBUTED_REPORT, FORECAST, or RUMOR. Third-party news remains UNVERIFIED; official agency CPI, PPI, employment and regulation feed headlines remain CANDIDATE until primary-source evidence is independently checked. The only VERIFIED status retained is text-rule-supported monetary policy decisions from official Fed/BoK policy feeds, **not** external document authentication.

Extractor version `multi-v2` generates append-only rows, retaining old `policy-v1` results. A one-to-one `event_metadata` table records claimKind, tags and verificationNote without changing the original `economic_events` layout. The existing API returns these as additional fields; legacy events without metadata return null/empty values.

Limitations: matching is intentionally conservative and can miss stories or confuse a reported claim with truth. Corrections, negation, temporal attribution and cross-publisher clustering are not solved. Headline classification is not factual verification, event direction, ticker attribution, price prediction or a trading signal. Models still only accept monetary policy events until Phase 5-C. Live-source E2E remains outstanding.
