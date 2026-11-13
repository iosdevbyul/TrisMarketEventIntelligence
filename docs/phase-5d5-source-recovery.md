# 5-D5 follow-up: actionable source recovery and safety

The previous public-network probe collected 35 Fed RSS articles, but none
matched the current event claim rules. **35 articles and 0 claims is not a
proof that the claim extractor is broken**: the RSS article sample must
be independently labeled before adjusting thresholds.

Changes:
- HTTP errors record `HTTP_403`, `HTTP_429`, etc. without leaking
  query URLs or authorization data.
- The official `bls_latest` RSS is an **additional** source, not an
  alias for `bls_ppi`. If the latter returns 403, it remains failed.
- BBC business/technology feeds are added to the public-network
  probe to test broader financial-news discovery.
- The GitHub probe retrieves only portal `robots.txt` (no article
  HTML). Naver/Daum/Nate permitted/blocked/unknown status is reported
  separately; an absent robots file is **not legal permission**.
- Article counts, summary coverage and current publication-period
  indicators help distinguish no-event input from missed events.
- The independent 10-news benchmark is not injected into collection.
  No headline replay score counts as live news discovery.

No automated rate-limit bypass, CAPTCHA circumvention, or paid AI calls.
Phase 6 remains blocked until independent local collection and
precision/recall evaluation pass.
