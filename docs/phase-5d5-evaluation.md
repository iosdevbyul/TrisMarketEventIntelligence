# Phase 5-D5: source-linked news replay and operational collection audit

The fixed 10-item benchmark records genuine source URLs and published
headlines centered on November 13, 2026. NVIDIA's investor-relations release
was dated November 12, and is explicitly marked so it is never misrepresented
as a November 13 release. News examples are not automatically verified as
market-moving or representative of all news.

### Two metrics which must NEVER be conflated

1. **headline_recall**: deterministic extractor recognition from hand-supplied,
   source-linked article TITLES. It says nothing about whether a collector
   actually discovered those stories. No article bodies are fabricated.
2. **article_coverage**: matches those URLs or exact normalized titles against
   articles already present in a local SQLite database from collectors. This
   proves only stored data presence, not that a specific article was truly
   retrieved in an unattended run or that it describes the correct event.

Use:
```bash
market-event-evaluate --benchmark benchmarks/2026-11-13-headlines.json
market-event-evaluate --benchmark benchmarks/2026-11-13-headlines.json \
  --audit-db --db data/news.sqlite3
```

For an independent operational check, run permitted RSS/discovery and portal
collectors during a real capture window, then record fetch success/errors,
source timestamps, new articles, discovery coverage and claims. Always
distinguish collection, claim extraction and factual verification.

**Phase 6 remains BLOCKED until the 10 news cases are actually observed
from permitted network collection and extracted as expected, plus negative
controls and a larger independent validation sample pass agreed criteria.**
A green CI with fixture replay is not a real-world collector E2E.
The fixed benchmark is a diagnostic, not a comprehensive precision/recall study.
No local model execution or price forecasts are part of this phase.
