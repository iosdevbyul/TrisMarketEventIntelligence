# Unified market event integration — candidate ledger v1

Market Learning 1–2: https://trello.com/c/FniGiaYm and https://trello.com/c/GxK05N2i.

Read existing news and economic SQLite databases without modifying either. Reuse
multi_events, news_content_quality, StoryIndex identity functions, SEC canonical
identity and registered sources. Store immutable snapshot candidates and direct
candidate links in a separate SQLite DB. No network, model, pricing, or training.

Run (after CI passes):
```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
PYTHONPATH="$PWD/src" python -m tris_market_event_intelligence.unified_event_integration_cli \
  --news-db "$DATA/news.sqlite3" --macro-db "$DATA/economic.sqlite3" \
  --event-db "$DATA/unified-events.sqlite3"
```

Candidate records distinguish SEC acceptance, publisher-declared time, detected
time, evidence observation, macro observation period and vintage; first public
time remains unknown. SHA checks authenticate stored bytes, not semantic truth.
Official PDFs have separate raw and extracted-text evidence, not certified
complete releases. Unknown or incomplete source records remain visible as
metadata and cannot enter prediction/training. Existing historical versions are
not reconstructed by this first operational snapshot; source databases are read
in separate transactions. No point-in-time forecasting claim. Scan truncation
is explicit; default latest 10k articles and 50k numeric rows, adjustable.

Direct links cover SEC accession identity, URL, exact title/day, existing story
candidate, macro source/series/period versions, and six configured issuer alias
candidates with matching category and declared time within two days. No
transitive merging, no cross-source link asserted as fact, no inferred first
disclosure or causal direction. Links are bounded to 12 prior neighbors per
key; lexical classification and alias matches are incomplete. Numeric context
is not automatically attached to news based on temporal proximity. Explicit
macro-release-to-series mappings and semantic novelty/follow-up adjudication
remain necessary before full event integration acceptance. Actual Mac run,
coverage review and prediction integration still pending.
