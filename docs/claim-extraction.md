# Phase 5-D3 — Multiple independently described claims in one article

Run `market-event-claims --db data/news.sqlite3` after news discovery. Previously
ingested articles are processed without rescraping the publishers.

This additive pipeline does not overwrite the existing economic_events/impact tables
or their cursors. New `article_event_claims` rows represent **claims in source
headlines and summaries**, not authenticated real-world events. Claim IDs are
monotonic, versioned by `claims-v1`, and exposed through
`GET /api/v1/event-claims?after_id=0&limit=100`.

Extraction splits title and description into sentences/semicolon-separated clauses,
matches the existing ten event categories in each evidence span, and records
claimKind (REPORTED, FORECAST, RUMOR, ATTRIBUTED_REPORT, DENIAL,
HISTORICAL_REFERENCE), polarity (AFFIRMED/DENIED), and temporal context
(PAST/UNSPECIFIED). A published timestamp must not be assumed to be the
event's occurrence time. Per-article atomic checkpoints preserve idempotency.

Only current, unambiguous affirmative rate decisions in official Fed/BoK feeds
can inherit rule-supported VERIFIED; even this is **not** external primary-document
authentication. Third-party reports remain UNVERIFIED and official economic
headlines remain CANDIDATE. No LLM calls or trade signals are made.

## Limitations

- Sentence splitting is heuristic; abbreviations, nested quotation, implied
  negations, relative chronology and multiple occurrences of one event type
  need stronger language processing.
- A single clause may generate multiple category claims; they are not proof
  of distinct real-world incidents. Cross-article story identity remains
  in the separate conservative story index.
- Existing `economic_events` still contains one primary event per article,
  preserved for backward compatibility. Consumers needing multiple mentions
  should read the new claims endpoint.
- No 11 November or 13 November live-news detection claims follow from
  synthetic offline tests. Live accuracy evaluation and source confirmation
  remain required before Phase 6.
