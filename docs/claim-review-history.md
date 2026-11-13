# Phase 5-D4 — Manual evidence reviews and correction history

This phase adds a conservative evidence-review log; it does **not**
automatically verify article claims. The source articles, extracted claims,
their original status and raw captured bytes remain unchanged. Reviews are
append-only and use monotonically increasing per-claim revisions.

## Status transitions

The local reviewer may record UNDER_REVIEW, CORROBORATED, DISPUTED,
RETRACTED, or CONFIRMED. Every decision requires a stored evidence article,
a **literal excerpt** from that article, an evidence kind and a reason.

- CORROBORATED records supporting reporting; it is not proof of truth.
- DISPUTED requires a DENIAL or CORRECTION.
- RETRACTED requires a CORRECTION (source correction must be evaluated by reviewer).
- CONFIRMED requires PRIMARY_DOCUMENT, source ID mapped to known official
  agency, HTTPS document URL on its agency domain, and **explicit human review**.
  The check proves neither that the source actually published the document
  nor that it pertains to the original event; those checks remain human tasks.
  A Fed source cannot be used to confirm an unrelated CEO story simply by
  recording a review. Do not interpret CONFIRMED as automatic authentication.

Future phase 5-D5 should improve cross-source matching, evidence authenticity
and whether a correction/denial actually contradicts the same subject/action.

## CLI (local write access only)

```bash
market-event-review --claim-id 123 --status DISPUTED \
  --evidence-article-id 456 --quote 'Company denies the report' \
  --kind DENIAL --reason 'Publisher reported a direct denial' \
  --expected-revision 0 --operation-id review-claim-123-1
```

Replaying the identical operation ID is idempotent. Using an out-of-date
`--expected-revision` fails rather than overwriting another review.

## Read-only API

`GET /api/v1/claim-reviews?after_id=0&limit=100` returns the current
manual review of each claim (cursor is the claim ID).
`GET /api/v1/claim-review-history?after_id=0&limit=100&claim_id=123`
returns immutable review revision records (cursor is the revision row ID).
No HTTP write endpoint is exposed.

No LLM, remote account system, or news source requests are added. No automatic
event verification is claimed by CI; all fixtures are fictional.
