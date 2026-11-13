# Phase 7-C — review triage without fact certification

The new read-only endpoint `GET /api/v1/claim-assessments?after_id=0&limit=100`
joins extracted claims to the latest append-only human review, where available.

Each response item includes a source-claim ID, `verification_level`,
`latest_review_status`, `review_revision`, `review_priority`,
`priority_reason`, `fact_confirmed`, and `analysis_suitability`.

The projection makes a strict distinction between a news report, a
rule-supported official feed and an operator's primary-document review.
`fact_confirmed` is always false because none of these alone constitutes
independent verification of the underlying event. A manual `CONFIRMED`
entry is reported as `HUMAN_REVIEWED_PRIMARY_DOCUMENT`, not automatically
promoted into the raw claim's extraction status.

`review_priority` is a transparent, deterministic human-work queue, **not**
economic importance, stock price impact, confidence or truth probability.
Monetary policy, indicators, regulation, geopolitical and supply-chain
reports require review first; forecasts/rumors and historical references
are deferred, while disputes/retractions are prioritized for attention.
`analysis_suitability` is descriptive only: it does not rewrite existing
model inputs or silently alter ongoing impact processing.

The API uses read-only SQLite, bounded claim-ID cursor pagination and no
external news/model requests. Reviews after the cursor was consumed require
explicit refresh as in the existing claim-review API. A missing review table
is treated as no review, not created on API read. No LaunchAgent or other
scheduled workflow is modified.

7-B Gemma 2 real-device inference is still unverified. The latest v5 run
rejected a too-short `rationale` with
`SCHEMA_STRING_TOO_SHORT_RATIONALE`. This is a separate model-output
quality issue; it does not affect this 7-C read-only projection.
