# Stage 2 — Explicit human review of ticker candidates (research-only)

## Boundary

Stage 2's issuer candidate proposals are **not confirmed links**.
A second, independent human decision CSV can now be audited against
the *exact bytes* of an earlier `market-event-ticker-proposals` JSON
response. The review tool never reads or writes the live SQLite DB,
invokes the model, installs a service or creates training-ready links.

Human approval here means only **post-hoc research association**. Since
review happens after the original news arrival, that approval cannot be
presented as information known at news receipt, even when its external
company-identification material was available earlier.

The 15-minute RSS/economic-event LaunchAgent remains unchanged.

## Steps

1. Save the **unaltered** JSON result of `market-event-ticker-proposals`
   to `/private/path/proposals.json` in a private research folder.
   This must be the result of actual reviewed issuer registry data,
   not the synthetic test fixtures.
2. Calculate the raw file fingerprint:
   `shasum -a 256 /private/path/proposals.json`.
3. Independently inspect each proposed association and its original
   article URL. Record one explicit choice per reviewed candidate in a
   private CSV with **exactly** these headers:

```csv
event_id,market,symbol,decision,reviewed_at,reviewer_id,evidence_url,review_note,proposal_sha256
7,XNAS,MSFT,APPROVE,2026-11-14T14:00:00+00:00,operator-1,https://example.org/issuer,Reviewed identity; post-hoc only,REPLACE_WITH_ACTUAL_PROPOSAL_HASH
```

   This row is a **synthetic schema example**, not a real reviewed ticker
   or authenticated evidence. Replace all values with actual reviewer
   observations. `APPROVE` and `REJECT` are the only accepted choices.
   No default approval or automatic decision is applied.

4. Run a local read-only audit:

```sh
market-event-ticker-review \
  --proposals "/private/path/proposals.json" \
  --decisions "/private/path/reviews.csv"
```

The command rejects: malformed or oversized input, symlinked files,
unknown candidate identities, duplicated decisions, altered proposal
bytes (SHA-256 mismatch), extra CSV columns, absent reviewer/note,
non-HTTPS evidence, naive/future review timestamps, and reviews
dated before news arrival. It also refuses candidate reports claiming
independently verified listing or training readiness.

Each accepted approval is marked
`POST_HOC_RESEARCH_ASSOCIATION`; rejected candidates are
`REJECTED`. Unreviewed proposals remain unreviewed, not rejected.
The output includes a review count and a pinned proposal digest.
Neither raw review data nor the report is persisted by the tool.

## Do not confuse review with inference

Even a completed review reports:
- `actual_reviewer_identity_authenticated=false`
- `external_issuer_evidence_authenticated=false`
- `training_dataset_eligible=false`
- `auto_approved=false`
- `existing_database_modified=false`

No `EventTickerLink` CSV, benchmark symbol, or point-in-time
`evidence_seen_at` is emitted; it is impossible to accidentally feed
this post-hoc review directly into the pre-existing minute-event study.

Stage 2 still needs an authenticated historical issuer registry,
independent reviewer/provenance audit, a carefully constrained
reviewed-link persistence policy, and separate time-of-news
observability proof. No current or historical stock-price movement
is inferred from a company-name mention.

Files holding private source data and review decisions should remain
outside the Git repository. GitHub CI uses only synthetic fixtures.
