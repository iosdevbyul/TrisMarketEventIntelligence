# Stage 2 — Isolated append-only post-hoc review ledger

This feature stores **reviewed research decisions**, not a news-time
verified event-to-ticker map. It is strictly opt-in and does **not**
run in the existing 15-minute RSS LaunchAgent.

## Workflow

First generate the **real, private**, reviewed issuer registry and
read-only ticker proposals as documented in
`docs/stage-2-reviewed-ticker-proposals.md`. Save the unmodified
proposal JSON outside Git and independently inspect its proposed
issuer associations. Supply an explicit human decisions CSV as
documented in `docs/stage-2-manual-ticker-review.md`.

Then import only to a **dedicated new review DB path**, never the
production `news.sqlite3`:

```sh
market-event-ticker-ledger \
  --ledger "/private/path/research-ticker-reviews.sqlite3" \
  --import-reviewed \
  --proposals "/private/path/proposals.json" \
  --decisions "/private/path/reviews.csv"
```

To inspect without writes:

```sh
market-event-ticker-ledger \
  --ledger "/private/path/research-ticker-reviews.sqlite3" \
  --list \
  --event-id 7
```

The DB is a new isolated SQLite file, initialized **only** by an
explicit valid import. Attempts to write to any already-existing
non-review DB are rejected **before** enabling write access. Review
decisions are SHA-256 bound to their exact proposal JSON and
persisted in an append-only table. Repeating the identical approved
or rejected decision is a no-op; attempting to replace a decision
for the same proposal hash, event ID, market and symbol fails.

Each stored row retains review time, reporter ID, original
article URL and original article observed time, review reason,
independent evidence URL, and the proposal digest. Read-only
listing verifies each stored JSON record digest. This detects
accidental corruption, not a malicious user modifying both the
row and digest; SQLite is not an authenticated transparency log.

**Important semantics:** reviewer approvals in this DB mean
`POST_HOC_RESEARCH_ASSOCIATION`, even when the originally cited
listing was already observable when the news arrived. The approval
itself happens later; no automatic EventTickerLink, benchmark,
point-in-time training input, return label or market direction
is created.

No historical issuer directory is bundled or invented. Import
requires real review files, which are not included in CI.
Private data must not be committed to the Git repo.

Further integration requires audited issuer and reviewer provenance
and an explicit independent source of news-time association
evidence before exposing links for walk-forward model training.
