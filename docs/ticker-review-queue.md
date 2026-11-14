# Read-only ticker review queue reconciliation

Following PR #73's isolated append-only decisions and PR #74's full integrity
audit, inspect the **exact candidate bundle** that still requires human review:

```bash
market-event-ticker-review-queue \
  --proposals /path/to/private/proposals.json \
  --ledger /path/to/private/reviews.sqlite3
```

The command verifies the complete isolated ledger before reconciling. Only
decisions whose `proposal_sha256` matches the input file's actual bytes count.
The deterministic output reports `PENDING_HUMAN_REVIEW`, `APPROVED_POST_HOC`
or `REJECTED` per event/exchange/symbol, with aggregate counts. Review decisions
from old or differently generated proposal files never silently transfer.
A missing or uninitialized ledger fails closed; initialize it through an
explicit reviewed import, not an automatic operation.

**No auto-approval**: All decisions must still be supplied by an actual human
using `market-event-ticker-review` and then imported with
`market-event-ticker-ledger --import-reviewed`. No upstream provider listing
snapshot, news body, reviewer decision or ticker is manufactured.

The output is **not** independently authenticated issuer identity, verified
historical linkage, a time-of-news stock relationship, a return label or a
training-ready dataset. No production SQLite, LaunchAgent, inference model,
network provider or local operation state is changed. Read-only reconciliation
cannot demonstrate that real Nasdaq/KRX listing files or reviewers exist.
