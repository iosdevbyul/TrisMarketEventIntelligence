# Phase 6-F — opt-in macOS runtime verification

The collector's macOS LaunchAgent has been observed with `runs = 2` and
`last exit code = 0` on the user's Mac, demonstrating at least two
completed launches in launchd's current record. The API LaunchAgent is
`running` and `/ready/claims` has returned 200 with actual claims.
This **does not yet demonstrate** reboot restoration or sleep/wake continuity.

## One-shot read-only operator audit

```sh
git pull --ff-only origin main
python -m pip install -e .
market-event-runtime-audit
```

The audit checks (a) database integrity and freshness via the existing doctor,
(b) both macOS LaunchAgent statuses via `launchctl print`, (c) API readiness
through `127.0.0.1:8100/ready/claims`, and (d) multiple distinct completed
cycle timestamps in the private JSONL history. It only reads state and sends
one loopback HTTP GET. It never invokes bootstrap, bootout, kickstart, restart,
writes data, changes a schedule, scrapes a portal or invokes AI. A collector
in `not running` state with a successful exit is expected between intervals.

Exit code 2 indicates error; warnings exit 0 but have `status=warning`.
An initial one-entry history is intentionally a warning even when launchd
has multiple launches. History is not backfilled from before PR #20.

## On-device sleep/wake and reboot matrix

1. Record `market-event-runtime-audit` and the latest timestamp in
   `last-run.json` before sleeping, then allow normal sleep and wake.
2. After wake, check `/ready/claims` and repeat the audit. Record whether
   launchd has resumed and the collector's cycle timestamp advances
   **after** the next scheduled opportunity. Do not assume missed ticks
   are replayed.
3. After a normal Mac reboot and user login, repeat the audit. If the
   API is missing, check `launchctl print gui/$(id -u)/com.tris.market-event-intelligence-api`.
   If the collector is missing, inspect the separate collector LaunchAgent.
   Check the two separate stderr logs before making configuration changes.
4. Distinguish API availability from collection continuity: a running API
   can serve older claims even while collection is stalled.
5. Real Mac sleep/wake and reboot cannot be validated by GitHub CI; CI uses
   mocked launchctl and loopback checks only. Neither test proves historical
   news coverage or predictive impact accuracy.

No public network binding, credentials, new database migrations or changes
to DonghakStockVision. The later news-analysis contract must provide
**completed verified and labeled analysis outputs**, not only unverified claims.
