# Stage 6-A — macOS local collection cycle and opt-in launchd setup

This is an **initial implementation**, not verified always-on production readiness.
Current external network checks were run on a GitHub Linux runner, NOT the
user's Intel Mac. Real recall/precision, macOS sleep/wake, storage retention,
LaunchAgent persistence and data integrity over weeks remain unverified.

## Manual one-shot validation

Install this package in a local Python 3.11+ virtual environment, then run:

```bash
market-event-cycle --dry-run
market-event-cycle
cat "$HOME/Library/Application Support/TrisMarketEventIntelligence/last-run.json"
```

One cycle fetches only previously reachable **RSS** sources: Fed policy/all,
BoK all/statistics/corrections, BBC business/technology/world. It does not invoke
GDELT (observed HTTP 429), BLS (403), Naver API, portal HTML scraper, or the LLM.
After network requests it extracts new claims and indexes stories from the same
SQLite database. The run is guarded by a local flock lock to prevent overlap
and reports healthy/degraded/failed in an atomic private JSON status file.
A zero-new-article run with valid 304/200 responses can still be healthy.

## Optional future background activation

Generate a LaunchAgent plist without loading it:

```bash
market-event-launchd --interval 1800
market-event-launchd --interval 1800 --write
```

The file path is:
`~/Library/LaunchAgents/com.tris.market-event-intelligence.plist`.

Once the user explicitly approves enabling a scheduled job *on their Mac*,
create the logs directory and use standard launchctl commands:

```bash
mkdir -p "$HOME/Library/Logs/TrisMarketEventIntelligence"
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.tris.market-event-intelligence.plist"
launchctl print "gui/$(id -u)/com.tris.market-event-intelligence"
# Disable / rollback:
launchctl bootout "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.tris.market-event-intelligence.plist"
```

Installation is **not** performed in CI or by the application. A plist contains
absolute Python/db/status paths but never API secrets. LaunchAgent does not run
at load time. Schedules run only while the user's login session is active;
sleep/wake and missed runs must be checked on the real Mac.

The 10 historic news URLs still have **0/10 collector presence** in the
one-time GitHub probe. The real-time feed sample (436 raw articles, 38 claimed
events) does NOT prove general news recall or event precision.
An independent 100-article human review is still required.

No user data is uploaded, no trades are placed, no Mac settings are changed
by generating a plist, and no AI inference is invoked.
