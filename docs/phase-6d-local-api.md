# Phase 6-D — opt-in loopback news API service

The collector is a 30-minute one-shot LaunchAgent. **Do not replace or edit it.**
This phase adds a separate API process listening strictly on 127.0.0.1:8100,
with the same user-owned SQLite database opened read-only by event query handlers.

The API has `GET /ready/claims` for checking article/claim tables specifically.
The existing `GET /ready` keeps its stricter requirement for event/impact
tables. Because the current operational collector extracts article claims, but
does not populate the separate economic event / impact tables, `/ready` may
legitimately be 503 while `/ready/claims` and `/api/v1/event-claims` work.

## Local test without launchd

```sh
market-event-api-local --db "$HOME/Library/Application Support/TrisMarketEventIntelligence/news.sqlite3" --port 8100
# In another terminal:
curl -fsS http://127.0.0.1:8100/health
curl -fsS http://127.0.0.1:8100/ready/claims
curl -fsS 'http://127.0.0.1:8100/api/v1/event-claims?after_id=0&limit=5'
```

## Optional LaunchAgent preparation (does not activate)

```sh
python -m pip install -e .
mkdir -p "$HOME/Library/Logs/TrisMarketEventIntelligence"
market-event-api-launchd --write
plutil -p "$HOME/Library/LaunchAgents/com.tris.market-event-intelligence-api.plist"
```

Activation is a **separate, user-approved** macOS action:

```sh
launchctl bootstrap "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.tris.market-event-intelligence-api.plist"
launchctl print "gui/$(id -u)/com.tris.market-event-intelligence-api"
curl -fsS http://127.0.0.1:8100/ready/claims
```

Disable or roll back this API service without affecting collection:

```sh
launchctl bootout "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.tris.market-event-intelligence-api.plist"
```

API logs: `~/Library/Logs/TrisMarketEventIntelligence/api-stdout.log`
and `api-stderr.log`. Check port conflicts before activation. `RunAtLoad`
starts at explicit bootstrap/login, while `KeepAlive` restarts on unsuccessful
exit; launchd may throttle repeated crashes. API does not spawn a second
collector, mutate the DB, invoke AI, enable network port forwarding or modify
DonghakStockVision. As with all loopback servers, other local processes may
access it; local binding is not authentication. Device-specific sleep/wake,
reboot and cross-process read/write behavior require on-Mac verification.
