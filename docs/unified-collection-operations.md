# Unified collection operations

One-shot runs remain serial for bounded validation. Watch mode runs each selected job
in an independent thread; independent macOS LaunchAgents each run one job in their own
process. News polls 300 seconds after completion, SEC 900, economics 3600.
Per-job locks prevent overlapping unified workers of the same category on the same news DB.
SQLite shared writes can still contend. Legacy collectors do not honor these locks: stop them
before activation. Separate running markers retain interrupted-run evidence; existing
source-level deduplication handles replay. Latest reports are atomic; no claim of
exactly-once execution or recovery of a partially downloaded file.

Generate configurations with unified_collection_launchd_cli --repo --news-db --macro-db
--output-dir (absolute resolved paths); --write creates private 0600 plists in
~/Library/LaunchAgents, refuses overwrite and does not bootstrap. SEC contact must
already be in MARKET_EVENT_SEC_USER_AGENT; stored only in the private SEC plist.
Never commit or share that plist. Use a stable checkout and Python environment.
Explicit launchctl bootstrap gui/$(id -u) PATH for each of the three plists starts
them; bootout gui/$(id -u)/LABEL stops them. LaunchAgents restart after process exits
and load at user login, not before login; sleeping Mac cannot collect.
No Gemma, claims, forecast, price evaluation or OCR. Existing SEC six-company default
and economic coverage gaps remain. Live concurrent SQLite and login/restart validation
required before production acceptance.
