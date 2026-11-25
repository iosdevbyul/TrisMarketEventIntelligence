# Unified collection-only operations

Run unified_collection_cli with existing --news-db, distinct --macro-db, and --output-dir.
Without --watch, execute one bounded pass. With --watch, schedule news every 300 seconds,
SEC every 900 seconds and economics every 3600 seconds after each job finishes.
Jobs run serially; slow economic refresh may delay subsequent news or SEC polling.
Use --job to select specific categories. News covers the registered publisher feeds
and text-only capture; SEC defaults to the existing six CIKs, not the entire EDGAR universe.
Configure actual MARKET_EVENT_SEC_USER_AGENT locally; never commit it.
Use --cik repeatedly to select issuers, --sec-days to set the rolling 1–30 day window.
No inference, claim extraction, forecasts, price evaluation or OCR is scheduled.
Per-job latest JSON checkpoints preserve partial and blocked states.
The supervisor lock prevents overlapping supervisor instances, not legacy standalone workers.
Stop existing independent collectors before activation. No daemon is installed by this PR.
Economic operational source gaps remain as documented in PR #282.
