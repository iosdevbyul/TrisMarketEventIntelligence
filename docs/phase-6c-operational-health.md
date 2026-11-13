# Stage 6-C — offline health and failure checkpoints

Stage 6-B is manually verified on the user's Intel Mac for a one-shot LaunchAgent
cycle (eight successful RSS sources, exit code 0); automatic 30-minute recurrence,
sleep/wake, reboot and long-term coverage are **not yet verified**.

This increment adds no new daemon, service, credentials, model, scraping, or network
destination. Existing LaunchAgent configurations continue to call the same module
and do not require regeneration. Upgrade the installed checkout with `git pull`
when the merge is available (editable pip installs need no reinstall for module code,
but **new console scripts** require `python -m pip install -e .`).

## Offline checks

```sh
market-event-doctor
market-event-doctor --stale-after-minutes 90 --warning-mb 512
```

The doctor checks the latest atomic status JSON, verifies the last run is recent,
and executes SQLite `PRAGMA quick_check` on a read-only connection. It reports DB
file size and warns above the configurable threshold. It does not rotate, delete,
vacuum or truncate any data. A healthy or degraded run in the current 90-minute
window is distinguished from a failed or stale run; no proof of news recall,
independent event verification or actual uptime is inferred.

Return codes: 0 for OK/warning, 2 for error. JSON keys include `checks`,
`status`, `network_requests: 0`, and `modified_database: false`.
A stale observation is not automatically repaired; the existing launchd schedule
is expected to retry on its next tick, while an operator checks stdout/stderr.

## Exceptional collection errors

Previously a Python exception after the last successful cycle could leave the
old `healthy` checkpoint intact. The runner now records an atomic `failed`
checkpoint with `error_code` set to the exception **class** only (never the
raw error message or sensitive source data), and returns the existing exit code 2.
Normal per-source outages still use healthy/degraded/failed semantics.

Limitations: forced termination/SIGKILL, full disks, and failure to write status
can still prevent a checkpoint. The doctor flags stale status but cannot prove
why it is stale. File size warning excludes SQLite WAL and journal sidecars and
should not be treated as total disk use. No automatic deletion is implemented.
