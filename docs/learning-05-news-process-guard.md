# Learning-05: isolated Mac news research child with OS CPU limit

This is an opt-in **foreground-only** guard around the existing NEWS-only
research loop. It does not install, launch, or change any macOS LaunchAgent,
existing news collector, production API, model or SQLite news archive.

## Protection model

- Each launch creates an exclusive lock. A second attempt fails. A lock
  left after a crash **must be inspected manually**; this command never
  assumes a stale lock is safe to delete.
- A subprocess runs the existing research loop with its original
  immutable all-news cohort, time-gated human review, private journal and
  protected TEST controls. The supervisor does not access TEST labels.
- On macOS/Linux the child receives an OS `RLIMIT_CPU` quota. The hard
  limit is one second above the soft CPU limit to allow kernel escalation.
  It applies to process CPU time, not wall time and not its descendants.
- The parent polls the direct child's RSS via `ps -o rss` and sends SIGTERM
  to the child's process group if the observed RSS reaches the configured
  threshold. It also terminates at the wall-clock budget and on a stop-file
  request. RSS polling is **not** a hard memory limit: transient spikes
  between samples and unmeasured child descendants remain possible.
  If RSS sampling becomes unavailable while the child is alive, it stops
  rather than running without an observation.
- A bounded private JSON status file is replaced atomically on start and
  exit, noting the exit code and whether the guard or child stopped the
  run. The recorded status is historical; it does not certify live process
  status, completed research accuracy or model promotion.
- Existing journal fingerprints preserve completed experiments across
  restarts; no research data is fabricated, and no infinite learning loop
  is started. The watchdog cannot guarantee cleanup if the parent itself
  is forcibly killed, nor does it handle stale-lock recovery automatically.

## Mac commands, for later real human-reviewed news

These commands are **examples**, not a request to run now:

```sh
bash scripts/news-learning-review.sh --update guard-research --run \
  --cohort /private/news-cohort.json \
  --holdout /private/news-holdout.json \
  --review /private/human-reviewed-news.json \
  --journal "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-research.sqlite3" \
  --status-file "$HOME/Library/Application Support/TrisMarketEventIntelligence/guard-status.json" \
  --lock-file "$HOME/Library/Application Support/TrisMarketEventIntelligence/guard.lock" \
  --stop-file "$HOME/Library/Application Support/TrisMarketEventIntelligence/research.stop" \
  --max-cpu-seconds 30 --max-wall-seconds 300 \
  --max-rss-mb 1024 --max-iterations 1

bash scripts/news-learning-review.sh guard-research --status \
  --status-file "$HOME/Library/Application Support/TrisMarketEventIntelligence/guard-status.json"
```

An already-created stop file prevents the run. Request cooperative stop
with the existing `research-loop --stop STOP_FILE` action. Before
removing a lock or stop flag, independently confirm the previous process
has exited. Exit status of the child does **not** establish independent
news intelligence improvement.

The five-option CPU research space remains finite. Actual autonomous
24/7 exploration requires truthful human-gold news validation,
resource-bounded new research strategies, privacy-safe backups, operational
liveness checks and explicit opt-in. This PR intentionally activates none
of those services and uses no stock prices, OHLCV, trading volume, realized
returns or DonghakStockVision components.
