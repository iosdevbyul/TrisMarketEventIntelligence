# Learning-05 — bounded, cooperative NEWS research loop

This opt-in runner builds on the existing Learning-05 one-iteration CPU
classifier and its SQLite experiment fingerprints. It is **not** an installed
launch agent, continuously running service, or production model updater.

## Behavior

- Each `--run` invokes only existing NEWS title/summary + independently
  human-reviewed importance experiments on TRAIN and VALIDATION, with no
  stock prices, OHLCV, volumes, returns, or DonghakStockVision access.
- The caller selects hard limits of **1–1000 attempted iterations** and
  **1–86400 seconds of wall-clock budget**. Wall-clock expiry is checked
  between iterations (it does not forcibly interrupt an in-progress model).
  Candidate experiments use the PR #126 journal to deduplicate across
  restarts, so a repeated invocation does **not** learn the same settings
  again.
- `EXPERIMENT_GRID_EXHAUSTED` exits as `NO_NEW_EXPERIMENTS`, and
  `BLOCKED_INSUFFICIENT_HUMAN_GOLD` exits as
  `PAUSED_REQUIRES_NEW_HUMAN_GOLD`. Neither spins against unchanged
  local news or fabricates model improvement.
- Transient SQLite busy/locked errors retry with exponential backoff
  starting at 1 second, capped by `--max-backoff-seconds` (max 3600).
  All other storage, provenance and malformed-input errors fail closed.
- An optional ordinary local `--stop-file` triggers cooperative stop
  between iterations or during sleeps. The `--stop STOP_FILE` command
  creates the signal file with exclusive permissions. This does not
  kill an in-progress Python calculation or guarantee immediate shutdown.
- No background work, implicit network fetch, persistent model artifact,
  auto-promotion, or launchd scheduling is enabled.

## Mac command, only when independently reviewed real news exists

The existing `scripts/news-learning-review.sh` has a `research-loop`
subcommand. These are **examples**, not a request to run now.

```sh
bash scripts/news-learning-review.sh --update research-loop --run \
  --cohort /private/historical-news-cohort.json \
  --holdout /private/review-time-gated-news-holdout.json \
  --review /private/independently-reviewed-news.json \
  --journal "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-research.sqlite3" \
  --max-iterations 5 --max-seconds 3600 \
  --interval-seconds 1 --max-backoff-seconds 60 \
  --stop-file "$HOME/Library/Application Support/TrisMarketEventIntelligence/research.stop"

bash scripts/news-learning-review.sh research-loop \
  --stop "$HOME/Library/Application Support/TrisMarketEventIntelligence/research.stop"
```

Before restarting, move or delete an old stop file **after** verifying the
prior process has stopped. Never point the research journal at production
`news.sqlite3`. An unchanged dataset exhausts the five currently
supported smoothing-parameter candidates; the worker will not endlessly
improve itself just because the Mac stays powered on.

## Remaining for actual 24/7 learning

An independently validated real news gold corpus, broader candidate
search space, persisted candidate weights and reproducible inference,
protected evaluation gates, out-of-sample quality evidence, macOS resource
monitoring and an opt-in launchd service with safe pause/resume remain
separate tasks. Real deployment and continuous improvement are *not*
claimed by this PR or its synthetic CI.
