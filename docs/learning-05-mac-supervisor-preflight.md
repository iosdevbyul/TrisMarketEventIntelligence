# Learning-05: Mac news research supervisor status and resource sampling

This adds a safe, **manual foreground** research supervisor around the
previously implemented finite NEWS-only experiment loop. It does not install,
enable, start or modify any launchd agent. The existing integrated news
collector and production `news.sqlite3` remain untouched.

## Safeguards

The supervisor uses a single-owner, create-exclusive `lock-file` so two
cooperating research processes cannot run together. If an old lock exists
after a crash, it **refuses to delete it**: inspect the old process
independently before clearing a stale lock. Each completed iteration writes
an atomic replace-only JSON checkpoint to the private `status-file`.
On a caught error it records `FAILED_REQUIRES_INSPECTION`, then releases
its own lock. The existing research journal remains responsible for
experiment deduplication across restarts.

A status reader reports the last timestamp, research counts, observed CPU
seconds and maximum resident memory plus status age. It never claims the
process is alive just because a file is fresh; even `RUNNING_CHECKPOINT_ONLY`
is a checkpoint, not an authenticated liveness probe.

The supervisor checks time spent, process CPU used, process peak RSS, and
cooperative stop flag **between** one-iteration research calls. CPU is
measured as current process user+system seconds; max RSS from macOS is
reported in MB using its native byte units (Linux uses KiB). These are
**sampled thresholds**, not hard kernel-enforced limits. A single blocking
training attempt can temporarily exceed them, and RSS is a cumulative
high-water mark. Set a separate OS-level process quota before relying on
this for unattended long-running operations.

## Mac command (for later use after real human-reviewed news exists)

The existing shell script has `supervise` actions:

```sh
bash scripts/news-learning-review.sh --update supervise --run \
  --cohort /private/news-cohort.json \
  --holdout /private/review-time-gated-holdout.json \
  --review /private/human-reviewed-news.json \
  --journal "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-research.sqlite3" \
  --status-file "$HOME/Library/Application Support/TrisMarketEventIntelligence/research-supervisor.json" \
  --lock-file "$HOME/Library/Application Support/TrisMarketEventIntelligence/research-supervisor.lock" \
  --stop-file "$HOME/Library/Application Support/TrisMarketEventIntelligence/research.stop" \
  --max-iterations 5 --max-seconds 3600 \
  --max-cpu-seconds 600 --max-peak-rss-mb 1024

bash scripts/news-learning-review.sh supervise --status \
  --status-file "$HOME/Library/Application Support/TrisMarketEventIntelligence/research-supervisor.json"
```

For a stop request use the existing `research-loop --stop STOP_FILE`
command. A stale stop flag must be cleared only after verifying the
process has already exited.

**No Mac run is requested for this PR**: tests use synthetic source and
research labels. There is still no genuine independent human gold or
production model quality evidence. With the unchanged five-option
experimental grid, the supervisor exits once those options are exhausted
and does not endlessly fabricate learning. Model export/promotion and
untrusted TEST gold access remain disabled.

Before any 24/7 activation, add independently checked operational limits
that can terminate a runaway process, process-liveness diagnostics, clear
stale-lock handling, guaranteed recovery and truthful external human
evaluation. Never feed this module prices, OHLCV, trading volume or returns;
stock-price predictions belong exclusively to DonghakStockVision.
