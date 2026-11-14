# Historical all-news research cohort — Learning-01

The research source is the **existing** archived `news.sqlite3`. It is not
just `economic_events`: **every** observed RSS article is included, even if
claim extraction found nothing or the article appears insignificant. A missing
event is not an automatically negative training label.

## Read-only Mac workflow

From a checked out and updated project:

```sh
bash scripts/verify.sh --update
market-event-historical-news-cohort \
  --as-of 2026-11-15T12:00:00+09:00 \
  --limit 1000 \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/historical-news-cohort-page1.json"
```

The above cutoff is an **example**, not a claimed audited snapshot.
Choose the historical cutoff explicitly; the export does not fetch news
or market bars. Use `--source-db` for a reviewed private archive copy.

If `hasMore` is true, take the reported `nextCursor` and rerun with the
same `--as-of` plus `--after-id <cursor>` and a fresh output file name.
Outputs are create-only, not overwrites, and can contain copyrighted
RSS titles/summary and source URLs: **retain locally, do not publish**.
If the source database does not exist, the tool fails; it never creates
a pretend historical archive or silently treats no data as success.

## Provenance and interpretation

- Uses SQLite mode=ro and query_only. The original capture payload SHA-256
  must equal its archived `captures.sha256`; a mismatch fails closed.
  This is a **local integrity test**, not cryptographic authenticity of
  the publisher or independent content verification.
- The first observed receipt (`articles.detected_at`) is the conservative
  information availability boundary. `published_at` is reported, not
  authenticated. An article is included only if receipt <= requested cutoff.
- `claimCountProcessedByCutoff` and `eventCountProcessedByCutoff`
  show *already completed* processing stages **at that cutoff**, but are
  descriptive metadata and **MUST NEVER** be used as live inference
  features unless availability at that exact prediction instant is proved.
  Missing table => `null`, not zero.
- If already populated, `storyGroupId` reuses the existing optional
  story index. A missing group is **not** an independently unique event;
  unindexed duplicates and later corrections must be reviewed separately.
- Always report `importanceHumanGold=UNKNOWN`,
  `observedMarketReaction=UNMEASURED`,
  `negativeSampleConfirmed=false`, `marketPriceLabel=null`,
  `trainingEligible=false`. Neither heuristic news classification nor
  no-event status is a measured importance or no-move label.
- `sourceCountsByCutoff` and `totalObservedByCutoff` describe what the
  stored archive **actually contains**, not real-world RSS recall.
  `excludedCounts.AFTER_AS_OF` includes later arrivals; malformed reported
  publication times are reported as `PUBLISHED_AT_UNTRUSTED`.

## Limitations and next handoffs

This is Learning-01's initial deterministic inventory and export code.
It still needs a measured run against the **real Mac archive**, source
coverage/replay and independent human-gold labels for both material and
nonmaterial news. Human review cannot be invented in CI.

Learning-02 must provide verified price/benchmark availability, event-time
joins, observation of no-material-reaction cases, and timestamp-safe labels
before any automatic training. Current historical event-study, intraday
windowing, rolling backtests and CPU baselines should be reused, not
rewritten. Only an approved read-only data handoff can use the separate
DonghakStockVision-owned market history; this repository does not alter
that project. This tool is not a market predictor or a retraining worker.
