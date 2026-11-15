# Learning-05 — Real Mac NEWS inventory gate

**Goal:** stop adding speculative Learning-05 experiments and check whether
the actual private Mac news archive contains the pinned inputs required to
run the already-built research worker. This is a **read-only triage**, not a
model run, a performance evaluation, or approval to promote a candidate.

The command discovers only the app's existing private macOS storage:
`~/Library/Application Support/TrisMarketEventIntelligence/news.sqlite3`
and bounded JSON artifacts in that directory plus the optional direct
`research/` child directory. It never recursively searches home folders,
does not open external service connections, and does not ask for new secrets.
The SQLite inventory uses immutable query-only mode: it is an intentionally
read-only snapshot, so its count may lag live WAL changes. It does not
alter any production DB.

The inventory reports **only**:
- source DB presence and snapshot articles count;
- private cohort, human importance-review and time-gated holdout file counts;
- number of SHA-matched cohort / human review / holdout triples;
- when exactly one pinned triple exists, existing
  `news_learning_readiness` TRAIN human gold sufficiency and
  cross-split lexical overlap candidate counts (never TEST gold);
- one concrete next action, with flags identifying limitations.

No raw articles, sources, credentials, source paths, human identities,
human reasons or review rows are printed. No model calls, news collection,
network calls, training, or filesystem writes.

**Important:** This is a bounded inventory rather than a recursive index.
If existing private files live elsewhere, the tool may find zero matches.
A matching set is not proof of independent labels or semantic leakage
clearance; only a **single** unambiguous matching triple is auto-preflighted.
If multiple pinned research sets exist, the operator must select the
intended one via the existing explicit `learning-readiness` CLI.

## One necessary real Mac command

In the local cloned repository, on a clean main checkout:

```sh
bash scripts/news-learning-review.sh --update mac-inventory
```

If production data lives under a deliberately different directory, the
operator may explicitly pass `--data-dir /actual/private/data`. No need to
share a news DB, human review JSON or secrets publicly.

Send only the JSON summary back for the next development decision.
This provides evidence of *availability*, not a proof of learning efficacy.
Do not make new synthetic PRs just because private gold is unavailable.

## Limits

- `READ_ONLY_SQLITE_SNAPSHOT` is an immutable checkpoint view, which
  can lag active WAL; it is not an online collector health probe.
- Training sufficiency indicates only label counts at the time gate;
  editorial independent human truth remains unverified.
- Lexical duplication audit is not semantic, translated or cross-page
  leakage verification.
- No model installation, launchd activation, stock price/OHLCV/volume/
  realized return or DonghakStockVision pricing inputs are used.
