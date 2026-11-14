# Stage 1 — One-shot real-data operational evidence packet

## Fixed product goal

Korean and US real-time material-news detection → point-in-time ticker
mapping → learn stock/benchmark one-minute responses → calibrated
future price-impact prediction. No stage is complete until actual runtime
evidence meets a documented acceptance bar.

## This integrated bundle

The optional, one-shot **read-only** `market-event-operational-evidence`
command reuses previously developed and tested local SQLite audits to
compose one report:

- **News collection history** (all bounded rows): provider-declared
  publication time → local RSS/Naver/GDELT receipt latency, unknown/future
  publication stamps, request duration, 304 responses, failed attempts.
- **News → economic event → local Gemma result checkpoint** (all bounded
  rows): receipt-to-extraction and extraction-to-persisted-model-result
  median/p95, unavailable model checkpoints and anomaly counts.
- **Recent observation window**: count first-seen articles by source and
  the exact reason for treating them as old, missing publisher timestamp,
  suspicious publisher time, or potentially fresh. The default window is
  the preceding 24 hours. These counts are *not* headline recall or
  completeness and do not measure publication time independent of
  the publisher's claims.
- **Source health**: actual most recent successful poll, API errors and
  staleness for the explicitly requested RSS source IDs. A recent
  successful HTTP poll alone does not prove any headline coverage.
- **KR/US market-session readiness**: optional, operator-supplied
  `--sessions` uses the existing explicit XKRX, XNYS, XNAS calendar.
  Without a supplied calendar, exchange-session state is **unknown**,
  not guessed from weekday or local clock.

All stages use the existing production news DB **without creating tables,
migrating schema, fetching network data, invoking Ollama or altering
historical records**. The command does not automatically launch a
collector or run a background process. An output JSON path is
create-only; it will not overwrite an earlier audit.

### What different states mean

- `RESEARCH_OBSERVATIONS_ONLY`: all required local checks were
  readable, sources had recent successful polls, and no requested
  exchange calendar date was unknown. This is **not** evidence of
  real-time news coverage or trained forecasting.
- `BLOCKED`: at least one required check or requested market date
  lacks usable evidence. The per-check reason is preserved, rather
  than replaced by zero values or invented successful timings.
- `UNVERIFIED`: the known local news DB is missing, symlinked or
  unavailable. The tool does not silently create an empty database.

A missing model-run table is separately noted by the lifecycle audit,
while a genuinely unavailable lifecycle table yields a
`UNAVAILABLE` entry. Legacy pre-PR #58 request-start timestamps
remain distinguishable only if the source record's capture version
can be established; the report does not rebrand them as authenticated
HTTP-response first-seen observations.

**Important:** the recent freshness section alone is scoped by the
window. The existing collection and model lifecycle audits summarize
all records inside their 50,000-row limits, not just the selected
24 hours. These are different populations and must not be directly
compared as though denominators matched.

## Operator commands

A local Mac with an already populated DB can inspect its own records:

```sh
market-event-operational-evidence \
  --db "/path/to/existing/news.sqlite3" \
  --sources fed_monetary,bok_all \
  --window-hours 24 \
  --freshness-minutes 15 \
  --output "/path/to/new-operational-evidence.json"
```

To include reviewed session routing, additionally supply
`--sessions /path/to/reviewed-sessions.csv` and
`--markets XKRX,XNYS,XNAS`. A missing calendar fails session
readiness explicitly. To compare reproducibly, pass a
timezone-aware `--as-of` timestamp.

The CLI exit code is 2 when blocked/unverified or the
explicit operator configuration is invalid. It does not expose
news text, API credentials, raw article URLs or model reasoning.

## Real progress versus remaining blockers

This PR gives the project one **actual-data audit interface** but
cannot populate its report from GitHub CI. The non-public SQLite
news collection and Intel Mac Ollama instance are on the owner's Mac,
not available inside the GitHub test runner.

The next actual-runtime verification must evaluate:
(1) recent first-seen articles by source for KR/US sessions,
(2) Gemma extraction-to-checkpoint and receipt-to-checkpoint delay,
(3) stale/blocked source proportions, and
(4) missing source dates, article completeness, market-minute data and
publisher-first-seen provenance. Honest unavailable results are
expected when capture evidence is insufficient.

A real news→ticker→price model still requires independently verified
KR and US adjusted minute-bar history from an authorized provider,
point-in-time company identity, calibrated cross-validation and
observable live alert latency. No stage is marked complete by this PR.
