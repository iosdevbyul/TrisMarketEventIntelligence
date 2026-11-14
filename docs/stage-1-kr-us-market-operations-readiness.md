# Stage 1 — Integrated KR/US market-session operations readiness

## Non-negotiable product goal

**Korean + US real-time material-news detection → point-in-time affected
ticker binding → minute-price reaction learning → future impact prediction.**
This package extends the already merged bounded RSS→event→optional Gemma
pipeline, rather than adding another independent collector.

## What the package delivers

1. **Source-supplied exchange-session plan** for XKRX, XNYS and XNAS, using
   the existing exchange-session contracts. It respects the exact operator
   supplied local trading date, timezone, DST, early close and closed days.
   Missing days are `CALENDAR_UNAVAILABLE`, never silently guessed from
   weekdays or classified as a holiday.
2. **Operational plan**: `RUN` if every requested market date is known and
   at least one market is currently in session; `SKIP` if all requested
   markets are known but outside regular hours; `BLOCKED` if any required
   calendar day is unknown. The plan never invents market opening or
   economic indicator release times.
3. **Existing source telemetry integration**: a read-only preflight combines
   the session plan with actual local RSS source-poll evidence. Missing DB,
   absent source telemetry, stale successful poll, denied or rate-limited
   source produces a `BLOCKED` readiness result while the market is open.
   A successful 304 is a valid successful *poll*, not evidence of complete
   market news coverage.
4. **Opt-in market-session gate on the integrated pass**: passing
   `--sessions` to `market-event-integrated` activates a gate evaluated
   **before any network request, DB initialization or model lookup**.
   A `SKIP` is a normal no-work result; `BLOCKED` fails closed.
   Without `--sessions`, its existing collection behavior is unchanged.
   The optional model still requires explicit `--enable-local-analysis`
   and the existing live freshness and source health guards. No background
   launchd job is automatically installed or started.
5. **CLI + full synthetic tests**: an offline `market-event-ops-readiness`
   command prints the source/session preflight or writes a create-only
   report. CI checks calendar parsing, DST and early closing, source
   failures, missing dates, duplicate days, no-network skipped runs and
   absence of implicit model calls.

## Reviewed calendar contract

Provide an operator-reviewed, local CSV with exactly the columns:

```csv
market,date,status,open_at,close_at
XNYS,2026-11-16,OPEN,2026-11-16T09:30:00-05:00,2026-11-16T16:00:00-05:00
XNAS,2026-11-16,OPEN,2026-11-16T09:30:00-05:00,2026-11-16T16:00:00-05:00
XKRX,2026-11-17,CLOSED,,
```

**These rows illustrate the format, not authenticated exchange dates.**
A closed row needs empty open/close fields; an open row needs
timezone-aware real session boundaries. Only days actually included
in the supplied file can be evaluated. CSV is bounded to 1 MiB and
2,000 dates; symlinks and malformed rows are rejected.

### Operator commands

Inspect session routing and source health with no network or writes:

```sh
market-event-ops-readiness \
  --sessions /path/to/reviewed-sessions.csv \
  --markets XKRX,XNYS,XNAS \
  --sources fed_monetary,bok_all
```

Preview market-window decision without touching the DB or contacting
the local model:

```sh
market-event-integrated --dry-run \
  --sessions /path/to/reviewed-sessions.csv \
  --markets XKRX,XNYS,XNAS \
  --sources fed_monetary,bok_all
```

Then optionally run a single market-focused pass:

```sh
market-event-integrated \
  --sessions /path/to/reviewed-sessions.csv \
  --markets XKRX,XNYS,XNAS \
  --sources fed_monetary,bok_all
```

An explicitly opted-in analysis pass additionally needs the installed
local Ollama model and `--enable-local-analysis`. Existing limits
(maximum three new eligible incidents per pass, freshness/source
health checks and no duplicate checkpoints) remain in place.

All approved RSS feeds remain eligible when **either** requested market
is open. A Fed or Bank of Korea release can affect both countries:
the source's publishing location is **not** an instrument or geography
filter. Sources are kept unchanged rather than lost because only one
market is open.

### Known limitations and correct interpretations

- An exchange's session calendar is only **operator supplied** here.
  Actual holiday/early-close source provenance has not been independently
  checked; a fabricated calendar can still make incorrect decisions.
- Normal after-hours and premarket **news collection must remain
  available via existing collection/discovery commands**. This market
  session gate is a *separate opt-in run mode*, never a replacement
  for the news archive or after-hours strategy.
- Market-open + healthy polling is insufficient to claim immediate
  breaking news coverage or headline completeness. Per-source signal
  latency and real Korean/US market sessions must be verified.
- Real KR/US adjusted minute bars, tickers with validity periods,
  local Gemma speed under load, and data provenance are unverified.
  `ready_to_train_or_predict` and `kr_us_realtime_verified`
  are **always false** at this stage.

## Integration verification

The repository's existing Python 3.11/3.12/3.13 GitHub CI runs all
tests. This package is a GitHub code change, so the user does not need
to run a separate Mac test. Actual live operator validation is still
a prerequisite for completing the **first** of the user's four
product stages.
