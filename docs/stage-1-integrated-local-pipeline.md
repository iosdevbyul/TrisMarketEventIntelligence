# Stage 1 — Integrated bounded RSS → event → local Gemma pass

## Core product direction

1. Detect financially material news from KR and US sources in real time
2. Link those incidents to affected listed securities with point-in-time evidence
3. Learn historical 1/5/15/30/60-minute market-relative returns
4. Predict direction/magnitude/uncertainty after real, out-of-period validation

This milestone **integrates existing components** rather than recreating them.
It does not complete any full stage or claim realtime KR/US news coverage.

## What runs in one pass

`market-event-integrated` is an explicitly invoked bounded one-shot
pipeline. It reuses the existing `run_cycle`, which collects a selected
set of approved RSS sources, stores the original responses, deduplicates
news, and builds claims and story groups. It then uses the existing
`EventProcessor` to extract economic incidents (up to 1000 articles per
pass). It checks existing source-health telemetry and optionally uses the
existing `ImpactProcessor` to analyze **at most three candidate incidents
from articles first inserted in this pass**.

The integrated pass has a separate nonblocking pipeline lock covering
all phases, so two invocations cannot run AI work concurrently. The
individual collector retains its original lock and status. Existing
event/impact persistence keys enforce idempotency; duplicate feed
articles do not create repeated local model calls. Sources that are
stale or whose API is failing cannot be considered ready.

## Safety and opt-in boundaries

By default, **no AI or local Ollama call happens**. The only mode that
can call a model is a user-explicit `--enable-local-analysis` with
an already installed model configured by `MARKET_EVENT_LLM_MODEL`.
Before starting collection, the CLI checks the local Ollama tags endpoint
on loopback; remote inference is not supported by this command. The
model is never called for old or source-unhealthy news: the new live
freshness guards are mandatory in the integrated flow.

The newly inserted article-ID lower bound is captured **before**
the collection cycle, and analysis selects eligible economic events
from those articles only. This is not a general queue draining service.
Old research backlog cannot be silently sent to the model. Model
analysis budget is capped at three incidents for each pass. Skipped,
stale or publisher-time-unknown incidents remain uncheckpointed and
are not falsely marked processed.

Collection failures block downstream analysis; a partial source failure
is reported as degraded. Missing telemetry, unavailable model, analysis
errors and deferred events are surfaced separately. The status file is
atomic and contains only bounded operational numbers and event IDs,
not full article bodies, API keys, raw URLs or model text. An unchanged
historical news research workflow remains available separately.

## Commands (do not need to run these for GitHub CI)

Validate configuration with no network, filesystem writes or model calls:

```sh
market-event-integrated --dry-run --sources fed_monetary,bok_all
```

One bounded RSS collection and event extraction, no AI:

```sh
market-event-integrated --sources fed_monetary,bok_all
```

Optional **explicit on-device** Gemma inference in a supervised run:

```sh
MARKET_EVENT_LLM_MODEL=gemma2:latest market-event-integrated \
  --sources fed_monetary,bok_all \
  --enable-local-analysis \
  --live-freshness-minutes 15 \
  --source-success-minutes 15
```

The command does not install launchd agents, start an Ollama daemon,
retry infinitely, or promise that an article was globally published
for the first time at the RSS timestamp. RSS-only is intentional:
Naver and GDELT are independently quota/credential-dependent discovery
paths, and the existing bounded production cycle deliberately excludes
them. Adding those paths to a supervised unified operator run is a
later explicit, quota-aware decision rather than silently broadening
network calls.

## Verification

GitHub Actions runs the package lint and all tests across Python
3.11, 3.12 and 3.13. Synthetic HTTP responses and an injected fake
local model prove the entire collector→extraction→analysis flow,
idempotency, stale publication exclusion, source failure, lock
behavior and no-side-effect CLI dry run. No Mac-specific test is
required to validate these changes.

**Not yet verified:** true market-session latency, all missed headlines,
ticker-link accuracy, real KR/US adjusted minute bars, and local
Gemma CPU runtime speed. Production completion of stages 1–4 remains
open until real observations establish those conditions.
