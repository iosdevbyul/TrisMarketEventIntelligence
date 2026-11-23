# Market Learning Stage 1 — Point-in-time market-outcome contract

Status: implementation candidate, not production deployment or proof of forecasting accuracy.
Owner: TrisMarketEventIntelligence. Governing mission: [PROJECT_CHARTER.md](../PROJECT_CHARTER.md).

## Decision: one observed direction is not one causal truth

The outcome of a news-related market forecast is defined at an explicitly sealed
**target, benchmark, clock cutoff, observation horizon and direction deadband**.
Economic meaning (Gemma or another LLM) remains a separate hypothesis. Observed
absolute direction and benchmark-relative return are facts about sampled prices,
not certified causes attributable to the news. In particular, a reprint is not
automatically scored as a new causal event, even when its next return is positive.

The new independent contract lives in
`src/tris_market_event_intelligence/point_in_time_outcome.py`. Existing
`event_study.py`, historical daily-close pipelines, #252 research artifacts,
automatic collection worker and production SQLite tables remain unchanged.
This change is the **scoring-policy foundation**, not source collection, verified
historical forecasts or learning-rule promotion.

## Clock and identity contract

- `published_at`: publisher-stated article publication time, timezone aware;
  this may be incorrect, revised or different from the underlying event time.
- `first_public_at`: earliest **independently supported** public availability
  of that piece of information. Unknown is `None`; never infer from detection.
- `detected_at`: time the system saw the article. A live prediction cannot
  claim to have happened before this time.
- `forecast_at`: immutable model prediction's seal/cutoff. In historical replay,
  its information-availability claims require **separate evidence**; historical
  run timestamps do not establish original foresight.
- `family_id`: identifier for related event reports. Store reprint,
  substantive follow-up, correction, unknown as separate novelty states.
  Related article predictions stay individually auditable, but their targets
  cannot count as independent holdout cases without event-family controls.

In the conservative outcome calculator, a target's baseline is the most recent
completed price bar with `bar_end < first_public_at`; never use the opening
price for a 14:00 report. The exit is the most recent completed price bar after
`forecast_at` and at or before the **precommitted** `horizon_end`.
Both prices require specified age bounds, and benchmark bars must be aligned
within a specified tolerance. A bar with `bar_end == first_public_at` is
deliberately *not* eligible as a before-news baseline.

**Important limitation:** returning a prior close for a after-hours release
requires an explicitly justified anchor-age threshold. This module does not
pretend to infer per-market session schedules, auctions, halts, corporate
actions or trading permissions. A verified session calendar, source-specific
close conventions, minute-bar availability and adjusted-price provenance
must be attached by the ingestion/validation integration before operational use.

A forecast generated after publication is evaluated with TWO distinct baselines:
(a) the latest bar before first public availability, for an event-window description;
(b) the last completed bar no later than the sealed forecast cutoff, for the
actual post-seal forecast-direction score. This latter price must not be stale
and cannot be in the future relative to the seal. Neither return is a
causal news impact. Historical replay remains non-prospective without
independently demonstrated information availability.

## Output fields and fail-closed behavior

`observe()` reports `absolute_return`, `benchmark_return`,
`relative_return`, `actual_direction`, `forecast_return`,
`forecast_actual_direction`, `forecast_anchor_at`, `direction_match`,
`event_window_direction_match`,
`novelty`, `family_id`, and permanently
`causal_attribution=NOT_ESTABLISHED`. The absolute direction uses a
precommitted deadband; a positive market-adjusted performance is **not**
silently substituted for a negative absolute return.

Unverified sources, missing baseline, missing exit, missing benchmark, and
stale bars produce explicit unscored statuses with null metrics. They are never
counted as correct neutral/zero returns. A full evaluation report must show
such exclusions and denominator effects, not just scored matches.

## Required user counterexamples, implemented in tests

1. A share gains +10% before 14:00 and closes +2% versus the original
   daily opening price: source-aligned event window is 110→102 or -7.27%,
   not "UP" from the day's open.
2. Yesterday's adverse report is reprinted today and prices rebound:
   `REPRINT` and the common `family_id` remain visible; the positive
   observed price move does not invalidate the original adverse meaning.
3. After-close reports require explicit next-session bars and nonstale anchors.
4. Unverified release time, missing/stale minute bars and missing benchmark
   do not fabricate a score.
5. Bars stamped exactly at the publication time do not serve as pre-release
   anchors; future outcome bars must be after the sealed forecast.

## Remaining Stage 1 integration gates — do not mark stage Done yet

- Establish the source-verified original information-release timeline and
  handling of revised titles/redirects/time uncertainty. Ensure differing
  sources and follow-up reports map to event families without speculative
  causal identity.
- Specify market calendars and separate 30m, session-close, 1/3/5-session
  horizons, per-exchange anchor/max-age and suspended/after-hours behavior.
- Audit the new post-seal direction score on genuine timestamped sources, and
  ensure retrospective examples are not relabeled as live forecasts.
- Match source-verified real intraday price quotes, benchmark and provenance;
  exclude adjusted/unadjusted mismatches. Keep missing data and causal
  ambiguity visible.
- Run historical price cases after the prediction is sealed; audit outcome
  isolation and independent event-family denominators before a learning
  experiment or any "accuracy improved" claim.

## Implementation safety

This is an additive pure-Python module with synthetic regression fixtures.
It does not call Ollama, fetch prices, modify SQLite, start a background
worker, recalculate historical experiment records or change trading state.


## Existing verified-format market source integration (PR #263)

The opt-in `point_in_time_source_validation_cli` integrates the existing
`market_minute_source.audit_market_source` with
`exchange_sessions.SessionCalendar`, `TradingHalt` and the new dual-anchor
`point_in_time_outcome.observe` evaluator. No feeds or local Mac files are
downloaded by this command. Existing collection and operational databases are
not modified.

Input contracts:

- `--source-manifest` uses the existing `minute_source_v1` JSON pointing
  to SHA256-pinned, adjusted minute CSV files. The manifest identifies an
  original provider and instrument kind. Digest/schema matching proves
  **file integrity only**, not exchange authenticity or point-in-time
  existence of these observations.
- `--sessions` uses the existing `market,date,status,open_at,close_at`
  CSV; `--halts` optionally uses
  `market,symbol,start_at,end_at`. Dates absent from the supplied calendar
  remain UNKNOWN and are not synthesized. Source authenticity of the calendar
  is separately unverified.
- `--predictions` contains:
  `article_id,family_id,published_at,first_public_at,detected_at,novelty,source_url,source_verified,target,benchmark,direction,forecast_at,horizon_end,horizon_name,mode,model_version,rule_version,prediction_id,market`.
  `source_verified` is **supplied by the operator**; its "true" value
  does not certify original source availability independently.
  `forecast_at` must reflect an independently provable forecast seal to
  support prospective claims. For historical replay, setting the historic
  clock value alone is insufficient.
- `observed_at` on market minute bars is a source's supplied availability
  clock. Historical downloads captured after a replay forecast cannot be
  represented as if the prices were already in local custody at the cutoff.

The command computes descriptive price returns and post-seal direction
agreement **only for source/schema/session/halts-eligible rows**, and returns
explicitly null or withheld metrics for incomplete input. All reports set
`trainingEligible=false`, `marketSourceIndependentlyVerified=false`,
`exchangeCalendarIndependentlyVerified=false`, and
`firstPublicIndependentlyVerified=false` until an independent,
replay-appropriate source certification procedure exists. This is intentional:
even a CSV created from genuine later-fetched prices does not prove the
forecast actually preceded market outcomes.

Run on verified-format exports (not synthetic fixtures in production):

```bash
python -m tris_market_event_intelligence.point_in_time_source_validation_cli \
  --predictions "$DATA/market-event-predictions.csv" \
  --source-manifest "$DATA/minute-source-manifest.json" \
  --sessions "$DATA/exchange-sessions.csv" \
  --max-anchor-age-minutes 5 \
  --max-exit-age-minutes 5 \
  --output "$DATA/point-in-time-price-diagnostics-v1.json"
```

A first-public report after the exchange closes may use an older closing bar
ONLY with a justified, explicit `--max-anchor-age-minutes` that can reach
the earlier market close and matching calendar data. It does not silently
expand the age bound. Different exchanges and horizons will require
separate source-backed configuration policies; a blanket multi-day
tolerance is not suitable for active intraday release windows.

## Stage 1 acceptance still outstanding

Independent certification of earliest public information (versus a reporter's
timestamp), event-family linkage (versus user-provided identifiers),
authoritative exchange/halts calendar, genuine provider quality, and
real symbol-specific minute coverage remain required. The source-format
integration is not validation of those facts. The existing #261 collection
and #252 research engines are not automatically wired or promoted by this
work. No increased learning accuracy can be claimed.
