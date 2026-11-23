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
