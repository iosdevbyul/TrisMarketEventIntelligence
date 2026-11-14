# News impact ownership — revised Stage 3

## Decided responsibility boundary

**TrisMarketEventIntelligence**: discover and deduplicate financial news,
verify source claims, identify *candidate* related companies and markets,
evaluate conditional **news-driven upward/downward pressure**, its qualitative
intensity, expected short/medium-term persistence, rationale, evidence, and
uncertainty. It is not the stock-price forecasting engine.

**DonghakStockVision**: acquire and maintain equity/market prices,
merge news-impact signals with its own price/volume features,
predict stock direction, price changes and price/return magnitudes,
evaluate forecast accuracy and decide whether/how to use predictions.
Only DonghakStockVision calibrates news pressure against market returns.

The news module may say: "conditional negative pressure, moderate intensity,
high uncertainty, because the event may increase supply-chain costs."
It must **not** say: "the stock will fall 3% tomorrow" or output a
predicted target price. A qualitative news-pressure score of 0–100, if a
future model produces one, is **not** a return percentage, a price target,
or empirically calibrated market reaction.

## Contract

New additive `news_impact_signal_v1` model:

- `event_id`, `market`, `sector`, `direction`
- `intensity`: `LOW`, `MODERATE`, `HIGH`, `UNASSESSED`
- `intensity_score`: optional 0–100 **news pressure index**, never stock %
- `horizon`: `SHORT_TERM` or `MEDIUM_TERM`
- `uncertainty`, `rationale`, `evidence_excerpt`, `source_url`
- `model_id`, `prompt_version`, `generated_at`
- strict `market_prices_used=false`, `predicts_stock_return=false`,
  `predicts_stock_price=false`, `calibrated_against_market_prices=false`
- `price_prediction_owned_by=DonghakStockVision`

Existing `market-impact-v8` records have no intensity field. Their
adapter sets `UNASSESSED` and no score, rather than fabricating a
magnitude. An uncertain direction cannot assert directional strength.
New model prompts, validation against news evidence, storage and consumer
delivery of *actual* assessed strength are subsequent milestones; a schema
alone does not establish empirical calibration or real-world accuracy.

## Migration and guardrails

1. Preserve existing operational RSS, claim verification, v3 cutover,
   SQLite and existing `/api/v1/impacts` clients; no breaking migration.
2. Add the additive news-only contract and regression tests first.
3. Next: adjust local Gemma 2 output to provide defensible
   source-supported intensity and explicitly refrain from numeric stock
   return forecasts. Unsubstantiated strength must remain `UNASSESSED`.
4. Then expose versioned news-impact signals in a consumer API.
   Consumers may decide how to use them, but cannot infer stock-price
   returns directly from the news strength score.
5. Treat the existing minute source audits, intraday event study,
   daily event study and Stage 3-B coverage bridge as **legacy research
   utilities**, not part of this module's product roadmap. Do not
   expand them into a second stock data platform. They remain intact
   for compatibility until a separately reviewed deprecation.
6. Do not modify DonghakStockVision in this repository's work.
   Implementing its prediction logic belongs to that project.

## Definition of done for news-side AI

- A sourced event produces an appropriately conditional direction,
  intensity (or justified `UNASSESSED`), time horizon, uncertainty,
  causal transmission explanation, and traceable source excerpt.
- Ambiguous or unsupported events are withheld or marked uncertain.
- No ticker identity is assumed merely because a headline matches.
- News outputs never claim specific stock percentages, future prices,
  trade executions, or return-forecast accuracy.
- Independent empirical calibration must occur on the price-model
  side using genuine observations and temporal holdouts.


## Opt-in Gemma 2 news-pressure assessment (next milestone)

`GemmaNewsPressureProvider` is an **additional, opt-in** second-stage
native Ollama provider. It accepts an eligible `EventInput` and an
existing evidence-validated `Impact`, then asks the locally configured
Gemma model to grade the conditional **news transmission strength**
as `LOW`, `MODERATE`, `HIGH`, or `UNASSESSED`. The system prompt
explicitly prohibits stock prices, percentage returns, targets and
market consensus. An assessment must cite an exact substring from the
underlying event's source evidence; strength index and category must
agree (1–33 LOW, 34–66 MODERATE, 67–100 HIGH). This 0–100 **heuristic
news-pressure index** is subjective and **not calibrated** to equity
returns. When the direction is uncertain the provider skips a second
model call and produces `UNASSESSED` with no index.

For backward compatibility this PR does **not** enable the model in
the existing 15-minute LaunchAgent, alter `market-impact-v8`, change
the impact_runs checkpoint keys, rewrite the SQLite schema, or alter
existing consumer API contracts. Its behavior is testable with an
injected HTTP client; live Gemma 2 quality and inference speed on the
Mac remain unverified. Next: controlled pilot, persisted strength
records, then an additive versioned API. Until that validation, no
new pressure values are advertised as operational model output.


## Opt-in persistent news-pressure API (Stage 3)

The new `NewsPressureStore` requires an **existing** initialized news
SQLite file. Its `initialize()` method is an explicit migration that
creates `news_pressure_signals` only when requested. It never runs from
the existing v3 event extractor, market-impact-v8 analyzer, or 15-minute
LaunchAgent. `save(impact_id=..., signal=...)` records only a
`news-pressure-intensity-v1` result whose event ID, source URL,
market, sector, direction, horizon, uncertainty, and evidence quote
match an **existing persisted** `market_impacts` record. A source
reference cannot be invented merely by passing an arbitrary ID.
Exact retries are idempotent, and conflicting updates are rejected.
The original `impact_runs` checkpoints are not modified.

The read-only, additive API `GET /api/v1/news-impact-signals?after_id=0&limit=100`
returns monotonically increasing IDs, `sourceImpactId`, and validated
`news_impact_signal_v1` fields. It returns HTTP 503 if the opt-in
table has not been initialized or any returned record fails integrity
checks; it does **not** claim zero results on an unavailable database.
Existing `/api/v1/impacts` consumers remain unchanged.

This is **storage and delivery plumbing only**. No automatic
Gemma calls, production enablement, price labels, model calibration,
or live Mac execution are implied. The subsequent separately
authorized pilot must measure real Gemma quality before enabling any
periodic writes.
