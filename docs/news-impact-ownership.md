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
