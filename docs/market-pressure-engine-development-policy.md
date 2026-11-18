# General Market Pressure Engine — product contract and development policy

## Objective

**TrisMarketEventIntelligence analyzes financial news and economic indicators
to determine whether the newly available information can exert conditional
POSITIVE, NEGATIVE, MIXED or UNCERTAIN pressure on individual companies,
industries and broader financial markets.** It explains possible transmission
channels, qualitative strength and uncertainty. It does **not** train on
stock prices, forecast actual security returns, or place trades.

This module is independent of any stock-return forecasting engine.
Historical event replay, evidence screening, LLM runtime abstraction and
regulator-case research are **supporting techniques**. Accuracy at classifying
FAA/FDA regulatory decisions is not the product acceptance criterion.

## Responsibilities

| Layer | Responsibility | Implemented |
| --- | --- | --- |
| Ingestion | Existing RSS collector and durable SQLite article/event storage | Reused unchanged |
| Candidate discovery | Read time-eligible news and economic events; suggest related clusters with input-bound indexes | Opt-in prototype |
| Evidence verification | Verify source IDs, excerpts, cutoffs, cited companies and release values; retain unverified status | Structural checks |
| Event understanding | Extract cited reported facts, forecasts, opinions and novelty status | Model-assisted, review required |
| Economic transmission | State conditional channel from event to cash flows, demand, costs, funding, discount rates, FX, etc. | Model-assisted, review required |
| Target attribution | Distinguish market, sector and named company, with company names constrained to cited excerpts | Implemented |
| Direction, strength, uncertainty | Existing Impact direction and uncertainty scales; add LOW/MODERATE/HIGH/UNASSESSED conditional pressure strength | Implemented; semantic accuracy not verified |
| Publication | Create-only JSON research artifact, source digest and model/version audit | CLI; not yet integrated into production API |

The new general path reuses `impact.Impact` and the existing conservative
mechanism and unfinished-rationale validators. Its two-stage reasoning
adds new information without changing the prior REST API, SQLite schema
or previously merged impact calculations.

### Direction, strength and confidence are different

- Direction: `POSITIVE`, `NEGATIVE`, `MIXED`, `UNCERTAIN`.
- Qualitative strength: `LOW`, `MODERATE`, `HIGH`, `UNASSESSED`.
  This describes *potential economic pressure*, not the probability
  or size of an actual stock move.
- Uncertainty: `LOW`, `MEDIUM`, `HIGH` for the *hypothesis*, not
  source authenticity. If any analysis input was unverified, the
  current policy forces `HIGH` uncertainty across the analysis.
- An `UNCERTAIN` direction cannot be assigned a confident
  non-UNASSESSED strength.
- A HIGH strength conclusion cannot rest entirely on unverified
  reported sources. Even an official report does not independently
  verify economic causality, magnitude or whether a market priced it in.

Only the source quotation and some structural implications can be
mechanically authenticated. The correct economic direction and intensity
remain **unverified model judgments**.

## Local operator workflow

The existing private SQLite DB is not modified. First inspect candidates
without invoking a model:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
bash scripts/news-learning-review.sh market-pressure-cycle \
  --db "$DATA/news.sqlite3" --as-of 2026-11-18 --preview
```

A real model-run, requiring the locally installed `gemma4:e2b`:

```bash
bash scripts/news-learning-review.sh market-pressure-cycle \
  --db "$DATA/news.sqlite3" --as-of 2026-11-18 \
  --lookback-days 10 --max-groups 2 \
  --run --model gemma4:e2b \
  --output "$DATA/market-pressure-first-run-v1.json"
```

The output is create-only; repeated runs must use a different path
and be labeled as repeat experiments. The model can suggest up to 2
clusters by default, with at most 8 sources each; each cluster gets
one fact-extraction call and, if there are grounded claims, a second
impact-reasoning call. Grouping is unverified and may produce an empty
list rather than inventing a connection.

For reviewed source IDs, bypass clustering to target a particular
event through `market-evidence-pressure` with the existing `--db`,
`--article-id`, `--economic-event-id`, `--event-id`, `--subject`,
`--as-of` and `--preview|--run` arguments. A JSON evidence packet
also remains supported. Opt in to `--single-stage-compat` only to
compare earlier single-pass `ImpactResult` behavior.

## Critical evidence boundaries

- RSS title/summary is **not** proof of the original article body.
- All source publication times originate from existing DB metadata;
  they are not authenticated archived historical page snapshots.
- Candidate sources with missing, timezone-naive, malformed or
  outside-window publication timestamps are excluded.
- A forecast is not an observed outcome; a policy proposal is not
  an implemented action; approved products are not evidence of
  commercial success.
- An economic surprise is not inferred without both an explicit
  actual value and an explicit sourced expectation.
- Prompt text is untrusted source material and never eligible as
  operating instructions.
- Cited excerpts, source IDs, claim indexes and company names
  are checked against cutoff-visible inputs outside the model.
- A full score of structural checks does **not** prove correct
  sign, intensity, causality or actual stock-price movement.
- There is **no new background service or automatic trading**.
  Current collector and production API behaviors are unchanged.

## Verification and acceptance

Python CI checks schema, Ruff and unit tests including cutoff leaks,
future citations, fabricated fact quotes, unsupported market
expectations, fabricated companies, weak strength evidence, empty
grouping and malformed group indexes.

**Real Ollama inference on the user's machine is still required.**
Performance acceptance must use multiple families such as inflation,
rates, earnings, products, labor, geopolitics and industry supply shocks.
A holdout set should be independently curated from then-public
news/indicator facts, with *target-specific economic sign and economic
channel* reviewed by qualified analysts. Distinguish economic reasoning
from ex-post stock return labels, which are NOT learned in this module.

Future integration priorities:
1. Validate real Gemma4 end-to-end outputs against independent
   economic transmission reference examples and diverse topics.
2. Expand verified economic-series source adapters beyond currently
   stored event and RSS releases; establish precise publication/PIT.
3. Add robust event deduplication and article body evidence quality.
4. Add audited output APIs and versioned persistence, but preserve
   existing API/DB contracts until reviewed.
