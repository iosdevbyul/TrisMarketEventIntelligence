# News-only learning boundary (decision reaffirmed 2026-11-15)

TrisMarketEventIntelligence is a **news intelligence** module. It collects and
reviews news, classifies events and estimates conditional news-driven market
pressure (direction, intensity, horizon, uncertainty and evidence) using
information present in articles and human adjudication.

**It must not ingest, join, request, learn from, or score against stock-price,
minute-bar, OHLCV, trading-volume or realized-return data.** It does not
predict stock prices, calculate excess returns or train a trading model.
DonghakStockVision alone owns market data and price-related forecasting.
TrisMarketEventIntelligence produces a read-only *news analysis signal* that
a separate authorized consumer may combine with prices, outside this module.

The Learning-01 archived all-news cohort remains relevant: it includes
important-looking, ordinary and unclassified articles with explicit UNKNOWN
human importance. Valid local learning objectives are article significance
according to human review, source fidelity, evidence entailment, conditional
news pressure/horizon and properly stated uncertainty. Repeated evaluation
uses independent news/human-gold holdouts, not return labels.

PR #116's newly introduced historical market reaction join, CLI, associated
tests and documentation are removed by this correction. The aborted next
branch's draft market-price readiness module was never proposed/merged.

Some older intraday/event-study, market CSV compatibility code and historic
research documents predate this correction. They remain untouched for
backward compatibility only; **do not extend, schedule, promote or invoke
them as part of the news AI learning roadmap**. Any broader removal should
be scoped and tested separately.

Future tasks must not pull price data from DonghakStockVision or modify that
repository without a separate explicit user instruction. Never represent
news importance judgments as validated stock moves. Existing immutable
news review and evidence histories are not rewritten.
