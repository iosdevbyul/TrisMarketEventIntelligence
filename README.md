# TrisMarketEventIntelligence

An independent, local-first market-event research service that collects news and economic indicators, analyzes economic meaning, predicts subsequent market direction at a sealed point in time, and evaluates those predictions against **real observed stock, sector, index and ETF prices**. This project is independent of DonghakStockVision and does not execute trades.

**Project mission, six-stage roadmap, acceptance gates and change control:** [PROJECT_CHARTER.md](PROJECT_CHARTER.md). The charter governs future development when older documentation conflicts.

## Principles

- Keep financial event ingestion and news-only impact analysis independent from consumers.\n- TrisMarketEventIntelligence estimates whether a news event exerts positive, negative or mixed pressure and how strongly; it does **not** predict stock returns, targets or prices.\n- DonghakStockVision owns price collection, market forecasts, and any calibration of news pressure against realized returns.
- Store original source provenance and separate observed facts from AI hypotheses.
- Record publication, detection and processing times separately to avoid look-ahead bias.
- Treat event IDs as durable, deduplicated delivery cursors.
- Bind the local HTTP service to loopback by default.
- Never commit API keys, paid article bodies, or local operational databases.

## Local setup

Requires Python 3.11 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
ruff check .
pytest -q
market-event-api
```

Visit `http://127.0.0.1:8100/health` to confirm the service is running.

Optional settings: `MARKET_EVENT_HOST` (loopback-only) and `MARKET_EVENT_PORT` (default 8100).

## Delivery roadmap

1. **Foundation:** installable package, CI, isolated FastAPI health endpoint.
2. **Collector:** official-source adapters, source preservation, SQLite idempotency and retry.
3. **Event verification:** differentiate confirmed decisions from forecasts or commentary.
4. **News pressure analysis (legacy milestone):** conditional economic meaning, qualitative strength, evidence, uncertainty and model-provider adapter. The current market-learning roadmap additionally includes sealed market-direction forecasts and actual market-outcome verification.
5. **Event API:** versioned event schema and cursor-based consumer polling.
6. **Mac operations:** launchd, observability, reboot and failure recovery.
7. **DonghakStockVision integration:** consume verified events without rewriting existing KRX pipelines.
8. **v1.0 verification:** live-source end-to-end, historical leakage checks and release criteria.

No real-world news collection or AI analysis is implemented in the foundation phase. The health endpoint alone is not a production-readiness signal.

## Project responsibility boundary (Stage 3 revised)\n\nSee [docs/news-impact-ownership.md](docs/news-impact-ownership.md) for the ownership contract and migration notes. The first additive `news_impact_signal_v1` schema accepts directional strength (`LOW`, `MODERATE`, `HIGH`, or `UNASSESSED`), an optional **news pressure index from 0 to 100, not a price percentage**, and provenance. Existing `market-impact-v8` records are adapted with `UNASSESSED` rather than inventing a strength score. Producing genuinely evidence-supported strength estimates is a separate model upgrade and evaluation task. Legacy minute-price utilities remain for compatibility; real price validation is now in the project's research scope under PROJECT_CHARTER.md.\n

## Current market-learning roadmap (2026-11-23)

See [PROJECT_CHARTER.md](PROJECT_CHARTER.md) for the authoritative gates: (1) market outcome contract, (2) dual semantic/market forecasts, (3) real price and benchmark validation, (4) AI discrepancy research, (5) independent learning experiments, (6) learning dashboard. The earlier eight-stage delivery roadmap above is historical groundwork, not a replacement for these gates. Preserve previous results, review draft PR #252 before reimplementation, and do not claim improved forecasting until held-out baseline comparisons demonstrate it.
