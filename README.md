# TrisMarketEventIntelligence

An independent, local-first service for collecting financial news, verifying economic events, and publishing structured market-impact analyses to consumers including DonghakStockVision.

## Principles

- Keep financial event ingestion and analysis independent from the consumers.
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
4. **Impact analysis:** explicit evidence, hypotheses, uncertainty and model-provider adapter.
5. **Event API:** versioned event schema and cursor-based consumer polling.
6. **Mac operations:** launchd, observability, reboot and failure recovery.
7. **DonghakStockVision integration:** consume verified events without rewriting existing KRX pipelines.
8. **v1.0 verification:** live-source end-to-end, historical leakage checks and release criteria.

No real-world news collection or AI analysis is implemented in the foundation phase. The health endpoint alone is not a production-readiness signal.
