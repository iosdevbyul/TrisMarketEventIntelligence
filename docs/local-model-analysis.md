# Phase 5-C — Local-first multi-category market impact research

This release adds conservative research hypotheses for all ten event categories.
It DOES NOT assert third-party reports to be true. Non-monetary events are
eligible only when the extractor marks claimKind=REPORTED. Forecasts, rumors,
attributed reports and ambiguous monetary-policy candidates are skipped.
For CANDIDATE/UNVERIFIED events the model MUST return HIGH uncertainty.
All output remains a conditional hypothesis, NOT a realized market move.

## Mac setup (manual local deployment later, in Phase 6)

Install Ollama from its official release, install a model compatible with the
Mac's RAM/CPU architecture, and ensure the local server listens on loopback at
127.0.0.1:11434. Do not assume a particular 7B/8B model is viable on an Intel Mac.

```bash
ollama list
market-event-analyze --list-local-models
export MARKET_EVENT_LLM_MODEL=YOUR_INSTALLED_MODEL_NAME
market-event-collect --once
market-event-extract
market-event-analyze --provider local --batch-size 10
```

Local mode is the default: `http://127.0.0.1:11434/v1/chat/completions`.
It requires no paid API key. HTTP proxy settings are disabled for model traffic.
For an explicitly selected external provider, use `--provider remote` with
`MARKET_EVENT_LLM_BASE_URL` (HTTPS) and `MARKET_EVENT_LLM_API_KEY`.

## Stored contract

Results include market, sector, horizon, hypothesized direction, rationale,
exact supporting source excerpt, uncertainty and named entities copied exactly
from the excerpt. The analysis API also includes `eventStatus` and `claimKind`.
Schema migration adds `market_impacts.entities_json` for existing SQLite DBs.
Completed runs are keyed by (event_id, model_id, prompt_version).
Changing the prompt version to market-impact-v2 re-runs existing records
without overwriting earlier research.

The source excerpt and metadata are sent to the local LLM as UNTRUSTED input.
Unknown facts such as consensus surprise, ticker associations, exact price impacts
and observed market returns must not be invented. All evidence quoted must
exist verbatim in the event excerpt. The guard cannot fully prove factual
correctness of a model's rationale; manual and live data QA remain required.

No Ollama binary/model is bundled. GitHub CI tests mock the local API; they
do NOT certify actual model quality, performance, RAM adequacy or live ingestion.
