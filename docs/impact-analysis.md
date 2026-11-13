# Phase 4: Market impact research hypotheses

The service now supports an optional **real** LLM inference adapter through a configurable OpenAI-compatible Chat Completions JSON endpoint. It is not configured in CI. No API key, model, or live model response is shipped or simulated as production AI.

## Usage

```bash
market-event-collect --once
market-event-extract
export MARKET_EVENT_LLM_API_KEY="<set in your private environment>"
export MARKET_EVENT_LLM_MODEL="<your configured model>"
# Optionally MARKET_EVENT_LLM_BASE_URL=https://your-provider.example/v1
market-event-analyze --db data/news.sqlite3
```

Never commit API keys. For local model servers, an API key value may be a local placeholder if the server ignores authorization.

## Contract

- Only Phase 3 `VERIFIED` monetary-policy entries are submitted to the provider. This **does not independently authenticate original policy documents**; a verified RSS text match is an eligibility check, not a trade-ready fact.
- The provider sees event metadata and the stored *excerpt*, not complete RSS capture bodies.
- AI results must conform to strict enums and include rationale, literal source excerpts, uncertainty and drivers.
- `POSITIVE` and `NEGATIVE` mean **hypothesized** pressure, never observed returns.
- The analysis does not have market consensus, quantitative surprise, prices or timestamps of market reaction; those require separate data sources.
- Output has no calibrated probability, stock recommendation, or causal proof.
- All successful attempts, including `NO_HYPOTHESIS`, are versioned by event/model/prompt and stored idempotently. Failures remain retryable with private, non-payload error bookkeeping.
- To run a new analysis after changing prompts, increase `PROMPT_VERSION`. To compare models, use distinct stable model IDs.
- A future Phase 8 release gate must evaluate live feed extracts and independent market evidence, model quality, hallucination rates, and economic risk.

## Reliability and security

Provider calls time out after 40 seconds. External provider URLs require HTTPS; local loopback HTTP is allowed. The configured model provider can receive finance event excerpts, so respect upstream publisher terms. Raw model payloads and secrets are not logged.

This is for research; neither the provider nor consumer should issue trading instructions on these outputs.
