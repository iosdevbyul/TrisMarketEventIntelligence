# NEWS-only model-independent LLM execution

This is an **explicit opt-in** extension of the existing source-evidence
validation pipeline, not a new news collector, a new stock analysis service
or a replacement for the running Gemma deployment. Learning-05 and its
previously generated private results are unchanged.

## Existing architecture reused

- `ImpactProvider` remains the public news impact interface and
  `ImpactProcessor` keeps the same persisted SQLite rows and API contract.
- `OpenAICompatibleProvider` is reused for the existing impact prompt,
  source provenance, uncertain-event rules and evidence validator.
- `GemmaNewsPressureProvider` retains its original pressure schema, source
  quote restriction and conditional intensity rules; its historical class
  name remains available.
- `news_materiality_local._parse_model_hypothesis` and the human-gold/holdout
  gates remain authoritative regardless of selected model.
- `gemma4_news_comparison` retains its strict original Gemma-2/Gemma-4
  comparison and report contract; a separate opt-in generic entrypoint calls
  its same-event/same-prompt/evidence checking core.

The only new boundary is `LLMProvider.generate_json`: model execution
returns *untrusted JSON text* and never writes labels, outcomes or DB rows.
`LLMSettings` validates provider, endpoint, model, timeout and declared
capabilities. `ConfiguredLLMProvider` implements low-level Ollama native
`/api/chat` or a remote OpenAI-compatible `/chat/completions` API.
`news_llm_factory` constructor-injects it into **existing** analyzers.

```text
ImpactProcessor / NEWS pipeline       NEWS pressure / materiality pilot
            |                                      |
 Existing ImpactProvider                Existing validators
            |                                      |
 Existing OpenAICompatibleProvider    Existing Pressure/Research adapters
            |                        (opt-in runtime injection)
            +------------------+-------------------+
                               |
                 LLMProvider (Protocol)
                               |
                  ConfiguredLLMProvider
                   /                 \
           local Ollama         remote compatible
```

The existing legacy commands do not use this runtime unless
`--configured-llm` is passed. Merely adding/changing an environment
variable does **not** update launchd, current provider, training results,
existing analysis history or operational model selection.

## Configuration

| Environment variable | Default | Meaning |
|---|---|---|
| `MARKET_EVENT_LLM_MODEL` | `gemma2:latest` | Existing model ID |
| `MARKET_EVENT_LLM_PROVIDER` | `ollama` | `ollama` or `openai-compatible` |
| `MARKET_EVENT_LLM_BASE_URL` | `http://127.0.0.1:11434` / `https://api.openai.com/v1` | Provider base URL |
| `MARKET_EVENT_LLM_API_KEY` | unset | Remote-only credential |
| `MARKET_EVENT_LLM_TIMEOUT_SECONDS` | 300 local, 90 remote | Request deadline |
| `MARKET_EVENT_LLM_TEMPERATURE` | `0` | Generation temperature 0..2 |
| `MARKET_EVENT_LLM_JSON_SCHEMA` | `1` | Explicit declared support: 0/1 |
| `MARKET_EVENT_LLM_THINKING` | `0` | Explicit native thinking (0/1) |
| `MARKET_EVENT_LLM_CONTEXT_WINDOW` | unset | Ollama num_ctx |
| `MARKET_EVENT_LLM_MAX_OUTPUT_TOKENS` | unset | Ollama num_predict / compatible max_completion_tokens |

The default mode remains the existing hardcoded local/remote route for
legacy CLIs. There is **no automatic remote failover** and no model pull or
installation. Ollama model readiness uses read-only `/api/tags` with exact
installed-name matching before the explicit run. Remote endpoints must be
HTTPS (loopback HTTP allowed for dev) and require a credential; secrets
must never be committed or added to private comparison reports.
Capabilities are explicit declarations, **not autodetected guarantees**.
Unsupported JSON-schema models use JSON mode with unchanged post-response
validators; schemas are never silently assumed supported. Thinking is off
unless explicitly configured.

## Opt-in examples

Use the existing commands without flags to keep present production
semantics. To explicitly select another local model for a bounded impact
evaluation, for example:

```sh
MARKET_EVENT_LLM_PROVIDER=ollama MARKET_EVENT_LLM_MODEL=qwen3:4b \
python -m tris_market_event_intelligence.impact_cli \
  --configured-llm --event-ids 123 --batch-size 3
```

Run an opt-in pressure pilot without modifying production DB:

```sh
MARKET_EVENT_LLM_MODEL=qwen3:4b \
python -m tris_market_event_intelligence.news_pressure_pilot_cli \
  --configured-llm --run --limit 2 --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/pressure-qwen-pilot.json"
```

For NEWS-only materiality use the existing source-pinned 48-candidate
packet and `evidence-pilot --configured-llm --run`; the output stays
explicit, owner-private and create-only. No human gold or stock data.

## Same-news, different-model comparison

The **same archived event IDs, source text, prompt, validator and
measurement method** are reused for both models, without refetching RSS.
Use a local preview before choosing whether to incur explicit model calls:

```sh
bash scripts/news-learning-review.sh model-compare --preview \
  --model-a gemma2:latest --model-b qwen3:4b --limit 2
```

When both models are already installed, to run:

```sh
bash scripts/news-learning-review.sh model-compare --run \
  --model-a gemma2:latest --model-b qwen3:4b --limit 2 \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-model-compare-gemma-qwen.json"
```

`--kind-a`, `--kind-b`, `--endpoint-a`, `--endpoint-b` support
explicit remote-compatible execution; both remote endpoints use the
`MARKET_EVENT_LLM_API_KEY` credential for this limited CLI.
In code, configure independently with `LLMSettings` and use the factory
for more sophisticated distinct credentials. Avoid sharing sensitive
source material with a remote provider unless authorized.

The report identifies model IDs and provider types but does **not**
assert semantic news comprehension, human-gold agreement, investment
returns, forward prediction, promotion or factual correctness. Only
schema acceptance, verbatim evidence and runtime duration are observed.
Holdout, reserved TEST, sources and event extraction continue unchanged.

## Deployment boundary

Explicitly changing a model does not promote it. Production launchd
and default integrated CLI remain bound to the existing model selection.
Experiment before updating any scheduler, and keep unrelated
DonghakStockVision price/return prediction pipelines separate.
