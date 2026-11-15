# Learning-08 — Opt-in Gemma 4 side-by-side NEWS evaluation pilot

## Model selection and licensing

Google officially publishes Gemma 4 under Apache 2.0. The family supports
reasoning, long-context and multilingual processing. These capabilities may
help with news context **but have not been proven on this private corpus**.
Official sources:
- https://ai.google.dev/gemma/docs/core/model_card_4
- https://gemma.ai/deployment/ollama/
- https://ollama.com/library/gemma4/tags

Ollama's documented named tags include `gemma4:e2b`, `gemma4:e4b`,
`gemma4:12b`, `gemma4:26b`, `gemma4:31b`. This pilot defaults to
`gemma4:e4b` but does **not** assert it fits the actual Mac. Exact tag
selection must follow a separate private hardware and free-memory preflight.
Ollama model download size is not equivalent to loaded memory or KV cache
overhead. Never use `gemma4:latest` or a cloud tag for a private test.

## What the first pilot actually validates

It keeps the existing Gemma 2 model and production pipeline untouched.
Both Ollama models receive exactly the **same archived, sourced, eligible
NEWS event** via the existing provider (same prompt version and output
contract). Model output passes the existing hard evidence-excerpt,
uncertainty and transmission-mechanism validator. The report retains
event ID, source-input SHA-256, model ID, output accept/reject counts and
elapsed wall time. It intentionally does **not** store raw news/source
text, model-generated rationale, original private prompts or human gold.

Accepted format is **not** independently confirmed understanding of
context or quality; empty impact hypotheses may be correct. The pilot
reports `independentNewsAccuracyMeasured=false` and
`promotionAllowed=false`. Only separately verified human reference
judgments and protected time-split evaluation can establish improvement.
No price/OHLCV/trading volume/returns are used, and no automated
model promotion, training or collector scheduling is enabled.

## Mac workflow, when ready

Existing `scripts/news-learning-review.sh` supports:

```sh
# Read-only selection from existing private news DB, no Ollama inference
bash scripts/news-learning-review.sh --update gemma-compare --preview --limit 2

# Inspect installed local Ollama model names (no pull/download)
bash scripts/news-learning-review.sh gemma-compare --preflight --gemma4 gemma4:e4b

# Explicit run only AFTER both models are installed, actual free RAM checked,
# and private Ollama runtime validated separately
bash scripts/news-learning-review.sh gemma-compare --run \
  --gemma4 gemma4:e4b --limit 2 \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/new-gemma4-pilot.private.json"
```

The optional `--db` points to the existing local news SQLite. Pilot
selection uses a read-only connection. The real run aborts if either
requested exact model tag is not already installed, and writes a new
owner-only research report. It **never** calls `ollama pull`, never
starts Ollama, and never changes live analysis environment variables.

Prior macOS operations documentation records an **Intel** device on
which the existing Gemma 2 previously timed out; do not infer that a
larger Gemma 4 model will be faster or even load reliably. Real hardware
inference, model availability and semantic improvements are still
unverified by CI's mocked model responses.

## Next stage after this foundational pilot

Compare genuine captured paired responses against independently judged
news-only importance, factual attribution, source evidence, conditional
mechanism, correction handling, uncertainty and misleading headline cases.
Use blinded human evaluation with the locked holdout; do not ask a model
to grade itself. The model with the best structural acceptance rate may
not be the model with the best news understanding. Keep the current
Gemma 2 model until independently verified evidence justifies a change.
