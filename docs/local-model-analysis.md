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

## Phase 7-B bounded real Ollama pilot on Intel Mac

The collector-to-impact bridge was validated on the device with 38 created
events. The local Ollama endpoint reports one installed model,
`gemma2:latest` (5.4 GB on disk). Disk size does **not** establish loaded
RAM consumption or a safe speed; benchmark the installed model on the actual
machine before larger batches.

The pilot now selects **eligible** research claims *before* applying the batch
size. In particular, unverified monetary-policy reports and forecasts are
not allowed to consume an entire batch without inference. Existing conservative
eligibility and evidence validation remain unchanged. Local CLI defaults to the
same macOS SQLite database as the collector unless `MARKET_EVENT_DB` or
`--db` specifies another path. An explicit installed-model preflight fails
before touching the DB when Ollama or the requested model is unavailable.

```sh
git pull --ff-only origin main
python -m pip install -e .
export MARKET_EVENT_LLM_MODEL=gemma2:latest
market-event-analyze --list-local-models
market-event-analyze --provider local --batch-size 1
curl -sS 'http://127.0.0.1:8100/api/v1/analysis-runs?after_id=0&limit=5'
curl -sS 'http://127.0.0.1:8100/api/v1/impacts?after_id=0&limit=5'
```

After inspecting one event's output and errors, test a batch of at most three.
`processed=1, impacts=0, failed=0` is a valid `NO_HYPOTHESIS` conclusion;
`failed=1` means inference/validation failed and the event remains retryable.
No model may assert that an UNVERIFIED input is a verified fact. Model outputs
are still conditional hypotheses and not final stock-level signals.

Ollama runs only when `market-event-analyze` is invoked. It is not added to
the 30-minute collector LaunchAgent or started as a public service. CI replaces
Ollama with a stub and therefore cannot establish real Mac memory, inference
speed or model validity.


## Troubleshooting actual model inference failures

A `market-event-analyze --provider local --batch-size 1` result of
`{"processed":0,"impacts":0,"failed":1}` means one selected event failed to
complete. Empty `/api/v1/analysis-runs` and `/api/v1/impacts` are expected
because failed events never become successful checkpoints.

Inspect the safe, previously stored error class **without loading the model
again or modifying SQLite**:

```sh
market-event-analyze --show-errors
```

The report includes event ID, model name, prompt version, exception class,
last attempt timestamp and attempt count only; it does not contain full
provider responses or user text. `ReadTimeout` suggests the 90-second
request timeout, `ValidationError` suggests malformed/contract-invalid
JSON, `HTTPStatusError` suggests an Ollama HTTP rejection, and `ValueError`
can mean unsupported evidence or incomplete response. Exception type alone
is not definitive; preserve the failed row for subsequent controlled diagnosis.

Do not rerun the whole backlog until a one-event pilot succeeds.
