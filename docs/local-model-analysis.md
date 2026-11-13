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


## Intel Mac Ollama ReadTimeout adjustment

Real `gemma2:latest` one-event pilot returned `ReadTimeout` at the
previous HTTP 90-second limit. The default local provider timeout is now
300 seconds (remote default remains 90 seconds); use
`--timeout-seconds N` to explicitly select 1–900 seconds.

```sh
git pull --ff-only origin main
python -m pip install -e .
export MARKET_EVENT_LLM_MODEL=gemma2:latest
market-event-analyze --provider local --batch-size 1 --timeout-seconds 300
market-event-analyze --show-errors
```

Only a successful validated model response creates `impact_runs`; a timeout
does not mark a claim as analyzed. The earlier error remains until this
event is successfully processed. No automatic retry, no collector/LaunchAgent
changes, no external model calls, and no impact validation weakening.

The timeout controls HTTP operations, not an independent wall-clock deadline.
The model's disk size (5.4 GB) does not tell us whether this Intel Mac has
enough free RAM or acceptable inference throughput. If this retry still times
out, gather `ollama ps` and local runtime resource observations before
changing models or retry policy.

## Specific analysis rejection codes

The real Gemma 2 second attempt returned `ValueError` following the
earlier `ReadTimeout`. The old error entry does not reveal the underlying
condition. Future rejected results record safe identifiers:
`PROVIDER_JSON_INVALID`, `PROVIDER_RESPONSE_SHAPE`,
`PROVIDER_CONTENT_INVALID`, `EVIDENCE_MISMATCH`,
`ENTITY_MISMATCH`, `UNCERTAINTY_MISMATCH`,
`DUPLICATE_IMPACT_KEYS`, plus existing `ValidationError` and timeout types.

This does not retroactively classify old `ValueError` entries.
Only a new, explicitly invoked one-event analysis can show a more
specific code; failed events remain retryable.

```sh
git pull --ff-only origin main
export MARKET_EVENT_LLM_MODEL=gemma2:latest
market-event-analyze --provider local --batch-size 1 --timeout-seconds 300
market-event-analyze --show-errors
```

Evidence checks and risk constraints are not relaxed. No raw generated
text, prompts or credentials are persisted as error descriptions.

## Phase 7-B source-aware Gemma 2 risk prompt

The real Intel Mac pilot rejected an UNVERIFIED event with
`UNCERTAINTY_MISMATCH`. Version `market-impact-v3` supplies
trusted, verification-status-specific system instructions for the model.
For UNVERIFIED or CANDIDATE inputs, every nonempty impact must set
`uncertainty=HIGH` and must not turn a reported claim into a verified fact.
If evidence is thin, `{"impacts":[]}` remains valid.

The existing post-response validator continues rejecting LOW or MEDIUM
uncertainty for unverified events. There is no silent rewriting of output
and no automatic retry. A new prompt version isolates completed runs
from historical v2 errors without destroying prior data.

One controlled on-Mac pilot:

```sh
git pull --ff-only origin main
export MARKET_EVENT_LLM_MODEL=gemma2:latest
market-event-analyze --provider local --batch-size 1 --timeout-seconds 300
market-event-analyze --show-errors
curl -sS 'http://127.0.0.1:8100/api/v1/analysis-runs?after_id=0&limit=5'
```

CI mocks model responses; only an on-Mac run can verify Gemma 2
compliance, runtime and market-analysis accuracy.

## Phase 7-B v4 — local Ollama structured output

The Intel Mac pilot generated a v3 `ValidationError`. Ollama documents
JSON Schema constraints through the OpenAI-compatible chat completions API.
Local requests now use `response_format.type=json_schema` with the
`ImpactResult` Pydantic schema, explicit allowed enum values and
`uncertainty=HIGH` as the *only* allowed value for UNVERIFIED/CANDIDATE inputs.
Remote-provider requests continue using the existing `json_object` mode.

The model output is still parsed and checked against the original Pydantic
schema, literal evidence/entity rules and verification rules; no repair,
fallback or implicit approval is permitted. New v4 outputs and failures remain
independent from previous v2/v3 attempts. When the schema fails, the stored
error identifies only an allowlisted field/type, e.g.
`SCHEMA_MISSING_RATIONALE`, never the generated text.

Run a single opt-in on-device pilot:

```sh
git pull --ff-only origin main
export MARKET_EVENT_LLM_MODEL=gemma2:latest
market-event-analyze --provider local --batch-size 1 --timeout-seconds 300
market-event-analyze --show-errors
curl -sS 'http://127.0.0.1:8100/api/v1/analysis-runs?after_id=0&limit=5'
```

GitHub CI tests the request shape with a mocked model only. Real local Ollama
schema compatibility and the analysis output still require Mac verification.


## Phase 7-B v5 — native Ollama compatibility and HTTP diagnostics

Real device evidence: v2 timed out or failed risk checks, v3 failed schema
validation, and v4 returned an HTTPStatusError from the local API. The HTTP
status was not recorded for that historical attempt, so its cause is unknown.

The local provider now uses the Ollama native `POST /api/chat` interface
with `stream: false` and a compact inline schema under `format`.
Unlike the former OpenAI-compatible nested `response_format`, this is the
official native structured-output contract. For HTTP 400 or 422 on the
structured request only, the local client makes **one bounded compatibility
retry** using Ollama's native `format: "json"` mode. No retry is performed
for timeouts or other status codes. The same strict Pydantic, evidence, entity
and risk checks apply in both modes. This is not an automatic correction
of a rejected hypothesis. Remote LLM requests are unchanged.

Future HTTP errors persist a sanitized status code (such as `HTTP_400`),
not a raw response body or credential. Historical v4 error rows are preserved.
`market-impact-v5` is a separate versioned analysis contract.

The next real-Mac run remains a **single-event pilot**, not a full backlog:

```sh
git pull --ff-only origin main
export MARKET_EVENT_LLM_MODEL=gemma2:latest
market-event-analyze --provider local --batch-size 1 --timeout-seconds 300
market-event-analyze --show-errors
curl -sS 'http://127.0.0.1:8100/api/v1/analysis-runs?after_id=0&limit=5'
```

No CI assertion can certify actual Gemma 2 quality or CPU/RAM performance.

## Phase 7-B v6 — model output contract recovery

The on-device Gemma 2 v5 pilot reached native Ollama inference, but
`rationale` failed the existing Pydantic minimum length of 10 characters.
Previously the local JSON Schema omitted this requirement even though the
application validator enforced it, so the model could emit a response that
would inevitably be rejected.

`market-impact-v6` aligns the native schema with Pydantic constraints
for rationale, evidence excerpts, driver strings, impact count and item
limits. The model still must provide exact source evidence, HIGH uncertainty
for unverified events, and truthful conditional research only.

**Bounded repair:** If the local model returns syntactically valid JSON that
fails the model contract or evidence checks, the client may generate one
new response using the same original source data and a safe failure-code
hint. It never sends rejected model text back to the model, edits a failed
rationale, pads a quote, or accepts unsupported evidence. After the second
failed output, the event remains retryable and records the safe failure code.
No additional retries are attempted for timeouts or 5xx errors. Old Ollama
400/422 schema-compatibility fallback remains separately bounded.

The end-to-end offline test suite covers a too-short initial rationale
followed by a correct complete result, repeated invalid responses,
unsupported evidence, empty hypotheses and transport failures. It does not
claim that the local Gemma 2 has succeeded yet. No batch automation or
scheduled LLM processing is added.

### Current live verification

The latest verified on-device status is still zero successful runs.
A **single** future Mac pilot should be considered only when ready to
verify deployment; no repeated ad-hoc prompts or model reinstalls are
necessary. If a run fails again, inspect the version-specific stored error,
not a new prompt revision.

## Phase 7-B: optional evidence-independent driver list

The real Intel Mac v6 pilot rejected an otherwise uninspected model response
with `SCHEMA_TOO_SHORT_DRIVERS`. This means `drivers=[]`, not that the
source quote, rationale or market hypothesis has been established as valid.

The `drivers` field remains a required array, but may contain **zero to six**
nonempty short phrases. An empty list means the model did not identify
separate causal factors. The system never manufactures drivers to satisfy a
minimum count. The original source quotation, rationale minimum length,
entity matching, uncertainty guard, market/sector validation, and strict
storage rules all remain unchanged. An invalid quote or fabricated rationale
still fails; empty drivers cannot convert it into success.

This targeted correction retains `market-impact-v6`: the real device has
no accepted v6 records and the failed event remains retryable. Do not
interpret CI mock output as real Gemma 2 validation, or automatically begin
processing more than one event.
