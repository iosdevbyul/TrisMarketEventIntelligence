# News impact ownership — revised Stage 3

## Decided responsibility boundary

**TrisMarketEventIntelligence**: discover and deduplicate financial news,
verify source claims, identify *candidate* related companies and markets,
evaluate conditional **news-driven upward/downward pressure**, its qualitative
intensity, expected short/medium-term persistence, rationale, evidence, and
uncertainty. It is not the stock-price forecasting engine.

**DonghakStockVision**: acquire and maintain equity/market prices,
merge news-impact signals with its own price/volume features,
predict stock direction, price changes and price/return magnitudes,
evaluate forecast accuracy and decide whether/how to use predictions.
Only DonghakStockVision calibrates news pressure against market returns.

The news module may say: "conditional negative pressure, moderate intensity,
high uncertainty, because the event may increase supply-chain costs."
It must **not** say: "the stock will fall 3% tomorrow" or output a
predicted target price. A qualitative news-pressure score of 0–100, if a
future model produces one, is **not** a return percentage, a price target,
or empirically calibrated market reaction.

## Contract

New additive `news_impact_signal_v1` model:

- `event_id`, `market`, `sector`, `direction`
- `intensity`: `LOW`, `MODERATE`, `HIGH`, `UNASSESSED`
- `intensity_score`: optional 0–100 **news pressure index**, never stock %
- `horizon`: `SHORT_TERM` or `MEDIUM_TERM`
- `uncertainty`, `rationale`, `evidence_excerpt`, `source_url`
- `model_id`, `prompt_version`, `generated_at`
- strict `market_prices_used=false`, `predicts_stock_return=false`,
  `predicts_stock_price=false`, `calibrated_against_market_prices=false`
- `price_prediction_owned_by=DonghakStockVision`

Existing `market-impact-v8` records have no intensity field. Their
adapter sets `UNASSESSED` and no score, rather than fabricating a
magnitude. An uncertain direction cannot assert directional strength.
New model prompts, validation against news evidence, storage and consumer
delivery of *actual* assessed strength are subsequent milestones; a schema
alone does not establish empirical calibration or real-world accuracy.

## Migration and guardrails

1. Preserve existing operational RSS, claim verification, v3 cutover,
   SQLite and existing `/api/v1/impacts` clients; no breaking migration.
2. Add the additive news-only contract and regression tests first.
3. Next: adjust local Gemma 2 output to provide defensible
   source-supported intensity and explicitly refrain from numeric stock
   return forecasts. Unsubstantiated strength must remain `UNASSESSED`.
4. Then expose versioned news-impact signals in a consumer API.
   Consumers may decide how to use them, but cannot infer stock-price
   returns directly from the news strength score.
5. Treat the existing minute source audits, intraday event study,
   daily event study and Stage 3-B coverage bridge as **legacy research
   utilities**, not part of this module's product roadmap. Do not
   expand them into a second stock data platform. They remain intact
   for compatibility until a separately reviewed deprecation.
6. Do not modify DonghakStockVision in this repository's work.
   Implementing its prediction logic belongs to that project.

## Definition of done for news-side AI

- A sourced event produces an appropriately conditional direction,
  intensity (or justified `UNASSESSED`), time horizon, uncertainty,
  causal transmission explanation, and traceable source excerpt.
- Ambiguous or unsupported events are withheld or marked uncertain.
- No ticker identity is assumed merely because a headline matches.
- News outputs never claim specific stock percentages, future prices,
  trade executions, or return-forecast accuracy.
- Independent empirical calibration must occur on the price-model
  side using genuine observations and temporal holdouts.


## Opt-in Gemma 2 news-pressure assessment (next milestone)

`GemmaNewsPressureProvider` is an **additional, opt-in** second-stage
native Ollama provider. It accepts an eligible `EventInput` and an
existing evidence-validated `Impact`, then asks the locally configured
Gemma model to grade the conditional **news transmission strength**
as `LOW`, `MODERATE`, `HIGH`, or `UNASSESSED`. The system prompt
explicitly prohibits stock prices, percentage returns, targets and
market consensus. An assessment must cite an exact substring from the
underlying event's source evidence; strength index and category must
agree (1–33 LOW, 34–66 MODERATE, 67–100 HIGH). This 0–100 **heuristic
news-pressure index** is subjective and **not calibrated** to equity
returns. When the direction is uncertain the provider skips a second
model call and produces `UNASSESSED` with no index.

For backward compatibility this PR does **not** enable the model in
the existing 15-minute LaunchAgent, alter `market-impact-v8`, change
the impact_runs checkpoint keys, rewrite the SQLite schema, or alter
existing consumer API contracts. Its behavior is testable with an
injected HTTP client; live Gemma 2 quality and inference speed on the
Mac remain unverified. Next: controlled pilot, persisted strength
records, then an additive versioned API. Until that validation, no
new pressure values are advertised as operational model output.


## Opt-in persistent news-pressure API (Stage 3)

The new `NewsPressureStore` requires an **existing** initialized news
SQLite file. Its `initialize()` method is an explicit migration that
creates `news_pressure_signals` only when requested. It never runs from
the existing v3 event extractor, market-impact-v8 analyzer, or 15-minute
LaunchAgent. `save(impact_id=..., signal=...)` records only a
`news-pressure-intensity-v1` result whose event ID, source URL,
market, sector, direction, horizon, uncertainty, and evidence quote
match an **existing persisted** `market_impacts` record. A source
reference cannot be invented merely by passing an arbitrary ID.
Exact retries are idempotent, and conflicting updates are rejected.
The original `impact_runs` checkpoints are not modified.

The read-only, additive API `GET /api/v1/news-impact-signals?after_id=0&limit=100`
returns monotonically increasing IDs, `sourceImpactId`, and validated
`news_impact_signal_v1` fields. It returns HTTP 503 if the opt-in
table has not been initialized or any returned record fails integrity
checks; it does **not** claim zero results on an unavailable database.
Existing `/api/v1/impacts` consumers remain unchanged.

This is **storage and delivery plumbing only**. No automatic
Gemma calls, production enablement, price labels, model calibration,
or live Mac execution are implied. The subsequent separately
authorized pilot must measure real Gemma quality before enabling any
periodic writes.


## Real Gemma 2 pilot — explicit, small, read-only

The following commands are required only on a Mac with an installed
local Ollama model. Update the checkout before running.

```bash
bash scripts/verify.sh --update
market-event-news-pressure-pilot --limit 3
market-event-news-pressure-pilot --limit 3 --run --model gemma2:latest \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-pressure-pilot-$(date +%Y%m%dT%H%M%S).json"
```

Preview only reads up to three distinct, eligible **already persisted**
impact analyses from the existing news SQLite DB. `--run` makes explicit
bounded local Ollama calls to `127.0.0.1:11434` and writes one new report
with model output, duration and deterministic rejection codes.
It **does not initialize `news_pressure_signals`**, persist a signal,
modify existing records/checkpoints, touch the 15-minute LaunchAgent,
or invoke any price feed. It does not retroactively extract events.

A `PILOT_OUTPUT_VALIDATED` status means the response met *structural*
and source-excerpt constraints, **not** that its strength is accurate.
Manual review of source support and consistency is still required.
If the report says `NO_EXISTING_ELIGIBLE_IMPACTS`, there is no
defensible live source observation to score; do not fabricate one.
The diagnostic report may contain news excerpts and should be kept
under the user's private Application Support folder.

## Pilot distinction: unassessed is not a Gemma inference

The 2026-11-15 real Mac pilot selected one GEOPOLITICAL / GLOBAL / ENERGY impact with UNCERTAIN direction. The pressure provider correctly skips inference, but the original pilot labeled the result PILOT_OUTPUT_VALIDATED. The revised pilot instead reports NO_MODEL_CALLS_UNCERTAIN_DIRECTION, model_call_attempts=0 and SKIPPED_UNCERTAIN_DIRECTION. Eligible directional impacts receive selection priority. Actual local Gemma quality remains unverified until a genuinely eligible directional impact exists in source evidence; never fabricate a directional event simply to pass a pilot.


## Two-stage end-to-end pilot without relying on old direction records

The previous read-only pressure pilot was blocked by a stored UNCERTAIN
impact. This **separate** workflow uses the existing strict
`pilot_candidates(..., new_only=True)` selector for up to three
not-already-checkpointed, non-duplicate, source-supported real events.
It then chains the existing `market-impact-v8` Ollama direction and
causal-mechanism validator with the opt-in news-pressure validator.
Only a defensible POSITIVE, NEGATIVE or MIXED stage-one hypothesis can
reach stage two. If the source does not support a direction, stage two
is skipped; source quote or mechanism validation failures reject the
result. No default live model calls are made.

Preview the actual selected events after updating the project:

```bash
market-event-news-two-stage-pilot --limit 3
```

An **explicit** manual, one-shot local run may then be performed:

```bash
market-event-news-two-stage-pilot --limit 3 --run --model gemma2:latest \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-two-stage-$(date +%Y%m%dT%H%M%S).json"
```

Both commands read the same live news SQLite with `mode=ro`. They do
not create market_impacts, impact_runs, news_pressure_signals, ticker
approvals or price forecasts. The output file is new-only, private,
and contains the evidence-linked hypothesis for quality review.
If no new eligible real event is present, the report says
`NO_ELIGIBLE_NEW_EVENTS`. This is not evidence of a model failure,
and the pipeline must not invent an example or replay already
checkpointed events to pass. The two-stage result is not a calibrated
real-world news intensity score or approval to enable automation.

## Real two-stage failure: quote grounding (2026-11-15)

The first live 3-event two-stage pilot made 3 direction-model calls and 1
pressure-model call; two events had NO_SUPPORTED_DIRECTION and event 69
failed NEWS_INTENSITY_UNSUPPORTED after the pressure call. The available
report does **not** expose the raw quote or a detailed error cause, so
whether the model omitted, paraphrased or invented a quote is not known.

The revised prompt supplies ALLOWED_EVIDENCE_QUOTE from the already
validated first-stage impact and requests an **exact copy** or UNASSESSED
when the evidence is insufficient. The safety gate is not weakened:
an assessed grade must quote the exact allowed impact excerpt; missing,
non-source and other-source excerpts receive distinct stable error codes.
The read-only pilot also reports the failed stage and aggregate codes,
without printing model outputs or modifying SQLite or the LaunchAgent.
This is a prompt and error-diagnostics improvement, **not evidence** that
Gemma 2's assessed news intensity is accurate. A subsequent explicit
real Mac pilot is still necessary before enabling automated scoring.


## Epistemic separation after the first valid real outage result

The 2026-11-15 local Gemma pilot produced a **structurally valid** hypothesis
for a reported Microsoft service outage: NEGATIVE / MODERATE / 55,
SHORT_TERM, HIGH uncertainty. This does **not** establish financial losses,
customer attrition or a justified 55-point market reaction.

Additive `evidence_assessment` fields in `news_impact_signal_v1` keep
three distinct kinds of information:
- `reported_fact`: the literal already source-validated event excerpt,
  **reported**, not independently verified just because the source wrote it
- `conditional_mechanism`: the previously validated *hypothesis* from
  first-stage news analysis, not an established market reaction
- `verification_needed`: fixed independent-check checklist for service
  incidents (duration, affected user scope, restoration, business materiality),
  **not assertions that the cited article lacks those facts**
- `fact_status`: existing source event status, including UNVERIFIED
- `strength_basis`: HEURISTIC_UNCALIBRATED or NOT_ASSESSED

These fields are **constructed from already validated source/provenance and a
fixed checklist**, never from additional LLM facts. The legacy signal adapter
remains valid with no assessment, preserving existing read paths. For
UNVERIFIED or CANDIDATE service incidents a HIGH impact-strength grade is
rejected, not automatically down-scored; independently verified evidence
would be required for HIGH. MODERATE remains a subjective uncalibrated
news-pressure score, not an expected stock return.

The local pilot and automatic 15-minute jobs remain independent; this change
does not populate SQLite or certify Gemma 2's real-world accuracy. A new
manual Mac run is required to observe the upgraded fields in real responses.


## Opt-in isolated evidence journal for real two-stage pilot reports

Real Mac test confirmed the new `evidence_assessment` in the report for
event 69. That does not mean the result was independently fact-checked
or is safe to publish as a production news-intensity signal.

The opt-in `market-event-pilot-evidence` command parses an existing pilot
JSON report, opens the operating `news.sqlite3` in **SQLite read-only mode**,
and verifies event IDs, stored classification/status, source URL, literal
evidence quote, strong-score guardrails, stage call counts, and model/prompt
version. Only `TWO_STAGE_OUTPUT_VALIDATED` entries with structured
fact/hypothesis/review data qualify. Every accepted output remains
`REQUIRES_HUMAN_REVIEW` — it is not externally validated.

Preview safely, with no writes:

```bash
market-event-pilot-evidence \
  --report "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-two-stage-20261115T061043.json"
```

After reviewing this report, **explicitly** archive it into an
isolated, new private research journal:

```bash
market-event-pilot-evidence \
  --report "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-two-stage-20261115T061043.json" \
  --commit
```

The journal is a separate `news-pilot-review.sqlite3` database, not
`news.sqlite3`, `market_impacts`, or `news_pressure_signals`.
It stores the original report hash, a per-event source evidence hash,
immutable JSON signal and a fixed review-required status. Reimporting
the identical report is idempotent; conflicting records are rejected.
The API and 15-minute operational cycle are not changed. No
automated imports, stock data, Donghak access, or production release
occur. A distinct approved review and publish workflow must be
implemented before a news signal can become operational.


## Human editorial review of the isolated pilot journal (opt-in)

After archiving a real pilot report into `news-pilot-review.sqlite3`,
review the stored evidence without invoking any model:

```bash
"$HOME/opt/anaconda3/envs/tris-market-event/bin/python3.11" \
  -m tris_market_event_intelligence.news_pilot_review_ledger_cli --list
```

Select an evidence **journal row ID**, not the source event ID (for the first
archived signal this is normally 1). Review approval is **editorial research
approval**, NOT independent source verification, portfolio advice, publication,
or readiness to expose a production API. A self-declared reviewer string
is recorded but is **not authenticated identity**.

Preview an approval, then record it explicitly:

```bash
"$HOME/opt/anaconda3/envs/tris-market-event/bin/python3.11" \
  -m tris_market_event_intelligence.news_pilot_review_ledger_cli \
  --approve 1 --reviewer local-reviewer \
  --reason "Source quotation checked; conditional effect and uncertainty acknowledged"
```

Add `--commit` to the same command **only after human review**.
For unsupported hypotheses use `--reject 1` in place of `--approve 1`
with a specific explanation. Both write to a new append-only
`news_pilot_review_decisions` table **inside the separate research
journal**; the original signal is not rewritten. The same decision
is idempotent and conflicting decisions are rejected. Changes to the
source text, URL, verification status or archived signal hash block decisions.
The list command exposes a derived editorial state but always
`published=false`. No production DB write, API exposure, scheduler change,
automatic decision, or downstream repository access is performed.


## Read-only audit and status-filtered research review queue

The research evidence journal can be inspected without making a review
decision or changing the production news database. After installing the
latest package, run:

```bash
"$HOME/opt/anaconda3/envs/tris-market-event/bin/python3.11" \
  -m tris_market_event_intelligence.news_pilot_review_audit_cli
```

Filter to a single editorial state and page over journal **row IDs**:

```bash
"$HOME/opt/anaconda3/envs/tris-market-event/bin/python3.11" \
  -m tris_market_event_intelligence.news_pilot_review_audit_cli \
  --status REQUIRES_HUMAN_REVIEW --after-id 0 --limit 25
```

Supported states are `REQUIRES_HUMAN_REVIEW`,
`APPROVED_FOR_RESEARCH`, and `REJECTED`. A research approval never
represents independently verified economic facts and never publishes a
signal to the production API. All audit outputs include
`published=false`, `productionDbModified=false` and
`journalModified=false`. Counts cover the **entire** journal while the
paginated items reflect the filter and cursor.

For every entry, even when filtering or using a small page, the audit
checks the original news evidence (SQLite read-only), the stored
payload SHA-256, evidence SHA-256, decision-to-evidence hashes, review
metadata and timestamp. Orphaned, altered, or structurally invalid
decisions fail the audit without showing a misleading partial success.

A deterministic local evidence digest is for **consistency checks only**:
a person capable of editing both content and hashes can also recompute
it. This is not cryptographic authenticity, a complete independent
fact check, proof of real-world impact, or a release authorization.
The 15-minute collector, all live tables and all downstream
repositories remain unchanged.


## Read-only full evidence packet for a human reviewer

The audit queue is concise. To view the real source context, the quoted
reported fact, first-stage conditional mechanism, full final model rationale,
second-stage intensity/uncertainty, and unverified follow-up checks, use:

```bash
"$HOME/opt/anaconda3/envs/tris-market-event/bin/python3.11" \
  -m tris_market_event_intelligence.news_pilot_review_packet_cli \
  --evidence-id 1
```

The ID is the **journal evidence row ID**, not the upstream event ID.
The command validates **all** journal records before displaying the selected
packet. It rereads the source database in SQLite read-only mode and checks
provenance; it shows a bounded section of the original event evidence
surrounding the literal quotation rather than fetching a web article.
Source text remains an untrusted report. Its quote check does not
verify the underlying event happened. The independent verification
checklist always shows `verified=false` because no external fact-check
has been performed.

The packet is read-only and shows `published=false` even when
`APPROVED_FOR_RESEARCH`. It does not approve anything, write to
the pilot journal or production DB, call an LLM, or interact with
price prediction. The 0-100 intensity score is subjective news
pressure, not a forecast of stock returns.
