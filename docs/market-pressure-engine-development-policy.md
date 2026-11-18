# General Market Pressure Engine — product contract and development policy

## Objective

**TrisMarketEventIntelligence analyzes financial news and economic indicators
to determine whether the newly available information can exert conditional
POSITIVE, NEGATIVE, MIXED or UNCERTAIN pressure on individual companies,
industries and broader financial markets.** It explains possible transmission
channels, qualitative strength and uncertainty. It does **not** train on
stock prices, forecast actual security returns, or place trades.

This module is independent of any stock-return forecasting engine.
Historical event replay, evidence screening, LLM runtime abstraction and
regulator-case research are **supporting techniques**. Accuracy at classifying
FAA/FDA regulatory decisions is not the product acceptance criterion.

## Responsibilities

| Layer | Responsibility | Implemented |
| --- | --- | --- |
| Ingestion | Existing RSS collector and durable SQLite article/event storage | Reused unchanged |
| Candidate discovery | Read time-eligible news and economic events; suggest related clusters with input-bound indexes | Opt-in prototype |
| Evidence verification | Verify source IDs, excerpts, cutoffs, cited companies and release values; retain unverified status | Structural checks |
| Event understanding | Extract cited reported facts, forecasts, opinions and novelty status | Model-assisted, review required |
| Economic transmission | State conditional channel from event to cash flows, demand, costs, funding, discount rates, FX, etc. | Model-assisted, review required |
| Target attribution | Distinguish market, sector and named company, with company names constrained to cited excerpts | Implemented |
| Direction, strength, uncertainty | Existing Impact direction and uncertainty scales; add LOW/MODERATE/HIGH/UNASSESSED conditional pressure strength | Implemented; semantic accuracy not verified |
| Publication | Create-only JSON, opt-in immutable SQLite research record, read-only paginated FastAPI feed | Opt-in API implemented; operational activation pending |

The new general path reuses `impact.Impact` and the existing conservative
mechanism and unfinished-rationale validators. Its two-stage reasoning
adds new information without changing the prior REST API, SQLite schema
or previously merged impact calculations.

### Direction, strength and confidence are different

- Direction: `POSITIVE`, `NEGATIVE`, `MIXED`, `UNCERTAIN`.
- Qualitative strength: `LOW`, `MODERATE`, `HIGH`, `UNASSESSED`.
  This describes *potential economic pressure*, not the probability
  or size of an actual stock move.
- Uncertainty: `LOW`, `MEDIUM`, `HIGH` for the *hypothesis*, not
  source authenticity. If any analysis input was unverified, the
  current policy forces `HIGH` uncertainty across the analysis.
- An `UNCERTAIN` direction cannot be assigned a confident
  non-UNASSESSED strength.
- A HIGH strength conclusion cannot rest entirely on unverified
  reported sources. Even an official report does not independently
  verify economic causality, magnitude or whether a market priced it in.

Only the source quotation and some structural implications can be
mechanically authenticated. The correct economic direction and intensity
remain **unverified model judgments**.

## Local operator workflow

The existing private SQLite DB is **not modified by preview or by default run**.\nOnly an explicitly requested `--persist` run adds an independent research\ntable. First inspect candidates
without invoking a model:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
bash scripts/news-learning-review.sh market-pressure-cycle \
  --db "$DATA/news.sqlite3" --as-of 2026-11-18 --preview
```

A real model-run, requiring the locally installed `gemma4:e2b`:

```bash
bash scripts/news-learning-review.sh market-pressure-cycle \
  --db "$DATA/news.sqlite3" --as-of 2026-11-18 \
  --lookback-days 10 --max-groups 2 \
  --run --model gemma4:e2b \
  --output "$DATA/market-pressure-first-run-v1.json"
```

The output is create-only; repeated runs must use a different path
and be labeled as repeat experiments. The model can suggest up to 2
clusters by default, with at most 8 sources each; each cluster gets
one fact-extraction call and, if there are grounded claims, a second
impact-reasoning call. Grouping is unverified and may produce an empty
list rather than inventing a connection.

For reviewed source IDs, bypass clustering to target a particular
event through `market-evidence-pressure` with the existing `--db`,
`--article-id`, `--economic-event-id`, `--event-id`, `--subject`,
`--as-of` and `--preview|--run` arguments. A JSON evidence packet
also remains supported. Opt in to `--single-stage-compat` only to
compare earlier single-pass `ImpactResult` behavior.

## Critical evidence boundaries

- RSS title/summary is **not** proof of the original article body.
- All source publication times originate from existing DB metadata;
  they are not authenticated archived historical page snapshots.
- Candidate sources with missing, timezone-naive, malformed or
  outside-window publication timestamps are excluded.
- A forecast is not an observed outcome; a policy proposal is not
  an implemented action; approved products are not evidence of
  commercial success.
- An economic surprise is not inferred without both an explicit
  actual value and an explicit sourced expectation.
- Prompt text is untrusted source material and never eligible as
  operating instructions.
- Cited excerpts, source IDs, claim indexes and company names
  are checked against cutoff-visible inputs outside the model.
- A full score of structural checks does **not** prove correct
  sign, intensity, causality or actual stock-price movement.
- There is **no new background service or automatic trading**.
  Current collector and production API behaviors are unchanged.

## Verification and acceptance

Python CI checks schema, Ruff and unit tests including cutoff leaks,
future citations, fabricated fact quotes, unsupported market
expectations, fabricated companies, weak strength evidence, empty
grouping and malformed group indexes.

**Real Ollama inference on the user's machine is still required.**
Performance acceptance must use multiple families such as inflation,
rates, earnings, products, labor, geopolitics and industry supply shocks.
A holdout set should be independently curated from then-public
news/indicator facts, with *target-specific economic sign and economic
channel* reviewed by qualified analysts. Distinguish economic reasoning
from ex-post stock return labels, which are NOT learned in this module.


## Four critical integration findings from the first local real attempt

The first real on-device request reached the SECOND market-pressure
inference step, then Ollama returned HTTP 400. Without its response body,
the exact validation failure is not known. The derived Pydantic impact
schema has inheritance and composite references, which Ollama's schema
backend may reject. The Ollama-only impact call now requests portable
JSON mode and independently enforces the original full Pydantic model
and source/economic guards. Fact extraction and source grouping remain
schema-first. This change fixes the request path but **real Gemma4
compatibility remains unverified until the user's next successful run**.

The `market-pressure-cycle` CLI now reports sanitized HTTP status
and stable error code instead of a full raw network traceback. It never
claims a successful artifact after an aborted model call.

### Economic indicator release intake

The optional `market_indicator_releases` SQLite table records release
ID, source time, period, original actual value, optional sourced previous
value, and optional **pre-release independently sourced consensus**.
The actual and consensus quotes and URLs are separately preserved;
consensus data published after the official release is rejected.
Unknown expectations remain absent, rather than invented from the actual.

An explicit JSON batch must follow
`market_indicator_release_batch_v1`, with an array of releases with:
`releaseId`, `indicatorName`, `period`, timezone-aware
`publishedAt`, `actual`, `actualSourceName`, `actualSourceUrl`,
`actualExcerpt`, and optional `previous`. Consensus is optional, but
if present requires `consensus`, `consensusPublishedAt` **earlier**
than the official release, `consensusSourceName`,
`consensusSourceUrl`, and `consensusExcerpt`. Every numeric value
must appear in its referenced excerpt. This is a general **verified
input-schema and import path**, not a completed automated BLS/FRED
or market-consensus vendor connector.

```bash
bash scripts/news-learning-review.sh market-indicator-import \
  --db "$DATA/news.sqlite3" --input /path/to/reviewed-releases.json --preview
# Explicitly opt in to immutable import, once the sources are checked:
bash scripts/news-learning-review.sh market-indicator-import \
  --db "$DATA/news.sqlite3" --input /path/to/reviewed-releases.json --import
```

Import never edits existing `economic_events` or `market_impacts`.
A structured release can be analyzed by
`market-evidence-pressure --indicator-release-id 1` (example ID).
Auto event discovery also includes this table when it exists.

### Full-history candidate filtering and UTC cutoff

Candidates are scanned against the requested publication window
**before** truncating for model context; they are no longer restricted
to the latest handful of record IDs. Their timezone-aware timestamps
are compared as absolute UTC instants. Exact-time replay may provide
`--as-of-time 2026-11-18T14:00:00+09:00`, consistent with
`--as-of 2026-11-18` in UTC-date terms. The `--page 0`,
`--page 1`, etc. controls bounded 24-source model batches without
silently claiming full search coverage. Preview reports available
page metadata, the number of candidate matches and deduplication.

This is still a bounded, unverified candidate relevance search, not
a proven comprehensive discovery or causal-priority ranking engine.
No article published after the precise cutoff may be sent to the model.

### Research persistence and REST delivery

The new append-only, independently keyed
`market_pressure_research` table is created ONLY with opt-in
`--persist` after a successful model run. The original `articles`,
`economic_events`, `market_impacts`, source captures and older
research artifacts are not migrated or overwritten. Duplicate
source/model/cutoff submissions are idempotent; conflicting outputs
are rejected. Full report JSON and its SHA-256 are kept for audit.

The existing FastAPI service exposes a read-only cursor endpoint:

```http
GET /api/v1/market-pressure-research?after_id=0&limit=100
```

Before explicit initialization, this endpoint returns 503 rather than
fabricating a successful empty report. API GET neither starts Gemma4
nor performs database migration. A successful response is a
review-only pressure hypothesis, NOT a validated trading signal.

To opt in to persistence on the same completed model run:

```bash
bash scripts/verify-market-pressure.sh 2026-11-18 --run --persist
```

Do not call this automatically from the existing operational scheduler
until the actual Gemma4 inference has been reviewed.

### Remaining acceptance blockers

- Successful local Gemma4 real-data run on varied CPI, monetary policy,
  earnings, supply chain, and opposing sector-impact cases.
- Independent human evaluation of direction, qualitative intensity,
  economic transmission and contradictory sources (not mock data alone).
- Automated authenticated actual/consensus data retrieval and original
  publication-time verification, rather than only validated imports.
- A safe operational schedule, event lifecycle integration and consumer
  E2E validation of the API; opt-in read-only endpoint alone is not
  production readiness.

Future integration priorities:
1. Validate real Gemma4 end-to-end outputs against independent
   economic transmission reference examples and diverse topics.
2. Expand verified economic-series source adapters beyond currently
   stored event and RSS releases; establish precise publication/PIT.
3. Add robust event deduplication and article body evidence quality.
4. Add audited output APIs and versioned persistence, but preserve
   existing API/DB contracts until reviewed.


### Ollama natural-language abstention compatibility (first real v2 failure)

The second observed local run returned a well-formed JSON response with
`noImpactCategory="No Defensible Economic Transmission Channel"`. Ollama's
portable JSON response mode does not enforce enum values; a direct Pydantic
`Literal` validation raised an exception **before writing a report**.
This is an output-shape error, not evidence that an economically justified
POSITIVE or NEGATIVE pressure was identified.

The pressure-stage prompt now requests the five exact category machine codes
and separates them from the readable `noImpactExplanation`. For a nonempty,
unrecognized string category on an **empty** impacts array, the parser
conservatively maps to `OTHER`, preserving the exact raw string in
`noImpactAssessment.rawModelCategory` and flagging
`categoryNormalized=true`. Neither economic direction, evidence, strength
nor the explanation is fabricated. Missing/short explanation,
contradictions between an impact and abstention, non-string categories,
or invalid impacts **still fail validation**. Only the category compatibility
field is normalized. No change is made to previous immutable results.

The next default local run writes
`market-pressure-first-real-v3.json`; older v1/v2 filenames remain
untouched. CI tests include the exact natural-language category form that
caused the actual failure, unrelated unknown strings, correct enum inputs,
conflicting impacts, and malformed explanations. Full market-impact judgment
quality remains to be tested on the user's installed Gemma4.


### First real v4 run: zero accepted pressures despite fact extraction

The user ran the local `gemma4:e2b` market-pressure cycle and shared its
review-only summary. It completed **five actual model calls** and produced
two event groups. An article about solar adoption in Pakistan was rejected
at structural model validation, with a root `value_error` (the precise
cause is NOT established from the sanitized report). A group of Nvidia
earnings and leadership articles produced three source-bound claims but
returned an explicit `NO_DEFENSIBLE_ECONOMIC_CHANNEL` abstention.

This is **NOT evidence that those economic events exert zero market
pressure**. The missing conceptual distinction was that a publisher
reports the underlying event, while economic transmission is an
explicitly labeled conditional *inference* by this module. The source
need not contain a stock-market explanation for a legitimate,
high-uncertainty economic mechanism to be considered. The v5 prompt
clarifies that distinction across all corporate, macroeconomic,
infrastructure, supply and policy topics, without hardcoding a news
story or answer direction.

A single source cannot establish information novelty relative to
earlier coverage. An unsupported `NEW_INFORMATION` label is therefore
downgraded to `UNKNOWN` while preserving the verbatim quoted fact;
an invented quotation, company or future source is still rejected.
Empty optional abstention placeholders in native Ollama JSON are
treated as absent, not as conflicting economic judgments; nonempty
and incomplete abstention claims remain invalid. Regression tests
cover both boundaries.

The v4 research JSON is preserved. The next opt-in local experiment
writes create-only `market-pressure-first-real-v5.json`; it requires
human review of whether inferred directions, strengths, sources and
affected targets are actually coherent. This is not a semantic
accuracy result and must not automatically enable persistence.
