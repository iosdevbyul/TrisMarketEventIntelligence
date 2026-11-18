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


### Critical diagnostic correction after the real v5 rejected run

The user's v5 Gemma4 run completed five model calls and both event groups
were rejected with `MODEL_SCHEMA_INVALID` and a root-level
`value_error`. The previous safe response omitted the precise validator
reason AND erased the successfully extracted first-stage claims in the
rejected-group summary. Therefore `factsCount: 0` in that report **does
not prove Gemma4 extracted no facts**. The specific cause of the root
validation rejection must not be guessed.

Further prompt tuning without the underlying model response is suspended.
An opt-in **private, create-only, durable 0600 JSONL trace** now captures:

1. Source candidate inventory and selected IDs (with UTC cutoff).
2. Original model system/user prompts and schema for each event-group,
   fact-extraction and impact-generation call.
3. Original raw Gemma4 reply, call number, stage and SHA-256 checksum.
4. First-stage fact-validation checkpoints, with grounded claims.
5. Exact Pydantic field locations, public error types and full local
   validation errors/traceback after rejection. An unchanged partial trace
   remains on disk if any model call or validator throws.
6. Final per-group validation state and aggregate run status.

Full prompts and replies are intentionally kept ONLY in the private
local diagnostic file. They may include original news excerpts and
other sensitive source metadata; **never commit or upload the full trace**.
The ordinary service API and public result JSON do not include raw
model prompts or full error tracebacks.

```bash
git pull --ff-only origin feature/general-market-pressure-engine
# One instrumented run; no --persist; creates v6.json + v6.json.trace.jsonl
bash scripts/verify-market-pressure.sh 2026-11-18 --run

DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
TRACE="$DATA/market-pressure-first-real-v6.json.trace.jsonl"
bash scripts/inspect-market-pressure-trace.sh "$TRACE"
# The following intentionally prints full private prompts and model output.
# Use locally when examining the precise reason; do not post publicly.
bash scripts/inspect-market-pressure-trace.sh "$TRACE" --raw
```

If `v6` already exists, use a fresh `MARKET_PRESSURE_OUTPUT`
environment variable to a new path; the corresponding default trace
name is derived from that output path. Both are create-only.

This phase is **root-cause diagnosis**, not another attempt to force
positive or negative predictions. After inspecting the actual raw
response, fix the observed validator/prompt mismatch rather than
adding arbitrary story-specific exceptions.


### Root-caused v6 impact/abstention conflict — disjoint outputs

The actual local v6 trace proves both groups successfully passed fact extraction:
the Pakistan solar report yielded 2 grounded claims and the Nvidia news
group yielded 3 grounded claims. **Both impact-stage raw responses were
then rejected by the same root-level Pydantic validator**
`IMPACT_ABSTENTION_CONTRADICTION`, not by missing facts.
The request schema simultaneously advertised `impacts`,
`noImpactCategory` and `noImpactExplanation`, allowing the model to
fill both an impact proposal and a mutually exclusive abstention reason.
The validator correctly rejected this inconsistent contract.

The Ollama pressure generation schema is now strictly
`{"impacts":[...]}` **without any abstention properties**. A nonempty
pressure response is fully checked for source indexes, verbatim excerpts,
real company targets, uncertainty, and economic transmission.
Only if the validated impact list is empty is a **separate native model
request** made for `noImpactCategory` and `noImpactExplanation`;
the separate schema cannot contain `impacts`. The extra request is
conditional, never used for a nonempty accepted pressure result.

Any unexpected abstention field inside the impact response is explicitly
rejected, never silently stripped. This change addresses the observed
dual-output root cause rather than weakening evidence validators or
hand-labeling particular Pakistan or Nvidia news stories. Full local
stage-by-stage trace stays enabled, and the next create-only experiment
uses `market-pressure-first-real-v7.json`.

This is a structural contract and CI fix; successful economic reasoning
by the real on-device Gemma4 model is **not yet established**.


### Real v7: known validator codes, unknown offending model values

The user ran real Gemma4 v7 on the private SQLite news DB. Two groups
successfully extracted 2 Pakistan solar and 3 Nvidia source-grounded
claims, respectively. Pakistan was rejected by
`IMPACT_ENTITY_MISMATCH` (an `entities` name does not literally occur
in the cited claim excerpts). Nvidia was rejected at `impacts.1` by
`UNCERTAIN_DIRECTION_HAS_INDETERMINATE_STRENGTH` (the model's
`direction=UNCERTAIN` must be paired with
`strength=UNASSESSED`). Both are *real* downstream evidence/semantic
guard failures, not a failure of Ollama schema transport or fact extraction.

The ordinary result intentionally omits model-authored unverified entity
names and full generated response. Do not guess WHICH entity failed, nor
silently assign strength to an uncertain direction. **Before changing
validation or asking the user to rerun the model**, inspect the ALREADY
CREATED v7 private trace with this local-only, no-model-call command:

```bash
git pull --ff-only origin feature/general-market-pressure-engine
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
python3 scripts/diagnose-market-pressure-trace.py \
  "$DATA/market-pressure-first-real-v7.json.trace.jsonl"
```

It reports, per impact index and model call, exactly which model `entities`
are absent from its own cited claim excerpts, source index integrity,
whether its verbatim evidence excerpt is actually contained in the
referenced claim, and any UNCERTAIN-vs-strength contradiction. It does
NOT execute the model, edit the DB, replace the JSON or reveal full
article text. The user can share this much smaller diagnosis for a
precise corrective patch instead of repeatedly running v8/v9 blind.


### Economic semantic review of recorded market pressure — offline, human-gated

The local v7 replay recovered two **schema-/provenance-accepted** hypotheses
out of four. This is not the same as verifying real economic significance,
market-direction correctness, investor expectation effects, or magnitude.
In particular, `POSITIVE/MODERATE` solar-installation pressure and
`MIXED/MODERATE` Nvidia pressure are both reviewable hypotheses; the
model's label alone does not establish their actual strength.

A new read-only `market-pressure-semantic-review` workflow reads the
**existing private model trace** and runs the same strict replay checks,
then builds a create-only, source-linked JSON review worksheet for **each
original proposal**, including rejected ones. Fields include:
- original proposal index, target, market, sector, direction, horizon,
  qualitative strength and transmission channel;
- complete model-authored rationale and separate strength rationale;
- verbatim cited source observations, per-source verification/claim kind,
  disclosed actual/previous/consensus metadata, and link to source IDs;
- model drivers, countervailing factors, structural validator disposition
  and safely coded rejection reason;
- explicit warning flags for single-source support, unverified reporting,
  qualitative strength lacking independent audit, unexplained opposing
  forces, and unsourced surprise comparisons;
- six *unset* human-review questions and a required free-text justification.

The warnings are **review prompts, not an automated semantic correctness
score**. A schema pass is never converted into independently validated
causality, direction or strength. Human finalization is separately
explicit and rejects `PENDING`, incomplete criteria, missing explanation
or claims of full support with failed criteria. Finalized judgments
include a checksum of the original trace and canonical review file, but
remain human opinions, **not independent economic accuracy validation**.

Offline commands (no model calls, no DB changes):

```bash
git pull --ff-only origin feature/general-market-pressure-engine
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
bash scripts/news-learning-review.sh market-pressure-semantic-review \
  --trace "$DATA/market-pressure-first-real-v7.json.trace.jsonl" \
  --output "$DATA/market-pressure-v7-semantic-review-v1.json"
```

Review the private JSON's `reviewItems` carefully and edit every
`humanReview` value. Valid `humanDecision`: `SUPPORTED_WITH_LIMITATIONS`,
`NOT_SUPPORTED`, or `INCONCLUSIVE`. All six booleans and a reason at
least 30 characters long are required.

```bash
bash scripts/news-learning-review.sh market-pressure-semantic-finalize \
  --review "$DATA/market-pressure-v7-semantic-review-v1.json" \
  --reviewer "Human researcher" \
  --output "$DATA/market-pressure-v7-semantic-evaluation-v1.json"
```

Both outputs are local, create-only review artifacts and may contain
private original excerpts. Do not commit the results, send them to
external services or conflate them with the production market-pressure
API. PR #251 stays Draft until independent evaluation and operating
conditions are reviewed.


### Economic Reasoning Quality Gate — deterministic semantic review diagnostics

The supplied real v7 rationale review revealed an *additional* failure class
after the citation/JSON guards passed: reasoning that is plausible in broad
terms but does not substantiate its named channel, magnitude or forward
economic exposure.

The common, provider-independent `market_pressure_quality_gate.py`
attaches an explicit `reasoningQuality` object to every **source-validated**
impact. This does not change the model's direction, strength, selected
channel, cited evidence or rejection outcome. Its possible review concerns
include:

- `REGULATORY_CHANNEL_WITHOUT_REGULATORY_EVIDENCE`: reported CEO/board/
  governance language routed through `REGULATORY_OPPORTUNITY` despite no
  regulation or licensing mechanism in the cited observation or rationale.
  This flags the actual v7 Nvidia second hypothesis, though its original
  structural direction/strength rejection remains authoritative.
- `MIXED_DIRECTION_REQUIRES_TWO_SIDED_TRANSMISSION_REVIEW`: a MIXED label
  is not proof of both positive earnings and negative risk-premium paths.
  Verify each distinct direction and its conditioning claims.
- `STRENGTH_JUSTIFICATION_LACKS_ECONOMIC_MAGNITUDE_BRIDGE` and
  `QUALITATIVE_STRENGTH_NOT_EMPIRICALLY_VERIFIED`: the size of an event,
  deployment rate or CEO transition cannot be confused with a calibrated
  MODERATE/HIGH economic impact on the target.
- `OBSERVED_ADOPTION_NOT_EVIDENCE_OF_FUTURE_REVENUE`: observed adoption/
  electricity share (e.g. Pakistan 1% to 28%) does not independently prove
  future manufacturer orders, installation profits or equity response.
- `SINGLE_CITED_SOURCE_LIMITS_CORROBORATION`,
  `REPORTED_NEWS_NOT_INDEPENDENTLY_CONFIRMED`, and
  `PRIOR_MARKET_EXPECTATIONS_UNAVAILABLE` identify missing constraints.

These flags are **conservative review questions, not hard-coded economic
truth or an accuracy classifier**. Missing keywords do not establish that
a causal mechanism is false, and present keywords do not prove its
correctness. The gate explicitly returns
`HUMAN_SEMANTIC_REVIEW_REQUIRED`, `economicAccuracyVerified=false` and
`observedMarketReactionVerified=false`; it never manufactures a score
or repairs the model's unsupported strength/direction/channel.

The existing offline `market-pressure-replay` now includes these flags
for accepted proposals. The `market-pressure-semantic-review` worksheet
includes them for **every original proposal, including those rejected by
structural guards**, without a new Ollama call or DB write. This preserves
full root-cause context, makes the next human review materially useful,
and keeps all research in Draft PR #251 until reviewed.


## Architecture reset: arbitrary changing news, no article-specific rules

**The project objective is generality across continuously changing news and
economic releases, NOT passing tests for any one article.** The legacy
`market_pressure_quality_gate.py` keyword checks are retained as historical
diagnostics for old replay only. They are **not used to decide the current
general causal path engine output**.

The production `market-pressure-cycle` CLI now defaults to
`--reasoning-mode causal` and calls
`market_pressure_causal_engine.analyze_general_causal` for each discovered
event group, after the existing provenance-controlled news/indicator discovery.
The old `--reasoning-mode legacy` is available only for explicit comparisons;
none of its event-shaped text checks or mutable strength labels are the new
engine's contract.

The new **General Causal Paths V1** contract is data-driven, not keyed by
issuer, location, specific event category, or article text:

1. **Source-bounded observations:** extract exact citations only from sources
   visible before the cutoff; reported claims remain unverified claims.
2. **Conditional causal paths:** model proposes bounded ordered steps with
   cause, economic channel, effect and explicit assumption. Different paths
   can explain positive or negative effects for an affected market, sector,
   or literally-cited company.
3. **Independent strict guards:** validate every path's cited claim indexes,
   verbatim observations, company grounding, schema and bounds. Reject only
   invalid proposals; retain the original indexes of both rejected and accepted
   items. Do not silently rewrite model answers or force named companies.
4. **Derived direction:** Python computes POSITIVE/NEGATIVE/MIXED from the
   submitted independently cited causal-path signs, not from an ungrounded
   model label; a single counterargument does not magically count as an
   opposing economic path.
5. **No invented calibrated strength:** all outputs remain UNASSESSED and HIGH
   uncertainty until independent evidence supports a measured magnitude;
   no returns, stock-price predictions or trades are generated.
6. **No false verification:** conditional causal paths remain unverified
   economic hypotheses. Success under schema guards is not economic truth,
   generalization or factual verification.

Actual new mode is separately labeled `GENERAL_CAUSAL_PATHS_V1`. For
security, persistence of causal research via `--persist` is **disabled
even for fully schema-valid results** until the user explicitly approves a
semantically reviewed production boundary. New outputs remain private and
create-only. Prior v7 trace data is left untouched.

### Cross-event evaluation on the real rolling source DB

Use the new `market-pressure-generalization` command to inspect several
distinct bounded chronological source pages rather than the same article.
It reads stored news/indicators across the requested cutoff and lookback
period and reports source type/publisher coverage, model-call count,
event group count, number of proposed, grounded and rejected causal paths
and distributions of inferred directions. It explicitly does **not** report
economic accuracy or a verified holdout score because no independent
human economic labels were supplied. Source overlap across pages is
treated as invalid; future-dated sources are excluded.

First inspect a real DB without launching the model:

```bash
git pull --ff-only origin feature/general-market-pressure-engine
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
bash scripts/news-learning-review.sh market-pressure-generalization \
  --db "$DATA/news.sqlite3" --as-of 2026-11-19 \
  --lookback-days 30 --pages 4 --preview
```

The independent next stage (ONLY after confirming candidate coverage and
resource budget) is one bounded `--run` with a create-only new JSON output.
It uses collected sources, not hard-coded example stories. Synthetic tests
cover different macroeconomic, company, sector, transport and regulatory
topics and adversarial cite/strength/target failures, but are **contract
tests, not claims of independently validated real-model generalization**.

**Review gate:** any future attempt to declare the new engine complete must
sample independent unseen events and use external/human evidence-based
assessment of economic coherence, factuality and directional conditions.
Failure on fresh events is a system-level issue, not grounds to add
article-specific exceptions or loop over only the old two articles.


### Sampling validity after the causal-engine reset

A live generalization study cannot be credible if its discovery pagination
silently repeats or drops records. Prior logic fetched growing prefixes per
table and globally sorted them anew on each page. The candidate roster
therefore changed when advancing to the next page. Discovery now builds one
deterministic cutoff-visible ordering across article, event and indicator
release rows, applies source-level de-duplication consistently from the
beginning, then uses disjoint fixed-size pages. Regression tests interleave
more than 70 records across article and event tables to verify exactly-once
page coverage.

A second `source_balanced` sampling mode deterministically interleaves
chronologically ordered candidates across source TYPES (not news content,
ticker, publisher name, economic event wording or article IDs). This exposes
scarce economic releases rather than filling every sampled page with the
newest dominant RSS source. Normal daily `market-pressure-cycle` discovery
remains chronological by default; only `market-pressure-generalization`
uses the source-balanced mode. Every page stays deterministic and source
IDs must not overlap.

The evaluation report now retains bounded, original conditional causal
research items for independent human review (per-path cited observation,
economic steps, assumptions, opposing explanations and missing evidence),
alongside accepted/rejected structural counts. It also reports:
- sampling mode, source-kind counts, per-publisher counts,
- number and fraction of sampled candidates actually grouped/analyzed,
- sampling warnings when publisher or source-kind diversity is insufficient,
  or most candidates were not processed by the model.

The evaluation's `uniqueEligibleSources` is the count of **sampled
candidates within the page budget**, not exhaustive universe coverage.
Neither many sampled articles nor multiple publisher names prove
independence of the underlying economic events. Reviewers must inspect
actual causal-path content and independently verify event diversity and
source claims. The generalization flag remains false until that happens.


### One-shot generalization execution and durable model diagnostics

The multi-page generalization CLI now emits a local PRIVATE trace
automatically for `--run`:
`NEW_RESULT.json.trace.jsonl`. It uses the same create-only 0600
`MarketPressureTrace` and `TracedMarketProvider` as the older pilot;
every native JSON-schema request, model response, validation error and
truncation/transport failure is preserved before the process exits.
The `CAUSAL_PATH_GENERATION` stage is recognized separately from the
legacy impact-generation stage. Do not upload this trace or commit raw
source text to the repository.

Inspect candidate diversity *once* before making the Mac call:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
bash scripts/news-learning-review.sh market-pressure-generalization \
  --db "$DATA/news.sqlite3" --as-of 2026-11-19 \
  --lookback-days 30 --pages 4 --preview
```

If the preview shows an acceptable spread of stored source kinds and
publishers, one bounded real Gemma4 study is:

```bash
bash scripts/news-learning-review.sh market-pressure-generalization \
  --db "$DATA/news.sqlite3" --as-of 2026-11-19 \
  --lookback-days 30 --pages 4 --run --model gemma4:e2b \
  --output "$DATA/market-pressure-generalization-20261119-v1.json"
```

This is never an automatic benchmark PASS. Study the original novel causal
paths and their uncertainty rather than merely the schema acceptance count.
A failed run can be diagnosed from the same private trace rather than
blindly calling the model again. The code does not perform automatic
persistence, return prediction, or live trading.
