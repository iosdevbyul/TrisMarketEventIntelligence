# PR #275 — One real-price refresh, two rigorously separate evaluation cohorts

## Market-learning goal, not infrastructure expansion

The governing `PROJECT_CHARTER.md` makes real subsequent stock/index
returns and forecast-direction validation the primary project deliverables.
The actual Mac run of #271 showed **103 unchanged prior Gemma impacts** and
**218 newly created** impacts (321 total) but no newly scoreable 1d/5d
outcomes on the 2026-11-20-ended price file. #272 correctly refused an
attempt to append nonexistent daily closes. PR #273 then added opt-in
locally immutable *new-inference* commitments; #274 added independently
invocable read-only evaluation of the locally committed horizon/target
against actual adjusted stock and benchmark daily closing prices.

The missing operational linkage was that the real daily bar *refresh*
workflow still computed **only retrospective Gemma outcomes**, so the
operator had to manually run a separate command over another quote
snapshot to inspect the new commitments. This PR closes that gap.

## Integrated price → market-outcome workflow

`incremental_daily_market_maturity_cli` now supports opt-in
`--include-committed-forecasts`. After the user explicitly requests a
fresh completed-date price collection, the existing source-provenance,
overlap and adjustment validation appends **only** actual later completed
daily observations for previously reviewed symbols and benchmarks.

Against the exact newly staged, SHA-pinned CSV, the command now evaluates:

1. **Historical analysis cohort**: the existing automatic Gemma retrospective
   results with source-identity, real 1d/5d stock/benchmark returns, and
   exact same-source before/after comparisons; never prospective accuracy.
2. **Newly precommitted cohort**: the #274 immutable local commitment ledger
   with exact original-body/model/quote/clock integrity and precommitted
   target+1d/5d horizon, market-local future sessions and benchmark
   returns; independently labeled candidate, **not** a third-party
   timestamp authenticated prediction.

Both evaluation results share the **same exact price SHA** and live in
a single create-only output directory:
`expanded-adjusted-daily.csv` and its source-SHA manifest;
`automatic-observed-outcomes.json` for retrospective historical results;
`committed-forecast-outcomes.json` for opt-in local precommitments;
`same-cohort-coverage.json` with per-cohort coverage and report digests.
If source hash, price, or commitment integrity fails, **nothing is
published**, and all existing files and the news DB remain unchanged.
The option does not enable additional Gemma inference or ongoing tasks.

The option defaults **off**. An operator who has never opted into #273
will get an honest `NOT_OPTED_IN, committedRows:0` report, **not**
invented sealed forecasts.

## Matched-baseline improvement

The #274 committed-only market evaluator now provides, for each 1d and 5d
horizon, on exactly the same observed, non-abstained, locally admissible
forecast rows:

- Absolute correct/incorrect, abstained, eligible denominator, and
  locally observed match fraction.
- Market-relative correct/incorrect and locally observed match fraction
  on **identical** denominator.
- Confusion matrix of actual UP/DOWN/NEUTRAL by Gemma direction.
- Constant `alwaysUp`, `alwaysDown`, `alwaysNeutral` correct counts for
  absolute and benchmark-relative returns on identical rows.
- Distinct source articles and publisher-stated calendar dates in the
  observable cohort.

These constant-class baselines are *descriptive*, not an already approved
out-of-sample experiment, nor proof of model improvement. Historical and
committed populations are never merged into a single accuracy statistic.
There is no historical source repair, retroactive sealing, model
training, automatic rule promotion or trading.

### Example Mac execution after genuine later close observations exist

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
PR270="$DATA/real-listed-equity-pr270-v1"

PYTHONPATH="$PWD/src" python -m \
  tris_market_event_intelligence.incremental_daily_market_maturity_cli \
  --previous "$DATA/automatic-gemma-pr271-current-v1.json" \
  --db "$DATA/news.sqlite3" \
  --mappings "$PR270/reviewed-mappings.csv" \
  --prices "$PR270/expanded-adjusted-daily.csv" \
  --quote-targets "$PR270/new-listed-quote-targets.csv" \
  --model gemma4:e2b \
  --end "$(date -u +%F)" \
  --include-committed-forecasts \
  --output-dir "$DATA/real-committed-market-pr275-v1"
```

`--end` is exclusive. **Only invoke when fully completed NEW sessions are
available** (the source must have actually published valid adjusted
daily closes); no trading days within the query window produce a
descriptive error, not a model failure. The output directory must not
already exist. This is a one-shot explicit user-authorized price-provider
query, not an automatic background or 24/7 activation.

If a real commitment has not been created via `--seal-forecasts`, this
operation cannot invent one. News first-public availability still lacks
independently authenticated source-time evidence, and local SHA/SQLite
sealing does not establish legally or statistically verified prospective
performance. `prospectivePredictionAccuracy=null`,
`trainingEligible=false` remain non-negotiable. A later phase must
obtain independent publication-time attestations, reliable later market
trading prices and disjoint event-family out-of-sample holdout before
claiming learning success.
