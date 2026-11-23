# PR #271 — Same-source cohort market outcomes and actual observation maturity

## Real operator diagnosis

On 2026-11-24 (KST), the operator executed #270's optional real adjusted
daily quote collection and source re-evaluation. The CLI returned:

- New exact issuer mappings: **2**, NVIDIA `NVDA` and Capita `CPI.L`.
- Historical price observations fetched: **900** rows from `CPI.L`,
  `ISF.L`, `NVDA`, and `SPY`. No provider symbols missing.
- Latest actual quote date for *every* symbol: **2026-11-20**.
- Both original accepted source articles were dated **2026-11-23 UTC**:
  NVIDIA article 20975 (UP) at 00:08Z, and Capita article 20989
  (DOWN) at 10:46:56Z.
- Both correctly had `NO_ENTRY_SESSION`, `entryDay=null`; their
  1-/5-session returns and directional verdicts are **UNAVAILABLE**.
- Duplicate/other-source NVIDIA articles 20976/20979 were **blocked**
  by original provenance, not incorrectly re-awarded independent scores.
- The previous #269 report contained **90** impacts, while #270's
  subsequent automatic Gemma ledger query returned **103**. This is a
  different cohort; naïvely subtracting total scored or completed
  rows is not valid. New analyses likely occurred between snapshots,
  but the exact cause of 13 additional rows is not yet proven.

Thus **no scored improvement is expected** from the #270 price file
before future real trading bars are observed. A price fetch is not
evidence of an observed future return. These examples also remain
`RETROSPECTIVE_UNSEALED`, not prospective predictions.

## Functional change

`same_cohort_market_recheck_cli`:

1. Reads the immutable operator #270 `automatic-observed-outcomes.json`,
   the same source-reviewed map and recovered adjusted daily prices.
2. Re-evaluates the *current* automatic SQLite Gemma ledger without
   any model call, RSS collection, quote download or DB mutation.
3. Compares only exact **(article ID, impact index, original prediction
   JSON SHA256, target, direction)** identities appearing in *both*
   reports. Revised model outputs, changed source body, removed articles
   and new automatically analyzed articles **cannot** masquerade as
   more accurate predictions.
4. Reports source-cohort sizes and change counts separately. For
   1- and 5-trading-session horizons it reports scored before/after,
   **newly scored**, and **lost scored** rows on the same source cohort.
   It preserves verdicts in the full accompanying current report.
5. For `NO_ENTRY_SESSION` or `PARTIAL` rows, lists actual source-quote
   last-observed day per instrument and benchmark, original article
   release timestamp, entry day if observed, and individual horizon
   price statuses. No guessed future trading sessions or false
   accuracy.
6. Outputs two new create-only files: current source evaluation JSON
   and a same-cohort/maturity JSON with original source SHA, price SHA,
   mapping SHA and preserved research-only flags.

The existing `listed_equity_price_coverage_cli` is fixed to use this
same-cohort comparison when a provider fetch is authorized. It no longer
subtracts possibly different total cohort scored/observed counts
and incorrectly calls the difference `newDirectionScoresObserved`.

## Run once without fetching prices

From a checkout with the merged new feature:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
PR270="$DATA/real-listed-equity-pr270-v1"

PYTHONPATH="$PWD/src" python -m \
  tris_market_event_intelligence.same_cohort_market_recheck_cli \
  --previous "$PR270/automatic-observed-outcomes.json" \
  --db "$DATA/news.sqlite3" \
  --model gemma4:e2b \
  --mappings "$PR270/reviewed-mappings.csv" \
  --prices "$PR270/expanded-adjusted-daily.csv" \
  --price-manifest "$PR270/expanded-adjusted-daily.csv.manifest.json" \
  --current-output "$DATA/automatic-gemma-pr271-current-v1.json" \
  --output "$DATA/automatic-gemma-pr271-maturity-v1.json"
```

This is genuinely read-only against user production news and previous
price files. It does not fetch another 900 rows. It will likely show
`NO_ENTRY_SESSION` for the two 23-Nov articles until authentic completed
bars after release exist. No new empirical scores should be asserted
without those actual observations. Re-evaluations with future new quote
files are a *separate, explicitly authorized source fetch* step, and
the old evidence must never be overwritten.

### Remaining project charter objectives

The existing research is after-the-fact descriptive, not immutable
prospective forecasting. Future steps still need actual later close and
minute price source observation, verified first-public release time,
instrument-source eligibility, point-in-time prediction sealing,
disjoint holdout cohorts, and measurable improvement over matched
naïve baselines. The goal is not more taxonomy or a cosmetic
`marketPricesAnalyzed=true`.
