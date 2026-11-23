# PR #268: Real-target economic observability rather than arbitrary ETF mappings

## Empirical starting point

The actual 2026-11-23 operator cohort contains 19 original Gemma article
analyses, 41 directional impacts, **29 unmapped impacts over 25 distinct
target names**, three article-source-blocked impacts, and nine historically
observed adjusted daily market impacts. Only seven were non-abstained:
1-session absolute 5/7, benchmark-relative 2/7, both entirely
`RETROSPECTIVE_UNSEALED`. This is *not* prospective prediction accuracy.

The 25 unknown target strings were provided verbatim by the operator.
A large portion represent yields, inflation, GDP, lending, broad/foreign
markets, multi-component sectors or banks without independently identified
tradable shares. Assigning every string a US equity sector ETF would
change the meaning of the original Gemma forecast.

## Implemented typed observation inventory

`typed_market_targets_cli` reads the original source-preserving #266 gap
JSON and review queue and produces a new immutable target type inventory.

- GDP, CPI/inflation: `MACRO_RELEASE`, requiring official jurisdiction,
  statistical release timestamp, units and vintage/surprise. Not a stock
  price or a replacement for a daily ETF.
- Interest rates and Treasury yields: `YIELD_SERIES`, requiring currency,
  instrument, tenor and **yield vs bond-price direction semantics**.
  Bond yield UP is not automatically equity-price UP.
- Financial Markets / Equity Markets / Stock Market: `BROAD_MARKET`;
  market country, index and horizon must be specified. Two-country
  "Equity Markets (South Korea, Taiwan)" is `MULTI_MARKET`, needing
  individually justified index series.
- Lending / corporate credit / mixed sectors: dedicated
  `ECONOMIC_ACTIVITY`, `CREDIT_MARKET`, `MULTI_SECTOR` and
  `MULTI_VARIABLE` types; no universal stock price exists.
- Bank conversions, acquisitions, mutual holding companies and multi-bank
  source events: `BANK_SUBSIDIARY`, `CONVERSION_ENTITY`,
  `ACQUISITION_TARGET_BANK`, etc. No automatic publicly tradable ticker.

### Authoritative relationship evidence considered

- Federal Reserve [Nano Banc and related holdings enforcement action](https://www.federalreserve.gov/newsevents/pressreleases/enforcement20261105a.htm) identifies the grouped legal entities; it does not prove a single tradable asset.
- Federal Reserve [MF Bank conversion order, 2026-11-09](https://www.federalreserve.gov/newsevents/pressreleases/files/orders20261109a1.pdf) describes the MFB MHC/MFB Bancorp/MF Bank relationship; do not backdate future stock listings.
- Federal Reserve [BNS–MapleMark acquisition order, 2026-10-29](https://www.federalreserve.gov/newsevents/pressreleases/files/orders20261029a1.pdf) establishes target/subsidiary relationship, not that MapleMark itself has listed stock.
- [United Community Banks investor relations](https://ir.ucbi.com/) identifies NYSE `UCB` as the parent holding company for United Community Bank. **Only a candidate** equity exposure; a bank-level news claim must still be separately judged for parent price relevance and source-time validity.

The inventory retains exact original target, source article IDs,
impact frequency, proposed required measurement class, and available
specific entity evidence links. It does **not** populate a verified
`symbol`, fabricate macro prices or backdate event coverage. Every
unrecognized future target is `UNCLASSIFIED`, not guessed.

## Review safeguards enforced on existing #267 approval pipeline

The reviewed mapping CLI now requires `--semantic-plan` when any target
row is `APPROVE`:

1. Check plan schema, report SHA and exact target membership.
2. Check approved target/category/suggested parent symbol against the
   code's immutable catalog: a modified JSON cannot silently turn
   "Inflation" into a valid equity proxy.
3. The only equity review *candidate* presently identified is
   `United Community Bank` → `UCB` (parent exposure) and it is
   **not automatically approved**. Reviewer must provide true market,
   symbol, benchmark, evidence URL and truthful verification timestamp.
4. Macro releases, yields, broad markets, mutual banks and other categories
   cannot be approved through the single-equity price mapper. They
   require separate typed observation/series mapping in future
   development.
5. `PENDING` and `NOT_REPRESENTABLE` rows remain unapproved and
   the original source CSV and gap JSON remain unchanged.

Approval of UCB would not in itself improve the **7 scored** sample
because that report's original bank impact was `NEUTRAL`. Actual
historical prices and chronological source availability still need checking.

## Run on existing operator Mac artifacts

Run from a *new worktree based on updated main* or equivalent feature
checkout:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"

PYTHONPATH="$PWD/src" python -m \
  tris_market_event_intelligence.typed_market_targets_cli \
  --gap-report "$DATA/gemma-real-outcome-gaps-pr266-v1.json" \
  --review-csv "$DATA/gemma-real-unmapped-targets-pr266-v1.csv" \
  --output "$DATA/gemma-typed-market-targets-pr268-v1.json"
```

Read the new JSON's `classificationCounts`, `targets`,
`measurementRequirement` and per-entity evidence. This is a
source-type research gate: it creates **zero** new price
observations and changes no historical direction labels.

If an independently reviewed, source-backed UCB mapping is ever
entered into the existing editable CSV, invoke the #267 mapper with
`--semantic-plan "$DATA/gemma-typed-market-targets-pr268-v1.json"`.
Do not manually invent other equity proxies to increase denominators.

## Remaining required market-learning work

Source-backed official macro/yield/index series with source-vintage,
units and exact publish/observation timestamps; genuine real minute
price reactions for actual listed instruments; independent target/sector
links; immutable *prospective* source-first forecasts; independent
family/date holdout and improvement against baseline. Only actual
market outcomes, not merely semantic type classifications, can support
the project's directional-learning objective.
