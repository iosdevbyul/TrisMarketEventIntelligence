# Market outcome gap analysis — PR #266

## Real source observation (2026-11-23)

User ran the merged #265 evaluator using actual local Gemma historical article output, operational news.sqlite3 and historical Yahoo Finance adjusted daily files.

- Original Gemma article reports: **19**; impact rows **41**.
- Source matched: **38**; original source incomplete/duplicate: **3**.
- Source-approved but lacking a reviewed market instrument link: **29 UNMAPPED**.
- Recorded daily market observations: **9** (7 scored directional rows and 2 ABSTAINED).
- 1-session absolute direction agreement **5/7**, 1-session benchmark-relative agreement **2/7**.
- 5-session absolute **4/7**, relative **4/7**.
- Naive always-UP agreement on 1-session *absolute* cohort **4/7**,
  always-DOWN **3/7**. At 5 sessions, always-UP **3/7**,
  always-DOWN **4/7**.
- Directional denominator **7**, not 41; those rows represent **5 articles**,
  **4 publication dates** and **5 potential story groups**.
- The generated forecasts were retrospective and unsealed, so none of these
  rates represent prospective accuracy. A 7-row convenience sample is too
  small for model uplift conclusions.

Source of these counts: actual command stdout supplied by the operator.
There is **no claim** that GitHub CI has the user's private data.

## New gap-review CLI

The next useful result is an unapproved target review work queue. Read the
**existing immutable** real outcome report; classify each stored impact as:

- `MAPPING_REQUIRED` — a completed source-backed original article with
  a free-text target that is **not yet approved** as an exchange instrument
  or defensible sector/index proxy.
- `NEWS_SOURCE_BLOCKED` — invalid/repeated original news, cannot be
  rescued by inventing an ETF.
- `INSUFFICIENT_MARKET_BARS` — a mapped target without sufficient
  stock/benchmark adjusted closing observations.
- `SOURCED_DAILY_OBSERVATION` — descriptively observed, not causally or
  prospectively certified.

The review queue groups exact unmapped target text, tracks original article
IDs, incidence and predicted UP/DOWN/NEUTRAL/UNCERTAIN counts, and sorts by
how many currently uncovered impacts would benefit from researching that
target. Unapproved `reviewed_symbol`, `reviewed_market`, benchmark,
verified timestamp, supporting URL and proxy limitation are deliberately
**blank**. The application must not fill these by guessing.

Historical results are rejected if missing their explicit
`researchOnly=true`, `trainingEligible=false` or prospective-null
contract. SHA256 of the source JSON is included in the generated report,
and the original JSON is never rewritten.

### Run on existing real data

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"

PYTHONPATH="$PWD/src" python -m \
  tris_market_event_intelligence.market_outcome_gap_review_cli \
  --report "$DATA/gemma-real-daily-outcomes-pr265-v1.json" \
  --output "$DATA/gemma-real-outcome-gaps-pr266-v1.json" \
  --review-csv "$DATA/gemma-real-unmapped-targets-pr266-v1.csv"
```

No additional RSS, Gemma or price-provider API calls are made. Both output
files are create-only.

The review queue is for later **human/primary-source-supported**
instrument adjudication, not automatic trading or statistical labeling.
Economic indicator, broad market, currency, rate, and multi-sector
claims may be legitimately unmappable to a single listed equity or ETF;
do not turn every rejected row into a proxy to improve a score.

## Matched baseline correction

The evaluator also now reports `relativeConstantDirectionBaselines`,
using exactly the scored benchmark-relative sample (not absolute stock
directions). Keep both baseline denominators visible before interpreting
Gemma versus trivial constant direction policies. Do not infer directional
causality or a confidence interval from the seven descriptive scores.

## Remaining work

Real reviewer-backed mapping evidence plus authorized corresponding
historical adjusted prices, independently authenticated first-public event
times, minute/hour reactions, genuinely sealed out-of-sample predictions,
family-disjoint cohorts, and baseline-beating performance remain unproven.
The Stage1 gating policy and research-only source provenance limitations
are unchanged.
