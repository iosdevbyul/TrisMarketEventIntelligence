# Actual news impact → documentary market mapping → sourced daily price expansion

## Ground truth from the operator's real Mac, 2026-11-23

The original source audit contains 858 stored articles, of which 786 have
complete original bodies and 699 have no currently listed source blockers.
The first original Gemma batch (19 article JSONs, 41 directional impact rows)
was compared to actual source-provided historical adjusted daily stock/ETF
prices under merged PR #265.

Merged PR #266 classified the **actual** existing report as follows:

| Outcome reason | Impact count |
| --- | ---: |
| Mapping required | 29 |
| Original news source blocked | 3 |
| Observed adjusted daily prices | 9 |

The 29 unlinked impact rows span **25 different target names**. The
operator has generated the complete CSV, but its target-name content has
not yet been presented for source-specific review. Hence **no new ticker
assignments, quote downloads or improved accuracy can truthfully be
claimed**.

In the original 7 scored, non-abstained examples, the 1-day
retrospectively-unsealed absolute direction agreed 5/7, while
benchmark-relative agreement was only 2/7. The cohort contains only
five articles and four dates. This is not prospective forecasting
accuracy or evidence of a causal news impact.

## PR #267: how a reviewed source can become a real market observation

The new `reviewed_market_mapping_cli` reads the **immutable** #265
outcome JSON, the corresponding #266 gap report, the editable #266
review queue CSV, the previous reviewed map, and the previous
source-provided adjusted daily price CSV. All inputs are read-only.

Its authenticity guards include:

1. The gap JSON's `inputSha256` must match the outcome JSON's actual bytes.
2. All originally exported target names, impact counts, article IDs,
   prediction direction counts and issue classifications must match
   the original gap report. Dropping targets, rewriting source facts or
   duplicating target rows is prohibited.
3. The reviewer chooses one of `PENDING`, `NOT_REPRESENTABLE`, or
   `APPROVE` in the existing `review_decision` column.
   No decision is silently converted to approval.
4. An `APPROVE` row must have the true exchange `reviewed_market`,
   `reviewed_symbol`, `reviewed_benchmark`, `mapping_kind`
   (`DIRECT_EQUITY` or `SECTOR_ETF_PROXY`), an HTTPS
   `mapping_evidence_url` and **truthful timezone-aware**
   `mapping_verified_at`. ETF proxies must include a specific
   `proxy_limitation`. Date must reflect documentary verification,
   **never** a fabricated historical source timestamp.
5. `PENDING` and `NOT_REPRESENTABLE` leave the reviewed instrument
   fields blank. An economic indicator, Treasury rate, mixed-industry
   claim, geopolitical event, private organization or ambiguous regional
   macro target need not have a single ticker and may remain unmapped.
6. Existing reviewed mapping entries cannot be replaced or rewritten by
   a new review. Outputs are new, create-only files.

The output is a combined target-map CSV accepted by the existing daily
evaluation engine, an **approved-only price collection map** for symbols
without any stored daily observations, and a JSON report listing missing
symbols, approved targets, denied targets and coverage caveats.

## Usage on the current Mac data

First inspect the original target names and their source article IDs in
`gemma-real-unmapped-targets-pr266-v1.csv`. The operator must inspect
primary issuer/exchange and ETF prospectus documents before filling any
approved row. Editing this CSV **does not constitute independent
third-party authentication** of the mapping.

On current updated main:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"

PYTHONPATH="$PWD/src" python -m \
  tris_market_event_intelligence.reviewed_market_mapping_cli \
  --gap-report "$DATA/gemma-real-outcome-gaps-pr266-v1.json" \
  --outcomes "$DATA/gemma-real-daily-outcomes-pr265-v1.json" \
  --review-csv "$DATA/gemma-real-unmapped-targets-pr266-v1.csv" \
  --mappings "$DATA/stage6-sector-target-map-v2.csv" \
  --prices "$DATA/stage6-sector-prices-v4-recovered.csv" \
  --output-mappings "$DATA/stage6-reviewed-target-map-pr267-v1.csv" \
  --output-price-queue "$DATA/stage6-new-price-collection-map-pr267-v1.csv" \
  --output-report "$DATA/stage6-reviewed-mapping-plan-pr267-v1.json"
```

Run the above first with untouched `PENDING` decisions if only an
inventory report is desired; no securities will be approved or requested.

### For approved targets missing actual daily price series

The separate `stage6-new-price-collection-map-pr267-v1.csv` contains
only **approved targets requiring additional price symbols**. If it has
no rows, do not request quotes. If it has rows, the existing optional
Yahoo Finance research adapter can obtain actual adjusted daily
observations for only those mapped symbols and their benchmarks:

```bash
PYTHONPATH="$PWD/src" python -m \
  tris_market_event_intelligence.market_price_collection_cli \
  --mappings "$DATA/stage6-new-price-collection-map-pr267-v1.csv" \
  --start 2026-05-20 --end 2026-11-23 \
  --output "$DATA/stage6-new-price-observations-pr267-v1.csv"
```

Do not run this merely to simulate real prices. This calls a third-party
provider and requires the optional `yfinance` package and permission to
access the relevant data. Missing instruments remain missing; the
provided period is a user-adjustable example and **not** a universal
historical coverage guarantee. Review the source manifest and ranges.

Then **preserve every original instrument's full existing price series**
with the repository's existing whole-symbol recovery policy, using old
historical observations as primary and the newly acquired quotes only
for wholly absent symbols:

```bash
PYTHONPATH="$PWD/src" python -m \
  tris_market_event_intelligence.market_price_recovery_cli \
  --primary "$DATA/stage6-sector-prices-v4-recovered.csv" \
  --fallback "$DATA/stage6-new-price-observations-pr267-v1.csv" \
  --mappings "$DATA/stage6-reviewed-target-map-pr267-v1.csv" \
  --output "$DATA/stage6-expanded-source-pr267-v1.csv"
```

The recovery CLI will reject missing requested symbols and keep each
instrument's source snapshot intact, retaining per-symbol lineage and
explicitly **not** guaranteeing complete session coverage. It also
generates a `.manifest.json` with the expanded price and map SHA256.

Finally rerun the genuine retrospective price evaluation using the new
map, expanded prices and that recovery manifest:

```bash
PYTHONPATH="$PWD/src" python -m \
  tris_market_event_intelligence.market_outcome_validation_cli \
  --gemma-dir "$DATA/main-validation-gemma" \
  --news-db "$DATA/news.sqlite3" \
  --mappings "$DATA/stage6-reviewed-target-map-pr267-v1.csv" \
  --prices "$DATA/stage6-expanded-source-pr267-v1.csv" \
  --price-manifest "$DATA/stage6-expanded-source-pr267-v1.csv.manifest.json" \
  --price-source "Yahoo Finance adjusted historical daily, whole-symbol source lineage" \
  --output "$DATA/gemma-real-daily-outcomes-pr267-v1.json"
```

Compare the full count of original impacts, how many received new
**actual** observations, scored direction coverage, and always-UP/DOWN
baselines on equal samples. New approvals may still fail because
historical bars or source timestamps are missing. Do not silently
drop those failures.

## Remaining independence restrictions

All existing original Gemma article predictions are
`RETROSPECTIVE_UNSEALED`. Historical source and provider agreement is
descriptive only. This PR does not authenticate the earliest public
event release, certify causal effects or prospective forecasts, or
prove direction-model improvement. `trainingEligible=false` remains.
