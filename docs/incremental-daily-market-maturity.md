# PR #272 — only NEW real market bars, then check exact identical forecast cohort

## Production observation

Real operator PR #271 verified a growing stored original-news ledger:

- Previous evaluation: **103** Gemma impacts.
- Current evaluation: **321**, including **218 new** impact records.
- All original **103** were unchanged and present, none removed/revised.
- Source-matched newly scored one-/five-session predictions: **0 / 0**.
- **Four** common mapped cases had price-horizon maturity blockers.

The price file created by PR #270 already contained **900 real source-provided adjusted daily closes** for NVIDIA NVDA, US benchmark SPY, UK Capita CPI.L and FTSE100 ETF ISF.L, with all four ending **2026-11-20**. Two source-admitted directional original news articles published **2026-11-23**, so scoring them from 20-Nov prices is forbidden. This fact must never be represented as a 0% forecasting performance.

## New actual-market function

`incremental_daily_market_maturity_cli` uses the user's original:

1. `automatic-gemma-pr271-current-v1.json` (most recent scored source-cohort JSON);
2. `news.sqlite3` (read-only, exact original article and model);
3. `real-listed-equity-pr270-v1/reviewed-mappings.csv`;
4. `real-listed-equity-pr270-v1/expanded-adjusted-daily.csv`;
5. `real-listed-equity-pr270-v1/new-listed-quote-targets.csv` (only the two source-admitted equity company candidates).

With the operator's **explicit invocation**, it calls the existing yfinance adjusted `interval=1d` data source for the **smallest interval starting at the last previously observed date** and stopping before an unfinished UTC calendar date. It does not download the entire Jan–Nov original price history again.

- Checks each instrument's exact last-observed date is present in the new source response and that its adjusted close matches the original adjusted close within a tight numerical tolerance. Any changed/revised adjustment causes an explicit failure, rather than creating an inconsistent stitched stock-return time series.
- Verifies quote symbols are exactly those in existing source-reviewed mappings. No guessed stocks, no new private entity or macro ETF approvals.
- Appends **only strictly later actual returned daily prices**, preserving the original 900 price rows and all unrelated existing symbols. No synthetic close, no weekend interpolation, no forward value fills.
- Preserves previous immutable original news, original price CSV and prior reports; writes an entirely new directory containing expanded actual-price CSV, hash-pinned update/source manifest, current operational Gemma outcome JSON and same-source score/maturity report.
- Compares the newest operational report with `--previous` on source-identical *(article ID, impact index, prediction-source SHA, target, direction)* only. This prevents newly added articles or revised Gemma answers being counted as a model performance increase.
- For each horizon reports newly scored, lost and before/after exact-cohort predictions. Also reports new source impacts and current cases awaiting genuine later sessions. `trainingEligible=false` and `prospectivePredictionAccuracy=null` remain.

### Mac command — real completed quotes only

Execute from a checkout of merged main:

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
  --end 2026-11-24 \
  --output-dir "$DATA/real-incremental-market-pr272-v1"
```

`--end` is **exclusive**; on 24 Nov UTC it can include only
completed 23-Nov exchange-day bars *if the provider genuinely supplies them*.
Always adjust the cutoff to a past, completed UTC day and a **new**
output directory for future runs. The existence of 23-Nov closes is
not guaranteed by this command. Because the daily event study enters
on the first close on a strictly **later** trading date and measures
subsequent session closes, 23-Nov closing prices alone will **not**
mature either 1-session or 5-session return for these 23-Nov news items.
Later closes must genuinely exist, with matching benchmark observations.

### Safety and limitations

A provider snapshot can revise historical adjusted close levels after
corporate actions. The strict overlapping-anchor check stops incremental
series construction on inconsistency. In that case, re-fetch a
**complete intact adjusted series** in a separately evidenced research
output and re-evaluate source-identical impacts; do not disable the
consistency check. Yahoo price observations are not independently
exchange-authenticated or a point-in-time historical archive.

Historical analysis is still `RETROSPECTIVE_UNSEALED`, does **not**
prove causal news impact or prospective prediction accuracy, and
cannot legitimately feed automatic model learning as a validated
ground truth.
