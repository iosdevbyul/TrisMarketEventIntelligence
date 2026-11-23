# Automatic original Gemma ledger → observed real-market return evaluation

## Why this new PR belongs to the Market Learning charter

On 2026-11-23 the operator confirmed that PR #268 classified all **25**
previously unmapped target names (29 impact rows), but applied **zero**
new ticker mappings or prices. The live `automatic_news_pipeline` with
`--skip-collection --watch --interval 900` also reported repeated
batches of **10 successful Gemma analyses, zero failures**, while
`marketPricesAnalyzed=false` for every run. This is expected: the
worker processes text into the SQLite `automatic_news_analyses` ledger
but does not price-validate or train.

The earlier source-price comparison used a separate 19-article manual
Gemma JSON output, yielding only 7 non-abstained scored directions among
41 impacts. That result is historically **RETROSPECTIVE_UNSEALED** and
cannot be reported as prediction accuracy.

This new PR closes a practical integration gap by taking the growing
**actual local automatic analysis ledger** as input to the existing
read-only market outcome calculation, using existing reviewed mappings
and source-provided adjusted daily stock/benchmark price CSV files.
It is an explicit one-shot research command; it never activates
persistent price provider polling, automatic learning, or trading.

## New CLI

`tris_market_event_intelligence.automatic_gemma_market_outcomes_cli`

Only the following inputs are accepted:

- Actual operator-owned `news.sqlite3` with collected original
  `articles`, `captures`, `article_full_text`,
  `story_members` and `automatic_news_analyses`.
- Immutable completed `analysis_json` rows for the *requested model*
  whose article ID and full-text digest match the **current original
  body**. No silent reuse of an earlier model/body variant.
- A source-reviewed target → instrument/benchmark mapping file.
- Actual source-provided adjusted daily price observations in the
  `symbol,date,adjusted_close,source` schema, and optional
  recovered-price SHA manifest.

The CLI runs the existing full provenance audit, loads the current
completed Gemma rows, validates evidence quotes and source integrity,
temporarily adapts them to the existing historical evaluator's schema,
then evaluates actual adjusted daily absolute and market-relative
returns over one and five later trading sessions. A temporary
directory is removed before the report is returned. No original
Gemma result, production SQLite table, original CSV or historical
price file is edited.

Per-impact result includes original news ID, target, model, predicted
UP/DOWN/NEUTRAL/UNCERTAIN, textual reason and actual source evidence,
stock/index return, benchmark return, excess return, absolute/relative
direction verdict, unavailable/missing mapping status, and simple
constant-direction baselines with identical denominators.

The report includes current completed-article coverage and explicitly
sets `forecastSealedBeforeOutcomes=false`,
`prospectivePredictionAccuracy=null`, `trainingEligible=false`,
`productionDatabaseModified=false`, and provider authentication
false. All model outputs are **retrospective descriptive research**:
the real wall-clock `processed_at` from the ledger is NOT an immutable
forecast seal demonstrably earlier than market outcome observation.

## Run against existing operator Mac evidence

After checking out the newest main containing this feature, use:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"

PYTHONPATH="$PWD/src" python -m \
  tris_market_event_intelligence.automatic_gemma_market_outcomes_cli \
  --db "$DATA/news.sqlite3" \
  --model gemma4:e2b \
  --limit 1000 \
  --mappings "$DATA/stage6-sector-target-map-v2.csv" \
  --prices "$DATA/stage6-sector-prices-v4-recovered.csv" \
  --price-manifest "$DATA/stage6-sector-prices-v4-recovered.csv.manifest.json" \
  --price-source "Yahoo Finance adjusted daily historical prices; recovered lineage" \
  --output "$DATA/automatic-gemma-real-daily-outcomes-pr269-v1.json"
```

The command does not download prices, ask the model to reason again,
enable `--watch`, call RSS providers or write to `news.sqlite3`.
It joins ONLY the targets whose mapping is present in the existing
reviewed file and reports the rest as `UNMAPPED`. Source-blocked
rows remain in the diagnostic source gate, never hidden in a claimed
accuracy. Result is a new create-only file.

**Do not** infer expected market-return coverage from the worker's
`completed:10`: a completed model pass need not contain any
tradable or economically directional impact. A previous `--watch`
worker may also have been running against source records added during
another collector execution. The one-shot report counts actual
stored completed rows at the time the read-only query executes.

## Foreground watch policy

`--watch` is required only if the operator intends to keep automatic
Gemma *text analysis* cycling in the foreground. `--interval 900`
means a 15-minute pause between passes; `--skip-collection` means
the worker **does not collect new RSS articles**. It can be stopped
with Ctrl-C; its committed database analyses are retained. It does
not become 24/7 autonomous market learning merely by staying on.
The optional analytical bridge can execute later from persisted rows
without the watcher.

## Still outside acceptance

A successful evaluation creates actual **historical price comparisons**
when source-reviewed mappings and sufficient daily bars exist; it
does not create authenticated first-public source clocks, new
adjusted minute bars, causal event impact estimates, independent
event families, or pre-outcome sealed forecasts. Real forward
prediction integrity and out-of-sample improvement over matched
baselines remain unmet.
