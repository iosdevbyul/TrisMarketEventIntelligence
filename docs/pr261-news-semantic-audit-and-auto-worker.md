# PR #261 news semantics and automatic worker follow-up

This work intentionally contains **no market price fetching, ETF proxies,
return validation, or trading**. Keep the saved original article output
directories immutable. Existing automatically completed analyses are not
retroactively overwritten when evidence segmentation changes.

## Why the previous comparison failed

Two distinct Gemma4 source-only batches were run on the same 20 article IDs.
Run 1 finished 20/20; run 2 finished 19/20, with article #4 returning
`MODEL_CONTENT_EMPTY`. Impact counts varied between runs. The second batch
saw 773 SHA-256 verified COMPLETE original texts versus 770 at the start of
the first run; the **selected 20 article IDs still matched**, so the
cross-run audit must compare body SHA256 for each article rather than assume
that all source inputs are unchanged.

The error `No module named tris_market_event_intelligence.news_collection_semantic_audit_cli`
occurred when the local checkout did not include the separate audit PR #259.
This PR now includes the audit command and its tests.

## Local comparison using already saved evidence (no model calls)

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
python -m tris_market_event_intelligence.news_collection_semantic_audit_cli \\
  --db "$DATA/news.sqlite3" \\
  --run "$DATA/news-semantic-repeat-run-1" \\
  --run "$DATA/news-semantic-repeat-run-2" \\
  --output "$DATA/news-semantic-repeatability-pr261-v1.json"
```

Read `repeatability.comparableArticles`, `stableArticles`, `agreement`
and each `cases[].status`. Review `cases[].analysisByRun` to compare
the model's precise target, direction, reason, original cited passage and
limitations. Agreement is **only repeatability**. It is not proof that the
inference is economic fact, causal or a reliable investment forecast.
Article #4 should be identified as `MISSING_RUN` unless retry evidence has
subsequently been saved in the second output directory.

## Production auto-analysis and dashboard

The PR #261 worker stores completed analysis in SQLite once per
(article ID, verified body hash, model). It intentionally does **not**
re-run successful articles just to measure stochastic variation; the
two-output-directory process above is the isolated repeatability pilot.
When the daily ingestion backlog is large, the worker now allocates
capacity to recent articles, older backlog and cooled-down failures.

With an already running collector, use `--skip-collection` to avoid
racing two collection jobs. Do not start another automatic worker
alongside an existing one; use the currently running process.

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
python -m tris_market_event_intelligence.automatic_news_pipeline \\
  --db "$DATA/news.sqlite3" --model gemma4:e2b \\
  --skip-collection --limit 10
```

The previous command is a **single, manual pass** only, and can make model
calls and write automatic analysis results. Execute it only when no
other auto worker is running. The dashboard's aggregate metric shows
completed analyses, failed analyses and per-error-code counts from
SQLite; collection degraded status is no longer reported as healthy.
Neither a source error nor an empty Gemma reply counts as a completed
analysis. The failure retry remains limited and cooldown guarded.

Note that PR #261 is a code change, not automatic deployment to your Mac:
fetch/pull its feature branch, perform a real Ollama smoke test, confirm
the running worker uses the updated code, and check CI before merge.

## Repeatability finding and next controlled model experiment (2026-11-23)

**Local empirical result:** 20 article IDs observed, 19 comparable same-model
and same-source verified article reviews, **1 exact target+direction agreement
(5.26%)**. One missing result is article #4 after `MODEL_CONTENT_EMPTY`.
The saved runs are baseline evidence; **do not overwrite them** or infer that
18 of 19 are factually wrong. Distinguish different target labels from
different directional predictions of the *same* target.

The audit now reports `sameImpactCountArticles`, `sameTargetSetArticles`,
`sameSharedTargetDirectionArticles` as well as exact agreement and per-case
`targetSetsIdentical`, `sharedTargets`,
`sharedTargetDirectionsIdentical`, reasons and verbatim source passages.
Note that shared-target direction agreement is vacuously true for articles
with no common target labels, so never use it as a headline metric by itself.
Exact label matching is conservative; semantic synonyms can remain divergent
even when discussing the same industry.

The common Gemma4 adapter now includes stronger *source-based event scope*,
conservative sector expansion, explicit directional mechanism and abstention
instructions. Sampling requests `temperature=0` and an explicit seed; this
improves configuration reproducibility but does not promise deterministic
model results across execution environments. Increasing `num_predict` to 1536
provides room for structured responses without changing validation gates.
These changes affect both source-only batch mode and automatic SQL analysis.
Do not silently overwrite preexisting successful automatic analyses.

Run the baseline audit again with a **new immutable output filename**,
without invoking the model:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
python -m tris_market_event_intelligence.news_collection_semantic_audit_cli \\
  --db "$DATA/news.sqlite3" \\
  --run "$DATA/news-semantic-repeat-run-1" \\
  --run "$DATA/news-semantic-repeat-run-2" \\
  --output "$DATA/news-semantic-repeatability-diagnostic-v2.json"
```

**Controlled experiment after baseline review:** repeat two *fresh* Gemma
runs in directories `news-semantic-stability-v2-run-1` and
`news-semantic-stability-v2-run-2` with `--limit 20`, then audit those
two new directories into a new output file. Compare exact agreement and
target overlap against the old baseline, while noting any changes in original
article hashes, model builds and error counts. Do not interpret better
agreement alone as proof of economic meaning; sample the newly generated
reasons for relevance, direction mechanism, and potential hallucination.
