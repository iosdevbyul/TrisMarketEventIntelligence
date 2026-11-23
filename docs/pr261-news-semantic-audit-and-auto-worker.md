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

## v2 repeatability outcome and root-cause isolation (2026-11-23)

Latest on-device Gemma4 experiment: **20/20 successful** in both freshly
generated source-only runs, with **1/20 exact target+direction set agreement
(5.0%)**. The earlier experiment had 1/19 (5.26%) and an empty model response.
Thus JSON response reliability improved in this small sample, but **content
stability has not improved**. Neither metric measures semantic correctness.

Before attempting another prompt rewording, use the saved v2 run folders with
the expanded diagnostic audit. It now separates **same impact count**,
**same exact target-name set**, **target-set Jaccard overlap**, **articles with
no shared targets**, and **genuine direction disagreements among shared
target labels**. It records the actual directions and model-written reasons.
No semantic synonym mapping or automatic direction fixing is performed; those
would conceal the original instability. When there are no common labels,
shared-target direction agreement is *not* evidence of agreement.

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
python -m tris_market_event_intelligence.news_collection_semantic_audit_cli \\
  --db "$DATA/news.sqlite3" \\
  --run "$DATA/news-semantic-stability-v2-run-1" \\
  --run "$DATA/news-semantic-stability-v2-run-2" \\
  --output "$DATA/news-semantic-stability-v2-diagnosis.json"
```

Interpretation: If target Jaccard is low with little shared-target direction
disagreement, prioritize target identity and extraction policy; if target sets
mostly match but shared directions frequently conflict, prioritize reasoning
and directional evidence validity. Neither path can be chosen safely from
the 5.0% headline alone. Never erase the original run JSON, and do not score
agreement as news impact truth.

## v2 root-cause breakdown and optional automatic consistency gate

The local source-only v2 audit reported **20 comparable articles** and:
- Exactly identical target/direction sets: **1 / 20 = 5%**
- Identical impact counts: **11 / 20**
- Identical exact target-label sets: **1 / 20**
- No common target labels: **5 / 20**
- Articles with direction disagreement on a shared label: **5 / 20**
- Shared target-label pairs: **21**, with **6** conflicting directions
- Mean target-set Jaccard similarity: **0.3375**

The evidence shows **both target selection and shared-target direction
instability**, not a demonstrated source collection failure or 95% factual
error. The current exact target matching may overstate instability if synonyms
or issuer names are expressed differently; do not silently treat synonyms as
equal without independent source review.

The automatic pipeline now supports an **opt-in two-call agreement gate**.
This is an additional local model workload, **not** a retraining or
accuracy improvement. Use it only if no other automatic worker is running:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
python -m tris_market_event_intelligence.automatic_news_pipeline \\
  --db "$DATA/news.sqlite3" --model gemma4:e2b \\
  --skip-collection --limit 3 --verification-runs 2
```

By default `--verification-runs 1` preserves the existing worker's throughput
and explicitly marks completed single-run results `SINGLE_RUN_UNVERIFIED`.
With `--verification-runs 2`, two **independent** source-backed Gemma4
inferences for the same verified body hash are compared before persistence:

- `REPEATABLE_UNREVIEWED`: complete target and direction sets matched, but
  the underlying financial/causal reasoning is *still unverified*.
- `REVIEW_REQUIRED`: the two runs chose different target labels or
  conflicting directions; the initial analysis, second analysis, original
  reasons and evidence quotes are all retained for human inspection.
- A failed model request remains `FAILED`, subject to the existing bounded
  cooldown/retry policy. Two calls count as one article attempt.

The dashboard distinguishes `REVIEW_REQUIRED` from execution failures
and visibly labels completed single-run and repeatable analyses as unverified.
Completed historical results and the saved v1/v2 source-only batches are never
overwritten. A new two-call check applies only to newly analyzed verified
article body hashes; **no claim of semantic correctness** follows from
agreement, and **no market prices or returns** are collected.

To inspect already saved review evidence without any model calls:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
python -m tris_market_event_intelligence.news_collection_semantic_audit_cli \\
  --db "$DATA/news.sqlite3" \\
  --run "$DATA/news-semantic-stability-v2-run-1" \\
  --run "$DATA/news-semantic-stability-v2-run-2" \\
  --output "$DATA/news-semantic-stability-v2-details-final.json"
```

Inspect `repeatability.cases[].analysisByRun` for target, direction,
reason and exact original supporting quotation. Baseline and single-call
quality remain unchanged merely by adding a review gate.
