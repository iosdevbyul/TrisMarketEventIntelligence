# Learning-05 — private candidate comparison and no-promotion review gate

This tool compares **2–16 existing private NEWS importance model candidates**
that were created by the Learning-05 journaled research engine. It replays
each candidate from the exact immutable training source and its recorded
human-review time-gated news holdout. All candidate predictions use only
VALIDATION articles with human NEWS importance judgments. Reserved TEST
labels are never read or scored. There is no stock price, OHLCV, market
movement, volume, realized returns, or DonghakStockVision dependency.

## Descriptive result, never deployment permission

The research comparison reports multiclass log loss, Brier, macro-F1,
important precision/recall, an always-UNIMPORTANT baseline, and candidate
ranking by validation log loss. It checks immutable news cohort, holdout,
review and model artifact hashes, verifies replayed weights, and refuses
duplicate experiments, changed input files or altered model weights.

Fewer than **100 VALIDATION samples** or fewer than **20 independent
human-labeled examples per binary importance class** produces
`INSUFFICIENT_INDEPENDENT_NEWS_GOLD`. More samples may produce
`DESCRIPTIVE_SCREENING_ONLY`. **Neither status certifies accuracy or
authorizes promotion**: labels are still self-attested, duplicate/news
revisions require independent source auditing, and validation data may
have been repeatedly tuned against. Therefore
`promotionAllowed=false` and `modelImprovementValidated=false` always.

The tool intentionally provides no `--promote`, `--deploy`, production
pointer change, rollback action or automatic ranking integration with the
collector. It only writes a new private comparison report. This makes
unreviewed promotion impossible **through this command**, not through
every external process on the Mac.

## Existing Mac script (only once real human-reviewed data exists)

```sh
bash scripts/news-learning-review.sh --update compare-candidates \
  --candidate /private/research-candidate-1.json \
  --candidate /private/research-candidate-2.json \
  --cohort /private/news-cohort.json \
  --holdout /private/news-holdout.json \
  --review /private/human-reviewed-news.json \
  --journal "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-research.sqlite3" \
  --output /private/new-candidate-comparison.json
```

The examples are not a request to test now. Existing model artifacts may
contain source-derived private tokens. Never upload these files, reviewed
news or reports to public CI. The only runtime change is a create-only
research JSON report, with owner-only permissions.

## Remaining independent validation

A truly locked and never-seen news test set, human-adjudicated source
significance and evidence quality, cross-page/semantic duplicate audits,
minimum sample adequacy, model candidate selection independent of the final
TEST, and a separately auditable human decision trail are prerequisites
for any future shadow rollout. Promotion and rollback need a separate
design and tests; **no automatic deployment is approved here**.
