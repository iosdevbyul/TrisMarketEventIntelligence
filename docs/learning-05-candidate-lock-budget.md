# Learning-05 — single TRAIN-only candidate lock and TEST budget audit

This phase uses existing research data rather than inventing another model
training engine. After **all 12** pinned policy hypotheses have finished,
`candidate-lock --lock` selects one candidate by minimum TRAIN-fold
cross-validation log loss (SHA-256 as deterministic tie-break). It verifies
each row against the allowed immutable hypothesis catalog and provenance
hashes. The resulting small, private, create-only owner-mode manifest pins
the candidate selection, configuration, news cohort, temporal holdout and
human review file. No model weights, gold labels, or source articles are
included in the manifest.

This is **not independent validation**. It is not a deployable model artifact:
the current policy candidate model would require separate replay/export
before any later shadow evaluation could use it. No validation or TEST
gold is scored while choosing the candidate. Existing model candidate
comparison and artifact pipelines are unchanged, and their independent
review limitations still apply.

Run only with real existing private data, not the placeholder paths:

```sh
bash scripts/news-learning-review.sh --update candidate-lock --lock \
  --cohort /private/news-cohort.json \
  --holdout /private/news-holdout.json \
  --review /private/independent-human-review.json \
  --journal "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-research.sqlite3" \
  --manifest "$HOME/Library/Application Support/TrisMarketEventIntelligence/locked-candidate.json"

bash scripts/news-learning-review.sh candidate-lock --audit \
  --manifest "$HOME/Library/Application Support/TrisMarketEventIntelligence/locked-candidate.json" \
  --usage-ledger /private/initialized-news-test-usage.sqlite3
```

The optional audit re-checks an existing TEST reservation ledger and reports
how many final TEST reservations were recorded for the pinned cohort. It
never reserves or evaluates TEST. Use the existing test-usage CLI to
initialize and reserve only when genuinely authorized for an independent,
once-per-cohort final assessment. Cooperative ledger enforcement cannot
detect use by an unrelated process. A zero count means no *recorded*
reservation, not proof that a hidden TEST was never accessed.

The optional `--cohort --holdout --review --journal` quartet on `--audit`
replays and compares pinned research selection and journal contents.
The lock is create-only, not cryptographically immutable against someone
who can edit the local filesystem; keep it owner-only and independently
back up its SHA-256 before evaluation.

**No promotion is authorized by this phase.** Human gold authenticity,
cross-story near-duplicate leakage, a locked independent evaluation protocol,
sample adequacy and actual Mac capacity must be verified separately.
There is no new price, OHLCV, volume, returns or DonghakStockVision input;
no launchd, production model change or automatic continuous background run.
