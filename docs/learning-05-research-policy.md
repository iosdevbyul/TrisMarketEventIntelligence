# Learning-05 — bounded research-policy expansion

The existing CPU-only news importance research now includes two additional
**representation hypotheses** beyond the original 18 TRAIN-only configurations:
binary word presence versus bounded character-trigram presence. Both are fitted
as separate multinomial Naive Bayes models on TRAIN folds only; this is
**feature-hypothesis exploration, not a new independent classifier family**.

For each representation, evaluate first 32 or 128 news tokens with alpha
0.25 / 1 / 4. Exactly 12 configurations per pinned input are permitted.
Immutable SHA-256 configuration fingerprints and a separate SQLite table
provide deduplication after restart. If all configurations have already
run, the process returns `POLICY_HYPOTHESES_EXHAUSTED` and does not train.
Insufficient human TRAIN labels stop research instead of creating fake results.

The audit ranks TRAIN-only cross-validation log loss for inspection. Ranking
is **not** evidence of generalization and cannot authorize operational
promotion. The private human-reviewed TRAIN dataset and temporal holdout
provenance are mandatory. Reserved VALIDATION/TEST labels are never scored
to choose these hypotheses. Human label independence and actual improvement
have not been demonstrated by synthetic CI.

## Existing guarded Mac execution, manual only

```sh
bash scripts/news-learning-review.sh --update guard-research --run \
  --research-mode policy --max-iterations 12 \
  --cohort /private/news-cohort.json \
  --holdout /private/news-holdout.json \
  --review /private/human-review.json \
  --journal "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-research.sqlite3" \
  --status-file "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-guard-status.json" \
  --lock-file "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-guard.lock"

bash scripts/news-learning-review.sh research-policy --audit \
  --journal "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-research.sqlite3"
```

Do not run these placeholders verbatim; use real private files when ready.
Resource safeguards are the existing OS CPU process limit, wall-time limit,
RSS sampling, exclusive lock and parent-cooperative stop flag. No launchd
installation, model promotion, production service change, Gemma installation
or price/OHLCV/volume/return/DonghakStockVision data access.

## Remaining acceptance gates

1. Verify a real Mac run with independently reviewed human NEWS importance
   labels; synthetic fixtures do not prove learning effectiveness.
2. Independently audit reviews and prevent cross-story/time leakage.
3. Lock a candidate after TRAIN-only exploration, then evaluate once under
   a documented validation/test-use budget before considering shadow
   deployment. Independent human evaluation must show substantial benefit.
4. Keep research finite where information is exhausted; an always-on service
   must not consume compute merely to repeat identical experiments.

Learning-05 stays In Progress until these gates are met.
