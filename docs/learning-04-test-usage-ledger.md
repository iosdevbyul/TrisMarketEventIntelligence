# Learning-04 — News-only final TEST use ledger

This is an **opt-in local safety gate** for research processes that voluntarily use the registry. It records model candidate SHA, experiment identity, immutable news holdout plan SHA, cohort SHA, and whether an experiment uses TRAIN, VALIDATION, or the *one-time* TEST_FINAL partition. It neither trains a model nor reads human-gold values or market prices.

- TRAIN and VALIDATION can be reserved for distinct experiments repeatedly.
- **TEST_FINAL is reserved at most once per historical news cohort SHA**, even if someone produces a new split file from that cohort. No second final TEST reservation is permitted.
- A repeated experiment ID fails; SQLite's immediate transaction and unique index protect simultaneous cooperating writers.
- Any use of a fixed test set by another program **outside** this ledger is not detectable. Reservation itself is *not* an evaluation or authorization to read gold. Actual access controls, append-only backup verification, independent human labels and candidate-blind final assessment remain future requirements.
- Do not count this as evidence of model improvement, independent held-out accuracy or real news coverage.

## Mac script (only when real review is required)

Run the existing project script, which resolves the local Conda/venv environment automatically. The following examples are **illustrative**, not commands that need to run now:

```sh
bash scripts/news-learning-review.sh --update test-usage \
  --ledger "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-test-usage.sqlite3" --init

bash scripts/news-learning-review.sh test-usage \
  --ledger "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-test-usage.sqlite3" \
  --reserve VALIDATION \
  --plan /private/news-holdout-plan.json \
  --experiment-id "news-experiment-001" \
  --candidate-sha256 "<64 lowercase hex characters>" \
  --reason "Evaluate a news classifier candidate against validation only"

bash scripts/news-learning-review.sh test-usage \
  --ledger "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-test-usage.sqlite3" --audit
```

**TEST_FINAL must only be reserved for a genuinely frozen candidate after independent verification**. Do not run TEST_FINAL in exploratory loops. This gate fails closed on bad/missing plans, empty requested partitions, bad SHA IDs, duplicate experiments and repeat TEST reservations.

The ledger is deliberately isolated from production `news.sqlite3`, and all existing collected-news, human-review, and source files stay read-only. It stores metadata, not article text, reviewer notes or gold labels. No scheduler, background worker or price data pathway is enabled.

CI uses synthetic news fixtures; local Mac execution is not needed until an authentic validated corpus and separate experiment are available.
