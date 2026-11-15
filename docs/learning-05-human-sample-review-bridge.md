# Learning-05 — manually reviewed 48-news sample bridge

This phase connects the **already created** private candidate sample,
importance and materiality templates to actual, human-written reviews.
The bridge does NOT invent labels and does NOT require the entire 482-row
corpus to be reviewed first.

## Human review procedure

In your Mac's private
`news-review-sample-20261115.json`, inspect source URL, title and
summary (and original article when appropriate). For a selected article
you personally assessed, enter:

- `importance`: IMPORTANT, UNIMPORTANT, UNCERTAIN; with
  `sourceAlignment`: SUPPORTED, CONTRADICTED, UNRELATED,
  INSUFFICIENT_CONTEXT. The two fields must be completed together.
- `potentialMagnitude`: LOW, MODERATE, HIGH, CRITICAL or UNASSESSABLE.
  It describes potential **organizational consequences**, not market
  price direction or realized returns.
- `sameStoryConfirmedByHuman`: true/false only when independently checked,
  or null when unknown. No label is ever propagated to related articles.
- `reviewer`, `reviewedAt` (timezone-aware ISO timestamp after initial
  article observation) and `reason` (20–2000 characters) for any entered
  human judgment. Unreviewed entries must retain all null fields.

Importance and potentialMagnitude may be assessed separately; do not
invent a magnitude if only importance is clear. Keep all unedited article
metadata and the document header intact.

The code checks the complete source SHA, regenerated original sample,
all immutable metadata, permitted label values, time, and reviewer
self-attestation. Both existing full-review templates must still be
entirely blank and match the pinned source. This *does not authenticate*
who actually wrote the human assessment; it prevents silent software
filling or tampering of an asserted human review.

## First run (no writes)

```sh
bash scripts/news-learning-review.sh --update review-bridge --audit \
  --cohort "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-cohort-20261115.json" \
  --sample "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-review-sample-20261115.json" \
  --importance-template "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-human-review-20261115.json" \
  --materiality-template "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-materiality-review-20261115.json"
```

If any *human* reviews have been entered and the audit succeeds, the
operator may explicitly run the same command with `--apply` in place of
`--audit`, and additionally give these **new, unique** output files:

```sh
  --importance-output "$HOME/Library/Application Support/TrisMarketEventIntelligence/reviewed-importance-20261115.json" \
  --materiality-output "$HOME/Library/Application Support/TrisMarketEventIntelligence/reviewed-materiality-20261115.json"
```

The apply mode refuses to create blank all-null review files and never
overwrites or updates the two original templates. Any export contains
**all 482 original rows** with human-judged entries ONLY for those source
articles actually edited in the sample; other entries stay null.
Only safe summary counts reach stdout. The resulting self-attested
review files are NOT independent verified gold, never justify model
promotion, and do not remove the need for a longer chronological
cohort, leakage audit, future held-out TEST assessment or a second
independent reviewer.

No stock prices, OHLCV, volume, returns, or DonghakStockVision inputs.
No model, collector, launchd or operating DB changes.
