# Learning-05 — Local NEWS research readiness gate

Run this optional read-only check before requesting a real Mac guarded
TRAIN-only research cycle. It **does not start training** or require Ollama.
It reuses the existing provenance-aware cohort/holdout/human-review
validator, plus the preexisting cross-split lexical near-duplicate audit.
It reports the number of time-gated TRAIN labels per importance class,
whether the basic three-per-class research minimum is met, and potential
cross-partition lexical overlaps requiring human review. It never scores
VALIDATION or reserved TEST gold or claims performance improvement.

```sh
bash scripts/news-learning-review.sh --update learning-readiness \
  --cohort /private/news-cohort.json \
  --holdout /private/time-gated-holdout.json \
  --review /private/human-review.json
```

Optionally pass `--manifest` and `--journal` together (and an optional
`--usage-ledger`) to replay the preselected locked hypothesis and
inspect the existing cooperative TEST reservation ledger. The
manifest and journal must be provided together. This reports only
whether a final TEST reservation was recorded, not whether another
program has accessed the data.

A `PRELIMINARY_RESEARCH_READY_NOT_INDEPENDENTLY_AUDITED` result means
the **basic local preconditions** passed. It is not independent
human-gold certification, label accuracy evidence, or authorization to
open TEST. Lexical title matching cannot exclude semantic, translated,
cross-page, or correction-lineage duplicate leakage. A returned
`BLOCKED_REVIEW_OR_RESEARCH_PREREQUISITES` must be resolved by
human labeling or editorial review before a candidate evaluation.

No background process, new database, artifact, model deployment, stock
price/OHLCV/volume/return inputs, or production change is introduced.
CI tests contain synthetic human-label fixtures, not verified live data.
