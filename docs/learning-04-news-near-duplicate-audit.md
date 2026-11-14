# Learning-04: Cross-split news near-duplicate audit

This opt-in news-only audit augments the existing exact-hash, canonical URL,
and story-group holdout. It looks for **potential paraphrased titles** across
TRAIN, VALIDATION, TEST, EMBARGO, or QUARANTINED partitions.

It computes token-set Jaccard similarity on normalized Korean/English headline
tokens (minimum five shared meaningful tokens, sufficiently long headlines,
and a configurable 1–30-day observation window). It does not use LLM,
stock prices, returns, trading volume, or any price-based labels.

**Important:** The output is *not* a confirmed duplicate relation. A
high-overlap headline may refer to a different story or later correction,
and low-overlap or translated stories may escape detection. Every candidate
is `HUMAN_REVIEW_REQUIRED` with `confirmedSameStory=null`. Nothing is
silently merged, dropped, relabeled, or published, and the existing held-out
TEST is never read as a human importance label.

The output checks the cohort SHA against the plan, every article ID and
archived text SHA, first observed receipt, split memberships and counts.
The report is new, create-only, and private. No input or production SQLite
database is modified. The work is **not** a guarantee of no leakage;
`trainingDatasetEligible=false` remains unchanged regardless of findings.

Use the existing script **only when a real local news cohort is ready**:

```sh
bash scripts/news-learning-review.sh --update near-duplicates \
  --cohort /private/historical-news-cohort.json \
  --holdout /private/news-holdout-plan.json \
  --output /private/new-news-near-duplicates.json
```

Optional `--threshold 0.78 --max-days 30` tune review workload, not
duplicate truth. A report with no lexical candidates is
`NO_LEXICAL_CANDIDATES_NOT_VERIFIED`, *not* confirmation of an uncontaminated
test set. Cross-page near-duplicates, updated/corrected stories and
multilingual paraphrases require further human review and full-history
sampling. No automated training or model promotion is enabled.
