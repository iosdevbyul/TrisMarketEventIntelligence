# Learning-05 — Preregistered model-independent NEWS pairs

## Why this exists

The previous [blind transition review](learning-05-blind-transition-review.md)
takes pair candidates from a model's recognized entity names. That is a
selection bias: a model can inadvertently be evaluated only where it identified
a common entity. This research tool **chooses and pins pairs before loading
a model result**. It cannot certify that human labels are correct or that a
trained model has improved.

## Sampling and protected data

- Read the immutable NEWS cohort and validated chronological holdout only.
- Select exclusively within VALIDATION; TRAIN/TEST/EMBARGO and quarantined
  articles are excluded.
- Select at most 100 pairs in **linear time**, deterministically by an
  explicitly versioned policy and SHA-derived article ordering.
- Up to half the pairs come from preexisting, unverified `storyGroupId`
  memberships; the remainder compare different/unknown groups. An archived
  group is a candidate relationship, **not proof of the same event**.
- Pre-registration reads no local model outputs, prior model grades, human
  judgments or future news. No selection by materiality or stock return.
- The generated packet contains just pair IDs, the sampling stratum and
  blank human fields; **no model magnitude or model delta**.
- A reviewer checks each original article and records SAME_EVENT,
  DIFFERENT_EVENT or UNCERTAIN. Only SAME_EVENT pairs can have a
  materiality change INCREASE, DECREASE, UNCHANGED or UNASSESSABLE.
- Evaluating requires the exact same cohort and holdout, the completed
  human review and a previously produced PR #142 local research report;
  the report is replayed using the PR #143 provenance validator.
- The output measures descriptive agreement and ordinal direction error
  solely on same-event reviewed pairs in VALIDATION. Any fewer than 20
  scorable pairs are reported as insufficient.

**Important limitations:** Uniform-like deterministic pairing across a
small validation sample is not evidence of independent gold: human reviewers
are self-attested, event identity may be uncertain, story membership could
reflect prior automated processing, and article pairs are not statistically
independent when the same article occurs in multiple pairs. Future truly
held-out, independent human assessment remains necessary. No automatic
model selection, TEST readout, success claim or production promotion.

## Existing Mac script

The paths below are illustrative; do not run until a real local cohort
and validated holdout exist.

```sh
# First: freeze the article pairs BEFORE examining the model predictions.
bash scripts/news-learning-review.sh --update materiality-pairs --prepare \
  --cohort /private/news-cohort.json \
  --holdout /private/news-holdout.json \
  --output /private/blind-independent-pairs.json

# After a human reviewer fills that original packet, and an explicitly
# requested local model run has produced a complete research report:
bash scripts/news-learning-review.sh materiality-pairs --evaluate \
  --cohort /private/news-cohort.json \
  --holdout /private/news-holdout.json \
  --review /private/completed-independent-pairs.json \
  --report /private/local-materiality-report.json \
  --output /private/independent-pair-audit.json
```

The second step is not an independent TEST: it is descriptive VALIDATION.
No stock prices, OHLCV, volume, returns, or downstream forecasts are used.
No model downloads, unattended Mac processes or launchd changes.
