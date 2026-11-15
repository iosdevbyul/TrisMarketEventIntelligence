# Learning-05: blinded human review of chronological NEWS impact changes

PR #142 provides **model-estimated** changes in news potential materiality
across articles that mention the exact same entity. Matching entity names is
NOT proof that the news articles describe the same underlying event.
A magnified model grade is NOT evidence that the real-world impact escalated.

This phase introduces two explicitly manual, offline steps:

1. **Prepare** a source-pinned, unfilled human-review packet for article
   pairs whose earlier and later articles are **both in VALIDATION**.
   The packet has article IDs only, **no model grades/deltas**. A reviewer
   should independently read the referenced original news and determine
   whether it is the same evolving event. If yes, they should judge whether
   the possible organizational consequence materially INCREASED, DECREASED,
   or stayed UNCHANGED. Mark ambiguous cases UNASSESSABLE.
2. **Evaluate** the completed real-human packet against the model's earlier
   and later magnitude grades, reporting descriptive agreement and
   directional ordinal error **only on same-event human-assessed pairs**.
   Different events, uncertain relatedness, unassessable model output, and
   incomplete human reviews are counted but **never silently scored as
   correct or incorrect**. No TEST labels are opened or scored.

Important limitations:

- Pair candidates come from the model's exact entity names and thus are
  **model-selected** rather than independently sampled. Never interpret a
  good result as generalization, model improvement, or causal verification.
- Source identity is pinned by cohort SHA; holdout and model report SHA are
  also pinned. Each model hypothesis is re-checked against archived article
  text hashes and verbatim source excerpts before packet creation.
- The evaluation does **not** add factual truth to model-generated
  `fact_status`; independent authenticity and semantic evidence checks
  remain separate. Human reviewer identity is self-attested.
- The test partition is never part of the pair packet, even if the model
  produced a result on those articles. A valid TEST evaluation needs an
  entirely separate locked protocol.
- No stock price, OHLCV, volume, actual returns, market-derived gold labels,
  automatic training or model installation.

Once private source files and real human reviews exist, use the existing
script (these are example locations, not verified paths):

```sh
bash scripts/news-learning-review.sh --update materiality-transition --prepare \
  --cohort /private/news-cohort.json \
  --holdout /private/holdout.json \
  --report /private/materiality-local-report.json \
  --output /private/blank-transition-review.json

# After a person has filled the review packet:
bash scripts/news-learning-review.sh materiality-transition --evaluate \
  --cohort /private/news-cohort.json \
  --holdout /private/holdout.json \
  --report /private/materiality-local-report.json \
  --review /private/human-completed-transition-review.json \
  --output /private/descriptive-transition-audit.json
```

If no pair is eligible, the review packet is empty and evaluation returns
INSUFFICIENT_HUMAN_TRANSITION_GOLD. An empty packet is not positive evidence.

Still required: real Mac news data, blind independent human reviewers,
a non-model-selected event-pair benchmark, true source verification,
meaningful cross-source event matching and repeated prospective evaluation.
Learning-05 remains In Progress; Learning-06/07 manage further semantic
model improvements and deployment lifecycle.
