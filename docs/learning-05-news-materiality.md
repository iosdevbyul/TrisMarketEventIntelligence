# Learning-05 — NEWS-only potential materiality: direction-independent research

## Decision and boundary

A **reported event can be extremely material even when the subsequent
market direction cannot be inferred**. Organizational/sector exposure and
conditional causal pathways—not realized price movement—determine the
provisional materiality judgment. The news service must not read or label
with stock price, candlesticks/OHLCV, volume, realized returns, or
DonghakStockVision's market forecasts.

Illustrative hypothetical progression: a chief executive's personal-life
allegation may be LOW or MODERATE; a second allegation involving
information security may be HIGH if substantiated; a confirmed sudden
departure could be CRITICAL. **These are illustrations of possible
mechanisms, not established facts or calibrated model scores**. A
rumor/ALLEGED claim may have high hypothetical consequences if true,
while remaining disputed and unverified. A high magnitude is neither a
negative price prediction nor a 95% probability.

## New additive contract

`NewsMaterialityHypothesis` provides a research-only
`news_materiality_hypothesis_v1` output:

- NEWS article ID/text SHA, entity, event summary, exact source excerpt,
  conditional mechanism, needed verification;
- fact reporting status ALLEGED / REPORTED / CONFIRMED / DISPUTED;
- potential magnitude LOW / MODERATE / HIGH / CRITICAL / UNASSESSED;
- **independent** direction POSITIVE / NEGATIVE / MIXED / UNCERTAIN;
- mandatory explicitly *uncalibrated* magnitude basis;
- hard forbidden price/prediction/human-gold flags.

The earlier `NewsImpactSignal` v1 is unchanged; this new contract does
not alter the production collector/API, automatically compute severity, or
magically grant a model knowledge of corporate context.

## Human-only magnitude rubric and descriptive metrics

The blank `news_materiality_human_review_v1` document binds every original
article by ID + text SHA and requires a real reviewer, timestamp, and
substantive reason for nonblank labels.

Interpretation guide for human assessors:

- LOW: limited plausible relevance to organizational operations, governance,
  compliance, continuity, or wider stakeholders;
- MODERATE: meaningful but contained plausible secondary consequences;
- HIGH: material plausible legal, operational, security, reputational,
  governance, or strategic consequences if the report holds;
- CRITICAL: potentially existential or systemically consequential disruption
  (e.g. confirmed abrupt critical leadership transition with unresolved
  succession), **not an assertion of financial performance**;
- UNASSESSABLE: insufficient source detail to assign a magnitude.

Magnitude concerns **potential consequences**. Verification of facts,
conditional mechanisms, sentiment/direction, and importance are separate.
Actual media coverage alone is not evidence of materiality.

`news_materiality_predictions_v1` wraps hypotheses with explicit
`modelId` and `promptVersion`, pinned cohort SHA, and exactly one
prediction per source article. Hypothesis evidence excerpt must appear
**verbatim in the archived title or summary**. This detects some
fabrication but does *not* establish factual or semantic correctness.

`evaluate_materiality_validation` scores only original **VALIDATION**
split: exact band agreement and mean ordinal band error; TEST labels
are never scored. Fewer than 100 scored items produce
`INSUFFICIENT_REVIEWED_VALIDATION`. Any result is still
`DESCRIPTIVE_ONLY_NOT_INDEPENDENT_VALIDATION` unless truly independent
research design, human audit, and external evaluation are completed.
The human review file is self-attested; schema checks do not prove
reviewer independence or quality. No model comparison winner,
deployment or automatic promotion results.

## Manual Mac usage (once actual research files exist)

```sh
bash scripts/news-learning-review.sh --update materiality --prepare \
  --cohort /private/news-cohort.json \
  --output /private/news-materiality-human-template.json

# After a HUMAN reviewer completes the new template and a separate
# research model supplies news-only hypotheses:
bash scripts/news-learning-review.sh materiality --evaluate \
  --cohort /private/news-cohort.json \
  --holdout /private/time-gated-holdout.json \
  --review /private/news-materiality-human-reviewed.json \
  --predictions /private/news-materiality-model-predictions.json \
  --output /private/news-materiality-descriptive-evaluation.json
```

These are example file paths, not actual verified local files. Output
is create-only with owner-only permissions. No Mac execution requested
until local private files and a real human review exist.

## Remaining research tasks

1. Develop and independently compare a genuine news-context inference
   candidate that produces the v1 hypothesis contract. No synthetic
   expected probabilities, no fabricated 0–100 scales.
2. Set independent human rubric adjudication and acceptable
   agreement/error thresholds before examining held-out evidence;
   audit cross-story and time leakage, including semantic repeats.
3. Evaluate paired changes across related news stories and assess
   uncertainty/calibration without introducing market-derived labels.
4. Independently verify improvement and guarded Mac execution, keep
   Learning-05 In Progress until acceptance. Use Learning-06/07 for
   advanced semantic model lifecycle and approved shadow operation.

## References

- [Learning-05 Trello](https://trello.com/c/IZ1kjCnu)
- [Project GitHub](https://github.com/iosdevbyul/TrisMarketEventIntelligence)
- [Current readiness gate](learning-05-news-readiness.md)
