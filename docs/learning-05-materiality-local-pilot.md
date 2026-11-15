# Learning-05 — opt-in local NEWS materiality hypothesis pilot

This is the **first real, opt-in local inference path** for the additive
`news_materiality_hypothesis_v1` contract from PR #141. It uses the existing
locally installed `gemma2:latest` model by default (not Gemma 4), **without**
downloading, installing or replacing any model. The production analysis
provider, SQLite, launchd configuration and market-price forecasting remain
untouched. No auto-run.

## Research semantics

For each article, the local model receives **only that article's original
title, summary, observed time, article ID and immutable text SHA**.
It does not receive future articles, other labels, price/return/volume data,
or any answer from human gold. Source text is untrusted data, not instructions.
The candidate describes:

- the news event and entity;
- a source-verbatim evidence excerpt;
- its reporting/verification status and a conditional mechanism;
- LOW, MODERATE, HIGH, CRITICAL or UNASSESSED **potential consequences**;
- an **independent** POSITIVE, NEGATIVE, MIXED or UNCERTAIN *direction*.

Unverified source claims are not verified merely because a model produces
a plausible mechanism. CRITICAL and UNCERTAIN are compatible: high possible
organizational consequences say nothing about actual stock movement.

The successful research report embeds a complete, provenance-pinned
`news_materiality_predictions_v1` payload and chronological changes between
hypotheses mentioning the exact same model-output entity. These comparisons
are **not verified updates of one underlying story**, causal escalation, an
established fact, or a real performance improvement. Local model errors,
source-excerpt mismatches, invented entities and malformed results return
`RESEARCH_INFERENCE_FAILED` with **no partial prediction artifact**. This
pilot is limited to **20 articles** to preserve Intel Mac resources.

## Commands — only when the local operator opts in

```sh
# Preview never invokes Ollama.
bash scripts/news-learning-review.sh --update materiality-local --preview \
  --cohort /private/news-cohort.json --limit 10

# Run only with an already installed local gemma2:latest and a pinned
# cohort of <= 20 articles. --limit must include every cohort article.
bash scripts/news-learning-review.sh materiality-local --run \
  --cohort /private/news-cohort.json --model gemma2:latest --limit 12 \
  --output /private/new-materiality-local-report.json
```

The private path and count are placeholders, not verified files/size.
A successful complete report contains the `result` predictions payload
needed by the separate human-review `materiality --evaluate` command;
extract it privately into a new file with its original `cohortSha256`
and exact model metadata before using the validation evaluator.
Do not alter the source/holdout/review data, send private articles to a
remote endpoint, or store these reports in public GitHub.

**Not achieved by this PR:** actual independent human review, model
quality improvement, semantic evidence correctness, cross-story leakage
clearance, TEST evaluation, automatic model promotion, continuous workers,
or a new machine-learning algorithm. Development proceeds next with
human-adjudicated pilots and model-to-model comparisons, not an automatic
claims-of-accuracy loop.
