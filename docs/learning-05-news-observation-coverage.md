# Learning-05: automatic NEWS observation coverage and growth audit

This is the next step after the real Mac imported 482 NEWS articles,
selected 48 human-review candidates and verified the offline reviewer
runs. Human evaluations are **optional and remain paused** at user request.
Do not keep adding forms, and do not synthesize human gold with an LLM.

## Why this exists

The actual 2026-11-15 archived cohort has **482 articles** across only
**three observed UTC calendar days** (Nov 13: 446, Nov 14: 30,
Nov 15: 6). This is an inadequate basis to claim independent
time-ordered model generalization. Raw article counts are not event
counts or model-accuracy evidence. Earlier readiness checks focused
mostly on human-label availability and missed these limitations.

The `coverage` command reports descriptive original NEWS metadata:

- first/last observation, UTC daily counts, source counts, total active
  observed days and calendar span;
- biggest single-day/source concentration;
- original story-candidate count via the existing declared-story,
  exact-text and canonical-URL rules;
- preliminary 7 active day / 14 calendar day / 75% one-day-concentration
  thresholds as **warning heuristics only**. Passing them is **never**
  independent verification of source coverage, semantics, human labels,
  non-leakage or predictive performance.

There are no new scrapers, ML models, price labels, percentage returns,
volume, OHLCV, TEST scoring or automatic production changes.

## Run on original private Mac cohort

```sh
bash scripts/news-learning-review.sh --update coverage \
  --cohort "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-cohort-20261115.json"
```

The command **only reads** a complete `historical_news_cohort_v2`
snapshot with validated article text SHA, source capture SHA and
observed article dates. Output is safe metadata/counts, not full
news text, URLs or personal reviewer records. It runs without a human
review JSON or local Ollama and does not write any output file.

Once the existing RSS collector has accumulated more data, export
another complete read-only cohort using a *new* filename and a later
`--as-of` cutoff. Compare the immutable history:

```sh
bash scripts/news-learning-review.sh coverage \
  --cohort "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-cohort-20261115.json" \
  --newer "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-cohort-LATER.json"
```

The comparison fails closed on missing old IDs, modified original
titles/summary/source/URL/capture hashes or observation timestamps,
same/earlier cutoff, or new rows that claim to have already been
observed before the older cutoff. Mutable extraction metadata and
story indexing are intentionally not used as immutable identifiers.

If the archive grows but source diversity or calendar spread remains
poor, keep collecting rather than fabricating a success. Neither the
number of candidate story groups nor the number of days certifies that
independent events were observed. No persistent background process
is installed.
