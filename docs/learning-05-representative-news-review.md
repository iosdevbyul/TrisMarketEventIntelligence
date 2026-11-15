# Learning-05 — representative NEWS human-review pilot

The Mac archive currently has **482 news articles observed during three UTC
calendar days (446/30/6)**; this concentration is not enough to establish
independent chronological model generalization. The cohort and two full,
unlabelled review templates already exist privately. Do not recreate them.

## Goal and limitations

`news-review-sample` extracts a **local human-review candidate packet**
from the existing immutable cohort, without requiring a holdout yet.
Grouping deliberately distinguishes:

- **Candidate links** from the archived storyGroupId, exactly identical
  title+summary SHA, or normalized URL. These are *not* verified
  independently meaningful events.
- **Cross-unit lexical near-repeat warnings** based on title token Jaccard
  of at least 0.78, at least five shared title tokens. These are hints
  for human review, **never automatic merges**.
- **One representative per candidate unit**, selected reproducibly by a
  SHA-derived order, with round-robin balance over UTC observed day and
  original news source. The default is at most 48 candidates and at most
  100 allowed. No model scoring, no human label or performance metric
  affects selection.

The private JSON includes original article title, summary, source URL,
original article SHA and grouped article IDs so a real reviewer can
inspect them. Terminal output includes **counts only**; do not upload
the private JSON to GitHub, Trello or a public chat.

Every row has blank `importance`, `potentialMagnitude`,
`sourceAlignment`, `sameStoryConfirmedByHuman`, reviewer, timestamp,
and reason. This packet is **not** a gold-label file and does not
auto-populate the previously generated 482-row human-review templates.
The candidate group count must not be described as a proven number
of unique stories; semantic/translated repeats can remain.

## Actual Mac command

On clean main in the local repository:

```sh
bash scripts/news-learning-review.sh --update review-sample \
  --cohort "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-cohort-20261115.json" \
  --max-selected 48 \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-review-sample-20261115.json"
```

The command reads a source-verified local JSON cohort and creates an
owner-only, new, private candidate packet. It does not change SQLite,
operate on market prices/returns/volume, call an LLM, install software,
or examine reserved TEST labels.

The next step is human inspection and adjudication of the candidate
units and related article clusters, **not automated model training**.
Because all 482 are observed over only three days, retain future
news collection and do not force a TRAIN/VALIDATION/TEST split to pass
just for the sake of reporting accuracy.
