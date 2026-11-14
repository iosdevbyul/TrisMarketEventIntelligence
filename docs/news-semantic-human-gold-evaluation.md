# Human-gold semantic review against archived news hypotheses

This opt-in, read-only research tool follows PR #111–#113. Its job is **not**
automatic fact verification. It measures how often the simple lexical
`semanticContextReview.reviewCodes` warnings correspond to independently
**human-assessed** issues in a particular archived news hypothesis.

## Preparation

First inspect the existing immutable journal evidence record:

```sh
market-event-pilot-review-packet --evidence-id 1
```

The output contains `provenance.signalSha256`,
`provenance.sourceEvidenceSha256`, the original reported source context,
and the original archived conditional mechanism. A person must actually
review these; do not use an AI-generated judgment as a human gold label.

Create a private JSON file (the values in this example are placeholders
and MUST be replaced with actual journal hashes and human observations):

```json
{
  "schemaVersion": "news_semantic_gold_v1",
  "judgments": [
    {
      "journalEvidenceId": 1,
      "signalSha256": "<64 lowercase hex characters from packet>",
      "sourceEvidenceSha256": "<64 lowercase hex characters from packet>",
      "reviewer": "human-reviewer",
      "reviewedAt": "2026-11-15T10:00:00+09:00",
      "decision": "INSUFFICIENT_CONTEXT",
      "errorTypes": [],
      "reason": "The quoted report alone cannot determine the claimed causal mechanism."
    }
  ]
}
```

The human comparison categories are:
- `SUPPORTED`: the reported words support the carefully conditional
  hypothesis, **without asserting external truth or future market returns**.
- `CONTRADICTED`: the hypothesis reverses or misstates a reported claim.
- `UNRELATED`: the cited text does not establish the proposed relationship.
- `INSUFFICIENT_CONTEXT`: the human could not determine which is true,
  and this item is excluded from confusion-based metrics.

For contradicted/unrelated cases require at least one independently
identified `errorTypes` code, from:
`ACTOR_MISMATCH`, `POLARITY_REVERSAL`, `UNRELATED_EVIDENCE`,
`CONDITION_OMITTED`, `NEGATION_OMITTED`, `TIME_MISMATCH`,
`UNSUPPORTED_CAUSALITY`, `OTHER`.

## Evaluation (no writes)

```sh
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_news_semantic_gold.py
market-event-news-semantic-gold --gold /path/to/private-human-gold.json
```

For a different local archive, use `--source-db` and `--journal-db`.
The CLI prints a JSON evaluation to stdout. It does not write either
database or any results file, contact Ollama, download new sources, alter
previous model versions, schedule jobs, or publish signals.

The run fails closed on duplicate IDs, missing ledger entries, bad
identity hashes, changed sources, invalid adjudication, or orphaned review
records. There are no inferred or auto-filled human labels.

`confusion` compares whether **any** lexical warning appeared with
human-reported CONTRADICTED/UNRELATED cases; warning presence is not proof
of a true semantic error. `falsePositive` and `falseNegative` matter
more than a single accuracy number. Precision and recall remain `null`
for fewer than ten scored reviews; ten samples are *not* sufficient to
claim statistical validity, and larger, independently assembled sets are
necessary. A `SUPPORTED` review never establishes independent source
authenticity or market impact. All recorded reviewers are self-attested.

## Next operational evidence required

Collect a broad, independently reviewed, representative sample of real
Korean and English news, including important **and unimportant** items,
with claim/actor/polarity/qualifier/time reasons and disagreement reviews.
Audit false alarms and misses per source/type before changing model
behavior. The original code's lexical triage never auto-approves a news
impact or generates a trading prediction. Only report scores from actual
human-provided labels.
