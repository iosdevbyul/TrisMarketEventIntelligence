# Phase 7-D2 — Independent semantic review triage (initial implementation)

A verbatim `evidence_excerpt` match does not prove that an AI rationale
preserves the article's polarity, condition, timing, actor, or economic meaning.

`impact_semantic_review.semantic_review_packet` offers a **read-only**,
bounded review aid for the human reviewer. It highlights negation, conditional,
and temporal expressions within a 200-character context window surrounding
the exact quote, including qualifiers outside the selected quotation.
Explicit absolute certainty in the rationale is also flagged. The result
always states `REQUIRES_HUMAN_REVIEW`, `semanticallyVerified=false`, and
`independentlyVerified=false`, even when no flags are found.

This is **not** a general semantic-entailment model: lexical markers can miss
paraphrases, anaphora, sarcasm, actors, contradictory numbers, and long-range
context. A flag is a review question, not a proven error. Zero flags is NOT
a pass. The existing provider validation, `market-impact-v8`, production
tables, schedules, and stored model versions are unchanged. This is an
initial independently testable triage step, not completion of the full
Trello 7-D2 acceptance criteria or independent factual verification.

For local use:

```python
from tris_market_event_intelligence.impact_semantic_review import semantic_review_packet

packet = semantic_review_packet(
    source_text="The agency did not approve the merger.",
    evidence_excerpt="approve the merger",
    rationale="The merger is confirmed and will boost profits.",
)
print(packet["status"], packet["reviewCodes"])
```

Test on Mac with:
```sh
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_impact_semantic_review.py
python -m pytest -q
```

Next: human-labelled entailment/contradiction cases for evidence and market
mechanism, structured adjudication reasons, and a separate audited review
record that never retroactively overwrites old model results.
