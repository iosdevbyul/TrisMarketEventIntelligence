# News pilot semantic review integration

The established two-stage news research pilot stores immutable original model
signals in a separate evidence journal. PR #111 added human-provided follow-up
citation notes; PR #112 added a read-only contextual review helper. This change
connects that helper to the already audited research evidence packet.

## What the reader now sees

Run the existing CLI (no new workflow or schedule):

```sh
python -m tris_market_event_intelligence.news_pilot_review_packet_cli \
  --evidence-id 1
```

The packet now has a `semanticContextReview` object:
- `reviewTarget=CONDITIONAL_MECHANISM` — the original archived conditional
  explanation, not a new or modified inference.
- `reviewCodes` — heuristic warnings for negation, conditional language,
  temporal qualifiers around the archived verbatim quote, or over-certainty.
- `scope=BOUNDED_ORIGINAL_SOURCE_CONTEXT` and
  `sourceContextTruncated` — transparent about the limited source window.
- `status=REQUIRES_HUMAN_REVIEW`,
  `semanticallyVerified=false`, `independentlyVerified=false` and
  `published=false` even if no warnings are emitted.

The full journal audit runs before producing this packet, including verification
of stored signal/source hashes and human-checklist entries. The research source
DB, original journal, model/version records, market hypotheses, any existing
manual review decisions, and autonomous schedules are unchanged.

## Validation

```sh
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_impact_semantic_review.py tests/test_news_pilot_review_ledger.py
python -m pytest -q
```

The lexical warnings are not independent verification or proven contradiction.
No warnings do not imply the reported claims, causal mechanism, market direction,
intensity or stock returns are accurate. Manual citation URL/excerpt entries
remain self-attested rather than fetched or independently certified.

Next: adjudicated human-gold claims with explicit review outcomes and
conflicting-source analysis. This integration provides observations to the
reviewer but never silently upgrades an approval or disseminates research.
