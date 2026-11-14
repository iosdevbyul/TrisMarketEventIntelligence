"""Versioned, opt-in rule preview for real source-linked event wording.

This is deliberately NOT the default EventProcessor extractor. Existing
multi-v2 checkpoints, event IDs and training eligibility remain unchanged.
"""
from __future__ import annotations

import re

from tris_market_event_intelligence.events import PolicyEvent, clean_text
from tris_market_event_intelligence.multi_events import (
    ClassifiedEvent,
    extract_news_event,
)

PREVIEW_EXTRACTOR_VERSION = "multi-v3-preview"

# Requires both an outage/disruption AND an explicit recovery cue: a product
# being "restored" alone is not a service incident, and an outage mention
# alone does not imply recovery.
SERVICE_RECOVERY = re.compile(
    r"(?i)\\b(?:services?|outlook|teams|azure|cloud|network)\\b.{0,100}"
    r"\\b(?:restored|recovered|back online)\\b.{0,100}"
    r"\\b(?:outages?|disruptions?)\\b"
    r"|\\b(?:outages?|disruptions?)\\b.{0,100}"
    r"\\b(?:services?|outlook|teams|azure|cloud|network)\\b.{0,100}"
    r"\\b(?:restored|recovered|back online)\\b"
)
# Materially sized monetary sanctions, not vague allegations, smaller traffic
# fines or an unquantified policy discussion.
SANCTION = re.compile(
    r"(?i)\\b(?:fines?|fined|penaliz(?:es|ed))\\b.{0,85}"
    r"(?:[€$£]\\s?\\d+(?:[.,]\\d+)?\\s?(?:m|bn|million|billion)"
    r"|\\d+(?:[.,]\\d+)?\\s?(?:million|billion)\\s?"
    r"(?:euros?|dollars?|pounds?))\\b?"
)
NEGATION = re.compile(
    r"(?i)\\b(?:not|never|did not|has not|was not|weren't|denies|denied)\\b"
    r".{0,25}\\b(?:fines?|fined|penaliz(?:es|ed)|restored|recovered)\\b"
)


def preview_news_event_v3(
    source: str, title: str, summary: str | None,
) -> ClassifiedEvent | None:
    """Return a review-only hypothetical v3 event, never a verified event."""
    # Existing rules always take precedence. A new preview does not silently
    # reinterpret the status or category of any v2 event.
    old = extract_news_event(source, title, summary)
    if old is not None:
        return old
    # A title carries a concrete event claim; no cross-clause keyword mixing
    # from a loosely related summary is allowed for these new cues.
    headline = clean_text(title)
    if NEGATION.search(headline):
        return None
    if SERVICE_RECOVERY.search(headline):
        category, tag = "SERVICE_INCIDENT", "RECOVERY"
    elif SANCTION.search(headline):
        category, tag = "REGULATION", "MONETARY_PENALTY"
    else:
        return None
    return ClassifiedEvent(
        event=PolicyEvent(
            category, "UNVERIFIED", None, None, headline, PREVIEW_EXTRACTOR_VERSION,
        ),
        claim_kind="REPORTED",
        tags=(tag,),
        verification_note="THIRD_PARTY_REPORT_REVIEW_REQUIRED",
    )
