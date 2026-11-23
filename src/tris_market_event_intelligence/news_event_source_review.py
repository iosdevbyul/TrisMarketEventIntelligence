"""Conservative independent evidence checks on *model-proposed* event types.

These flags are not ground-truth classifications. They prevent a known
source/title contradiction from being silently treated as a market signal.
The original proposed eventKind is always retained for manual review.
"""
from __future__ import annotations

import re


def assess_event(article: dict, body: str, event: dict) -> dict:
    title = re.sub(r"\s+", " ", str(article.get("title") or "")).casefold()
    excerpt = re.sub(r"\s+", " ", body[:12000]).casefold()
    kind = event["eventKind"]
    flags: list[str] = []

    # Require independent hints in both the document heading and source body;
    # words like "schedule" in a policy discussion alone are insufficient.
    schedule_heading = (
        bool(re.search(r"\b(schedule|calendar|meeting dates)\b", title))
        and bool(re.search(r"\b(meeting|committee|fomc)\b", title))
    )
    schedule_source = bool(re.search(
        r"\b(revised|tentative|updated|released|announced)\b.{0,95}"
        r"\b(schedule|calendar|meeting dates)\b"
        r"|\b(schedule|calendar|meeting dates)\b.{0,95}"
        r"\b(meeting|committee|fomc)\b",
        excerpt[:2200],
    ))
    if schedule_heading and schedule_source and kind != "SCHEDULE_ONLY":
        flags.append("SOURCE_SCHEDULE_KIND_CONFLICT")

    # FOMC minutes can themselves be newly published but discuss historical
    # policy decisions and observed movements. Do not assert a new action
    # merely from that retrospective document form.
    minutes_heading = bool(re.search(
        r"\bminutes of (the )?(federal open market committee|board|"
        r"discount rate meetings)\b",
        title,
    ))
    if minutes_heading and kind == "POLICY_DECISION":
        flags.append("MINUTES_DECISION_TIMING_REVIEW")

    return {
        "status": "REVIEW_REQUIRED" if flags else "UNVERIFIED",
        "flags": flags,
        "sourceTypeHints": {
            "scheduleHeadingAndBody": bool(schedule_heading and schedule_source),
            "minutesDocument": minutes_heading,
        },
        "marketImpactEvidenceVerified": False,
    }
